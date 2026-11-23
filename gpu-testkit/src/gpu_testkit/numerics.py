"""Numerics assertions that know about low-precision dtypes.

``assert_close`` uses ``|actual - expected| <= atol + rtol * |expected|`` with default
tolerances chosen per dtype, and when it fails it says *where* (worst element
coordinates), *how badly* (ULPs in the target dtype) and *how widespread* (a histogram
of error relative to the allowed tolerance). "Mismatch 0.03" is not a bug report;
"one element at (12, 7) is 4 ULPs off and everything else is under a quarter of the
tolerance" is.

Pure Python by design: it accepts nested sequences or anything with ``.tolist()``
(NumPy arrays, torch tensors), so the framework has no array dependency.
"""

from __future__ import annotations

import bisect
import itertools
import math
import struct
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

DTYPE_ALIASES = {
    "float32": "fp32",
    "float": "fp32",
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "fp8": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "float8_e5m2": "fp8_e5m2",
}

# (rtol, atol). fp32/fp16/bf16 match torch.testing.assert_close defaults. For fp8 the
# rtol is one machine epsilon (2**-mantissa_bits) and atol is the smallest normal, since
# one rounding step in fp8 is already that large.
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "fp32": (1.3e-6, 1e-5),
    "fp16": (1e-3, 1e-5),
    "bf16": (1.6e-2, 1e-5),
    "fp8_e4m3": (2.0**-3, 2.0**-6),
    "fp8_e5m2": (2.0**-2, 2.0**-14),
}

# Bucket edges for |error| / allowed-tolerance.
HISTOGRAM_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 10.0, math.inf)
HISTOGRAM_WIDTH = 30
MAX_LISTED_MISMATCHES = 5
NAN_DISTANCE = sys.maxsize


def canonical_dtype(dtype: str) -> str:
    name = DTYPE_ALIASES.get(dtype.lower(), dtype.lower())
    if name not in DEFAULT_TOLERANCES:
        raise ValueError(f"unsupported dtype {dtype!r}; use one of {sorted(DEFAULT_TOLERANCES)}")
    return name


# --------------------------------------------------------------------------- formats


@dataclass(frozen=True)
class _Fp8Format:
    exp_bits: int
    man_bits: int
    bias: int
    ieee_special: bool  # e5m2 reserves the top exponent for inf/NaN; e4m3fn only S.1111.111 is NaN


_FP8 = {
    "fp8_e4m3": _Fp8Format(exp_bits=4, man_bits=3, bias=7, ieee_special=False),
    "fp8_e5m2": _Fp8Format(exp_bits=5, man_bits=2, bias=15, ieee_special=True),
}


@cache
def _fp8_positive_values(dtype: str) -> tuple[float, ...]:
    """All finite non-negative values of an fp8 format, indexed by their 7-bit code."""
    fmt = _FP8[dtype]
    values: list[float] = []
    top_exp = (1 << fmt.exp_bits) - 1
    for code in range(1 << (fmt.exp_bits + fmt.man_bits)):
        exp = code >> fmt.man_bits
        man = code & ((1 << fmt.man_bits) - 1)
        if fmt.ieee_special and exp == top_exp:
            break
        if not fmt.ieee_special and exp == top_exp and man == (1 << fmt.man_bits) - 1:
            break
        if exp == 0:
            values.append(man * 2.0 ** (1 - fmt.bias - fmt.man_bits))
        else:
            values.append((1 + man / (1 << fmt.man_bits)) * 2.0 ** (exp - fmt.bias))
    return tuple(values)


def _fp8_code(dtype: str, magnitude: float) -> int:
    """Round |x| to the nearest fp8 code, ties to even, saturating at the max finite."""
    values = _fp8_positive_values(dtype)
    if magnitude >= values[-1]:
        return len(values) - 1
    hi = bisect.bisect_left(values, magnitude)
    if values[hi] == magnitude or hi == 0:
        return hi
    lo = hi - 1
    below, above = magnitude - values[lo], values[hi] - magnitude
    if below < above:
        return lo
    if above < below:
        return hi
    return lo if lo % 2 == 0 else hi


def _f32_bits(x: float) -> int:
    try:
        return int(struct.unpack("<I", struct.pack("<f", x))[0])
    except (OverflowError, struct.error):
        return 0xFF800000 if x < 0 else 0x7F800000


def _bf16_bits(x: float) -> int:
    bits = _f32_bits(x)
    if math.isnan(x):
        return 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _f16_bits(x: float) -> int:
    try:
        return int(struct.unpack("<H", struct.pack("<e", x))[0])
    except (OverflowError, struct.error):
        return 0xFC00 if x < 0 else 0x7C00


def quantize(x: float, dtype: str) -> float:
    """Round a Python float to the nearest value representable in ``dtype``."""
    name = canonical_dtype(dtype)
    if math.isnan(x):
        return math.nan
    if name == "fp32":
        return float(struct.unpack("<f", struct.pack("<I", _f32_bits(x)))[0])
    if name == "fp16":
        return float(struct.unpack("<e", struct.pack("<H", _f16_bits(x)))[0])
    if name == "bf16":
        return float(struct.unpack("<f", struct.pack("<I", _bf16_bits(x) << 16))[0])
    if math.isinf(x):
        return math.copysign(_fp8_positive_values(name)[-1], x)
    value = _fp8_positive_values(name)[_fp8_code(name, abs(x))]
    return math.copysign(value, x)


def _ordinal(x: float, dtype: str) -> int:
    """Map a value to an integer so adjacent representable values differ by 1."""
    if dtype in _FP8:
        code = _fp8_code(dtype, abs(x)) if not math.isinf(x) else len(_fp8_positive_values(dtype)) - 1
        return -code if math.copysign(1.0, x) < 0 else code
    bits, sign_bit = {
        "fp32": (_f32_bits, 0x80000000),
        "fp16": (_f16_bits, 0x8000),
        "bf16": (_bf16_bits, 0x8000),
    }[dtype]
    raw = bits(x)
    return -(raw & (sign_bit - 1)) if raw & sign_bit else raw


def ulp_distance(a: float, b: float, dtype: str = "fp32") -> int:
    """How many representable ``dtype`` values apart ``a`` and ``b`` are after rounding.

    +0 and -0 are 0 ULPs apart. Any NaN is maximally distant (``sys.maxsize``).
    """
    name = canonical_dtype(dtype)
    if math.isnan(a) or math.isnan(b):
        return 0 if (math.isnan(a) and math.isnan(b)) else NAN_DISTANCE
    return abs(_ordinal(a, name) - _ordinal(b, name))


# --------------------------------------------------------------------------- comparison


def _flatten(value: Any) -> tuple[list[float], tuple[int, ...]]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, (int, float)):
        return [float(value)], ()
    if not isinstance(value, Sequence):
        raise TypeError(f"expected a number, a nested sequence, or an array with .tolist(); got {type(value).__name__}")
    if len(value) == 0:
        return [], (0,)
    parts = [_flatten(item) for item in value]
    inner = parts[0][1]
    if any(shape != inner for _, shape in parts):
        raise ValueError("ragged input: sub-sequences have different shapes")
    flat: list[float] = []
    for items, _ in parts:
        flat.extend(items)
    return flat, (len(value), *inner)


def _coords(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords: list[int] = []
    for size in reversed(shape):
        flat_index, rem = divmod(flat_index, size)
        coords.append(rem)
    return tuple(reversed(coords))


@dataclass(frozen=True)
class Mismatch:
    index: tuple[int, ...]
    actual: float
    expected: float
    abs_err: float
    ratio: float  # abs_err / allowed; inf for nan/inf disagreements
    ulps: int


@dataclass(frozen=True)
class CloseReport:
    dtype: str
    rtol: float
    atol: float
    shape: tuple[int, ...]
    total: int
    mismatches: tuple[Mismatch, ...]
    histogram: tuple[int, ...]  # counts per HISTOGRAM_EDGES bucket
    nonfinite_mismatches: int
    max_abs_err: float
    max_ratio: float

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def worst(self) -> Mismatch | None:
        """The finite mismatch furthest outside tolerance; a non-finite one only if that's all there is."""
        finite = [m for m in self.mismatches if math.isfinite(m.ratio)]
        return max(finite or self.mismatches, key=lambda m: m.ratio, default=None)

    def render(self) -> str:
        bad = len(self.mismatches)
        pct = 100.0 * bad / self.total if self.total else 0.0
        lines = [
            f"assert_close failed ({self.dtype}, rtol={self.rtol:g}, atol={self.atol:g}): "
            f"{bad} / {self.total} elements ({pct:.2f}%) out of tolerance, shape={self.shape}"
        ]
        worst = self.worst()
        if worst is not None and not math.isfinite(worst.ratio):
            lines.append(
                f"  worst at index {worst.index}: actual={worst.actual!r} expected={worst.expected!r} (non-finite)"
            )
        elif worst is not None:
            rel = worst.abs_err / abs(worst.expected) if worst.expected else math.inf
            lines.append(
                f"  worst at index {worst.index}: actual={worst.actual!r} expected={worst.expected!r} "
                f"|diff|={worst.abs_err:.6g} rel={rel:.6g} ({worst.ratio:.3g}x allowed), "
                f"{_fmt_ulps(worst.ulps)} in {self.dtype}"
            )
        if worst is not None:
            listed = ", ".join(str(m.index) for m in self.mismatches[:MAX_LISTED_MISMATCHES])
            more = f" (+{bad - MAX_LISTED_MISMATCHES} more)" if bad > MAX_LISTED_MISMATCHES else ""
            lines.append(f"  first mismatches: {listed}{more}")
        lines.append("  |error| / allowed tolerance, all elements:")
        peak = max(self.histogram, default=0) or 1
        for (lo, hi), count in zip(_buckets(), self.histogram, strict=True):
            bar = "#" * max(1 if count else 0, round(HISTOGRAM_WIDTH * count / peak))
            marker = "  <- fails" if lo >= 1.0 and count else ""
            lines.append(
                f"    [{lo:g}, {hi:g}){'':<{9 - len(f'{lo:g}, {hi:g}')}} {bar:<{HISTOGRAM_WIDTH}} {count}{marker}"
            )
        lines.append(f"    nan/inf mismatch {'':<{HISTOGRAM_WIDTH - 6}} {self.nonfinite_mismatches}")
        return "\n".join(lines)


def _fmt_ulps(ulps: int) -> str:
    return "NaN-distant" if ulps == NAN_DISTANCE else f"{ulps} ULP(s)"


def _buckets() -> Iterable[tuple[float, float]]:
    return itertools.pairwise(HISTOGRAM_EDGES)


def _bucket_of(ratio: float) -> int:
    return min(bisect.bisect_right(HISTOGRAM_EDGES, ratio) - 1, len(HISTOGRAM_EDGES) - 2)


def compare(
    actual: Any,
    expected: Any,
    *,
    dtype: str = "fp32",
    rtol: float | None = None,
    atol: float | None = None,
    max_ulp: int | None = None,
    equal_nan: bool = False,
) -> CloseReport:
    """Compare element-wise and return a report; never raises on mismatch."""
    name = canonical_dtype(dtype)
    default_rtol, default_atol = DEFAULT_TOLERANCES[name]
    rtol = default_rtol if rtol is None else rtol
    atol = default_atol if atol is None else atol
    got, got_shape = _flatten(actual)
    want, want_shape = _flatten(expected)
    if got_shape != want_shape:
        raise AssertionError(f"shape mismatch: actual {got_shape} vs expected {want_shape}")
    histogram = [0] * (len(HISTOGRAM_EDGES) - 1)
    mismatches: list[Mismatch] = []
    nonfinite = 0
    max_abs = 0.0
    max_ratio = 0.0
    for flat_index, (a, e) in enumerate(zip(got, want, strict=True)):
        if not (math.isfinite(a) and math.isfinite(e)):
            same = a == e or (equal_nan and math.isnan(a) and math.isnan(e))
            if not same:
                nonfinite += 1
                mismatches.append(
                    Mismatch(
                        _coords(flat_index, got_shape),
                        a,
                        e,
                        math.inf,
                        math.inf,
                        ulp_distance(a, e, name),
                    )
                )
            continue
        abs_err = abs(a - e)
        allowed = atol + rtol * abs(e)
        ratio = abs_err / allowed if allowed else (0.0 if abs_err == 0 else math.inf)
        histogram[_bucket_of(ratio)] += 1
        max_abs = max(max_abs, abs_err)
        max_ratio = max(max_ratio, ratio)
        if ratio <= 1.0:
            continue
        ulps = ulp_distance(a, e, name)
        if max_ulp is not None and ulps <= max_ulp:
            continue
        mismatches.append(Mismatch(_coords(flat_index, got_shape), a, e, abs_err, ratio, ulps))
    return CloseReport(
        dtype=name,
        rtol=rtol,
        atol=atol,
        shape=got_shape,
        total=len(got),
        mismatches=tuple(mismatches),
        histogram=tuple(histogram),
        nonfinite_mismatches=nonfinite,
        max_abs_err=max_abs,
        max_ratio=max_ratio,
    )


def assert_close(
    actual: Any,
    expected: Any,
    *,
    dtype: str = "fp32",
    rtol: float | None = None,
    atol: float | None = None,
    max_ulp: int | None = None,
    equal_nan: bool = False,
) -> CloseReport:
    """Assert element-wise closeness with dtype-aware defaults; returns the report on success."""
    report = compare(actual, expected, dtype=dtype, rtol=rtol, atol=atol, max_ulp=max_ulp, equal_nan=equal_nan)
    if not report.ok:
        raise AssertionError(report.render())
    return report
