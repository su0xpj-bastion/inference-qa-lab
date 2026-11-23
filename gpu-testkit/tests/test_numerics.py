"""Numerics helpers: rounding to low-precision formats, ULP distance, and failure messages."""

from __future__ import annotations

import math
import struct
import sys

import pytest

from gpu_testkit.numerics import DEFAULT_TOLERANCES, assert_close, compare, quantize, ulp_distance
from gpu_testkit.numerics import _fp8_positive_values as fp8_values


class Tensorish:
    """Stands in for numpy/torch: anything with .tolist() is accepted."""

    def __init__(self, data: list[list[float]]) -> None:
        self._data = data

    def tolist(self) -> list[list[float]]:
        return self._data


def test_fp8_tables_match_the_ocp_spec() -> None:
    e4m3, e5m2 = fp8_values("fp8_e4m3"), fp8_values("fp8_e5m2")
    assert (len(e4m3), e4m3[-1], e4m3[1]) == (127, 448.0, 2.0**-9)  # e4m3fn: no inf, max 448
    assert (len(e5m2), e5m2[-1], e5m2[1]) == (124, 57344.0, 2.0**-16)  # e5m2: IEEE-style inf/NaN


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (1 + 2**-8, "bf16", 1.0),  # exact tie -> even mantissa
        (1 + 3 * 2**-8, "bf16", 1 + 2**-6),  # tie -> rounds up to even
        (65519.0, "fp16", 65504.0),  # below the overflow threshold rounds to max
        (70000.0, "fp16", math.inf),
        (500.0, "fp8", 448.0),  # saturating cast
        (-500.0, "fp8_e4m3", -448.0),
        (0.3, "fp8", 0.3125),
        (1e40, "fp32", math.inf),
    ],
)
def test_quantize(value: float, dtype: str, expected: float) -> None:
    assert quantize(value, dtype) == expected


def test_fp32_ulp_matches_bit_patterns() -> None:
    one = struct.unpack("<f", struct.pack("<I", 0x3F800000))[0]
    next_up = struct.unpack("<f", struct.pack("<I", 0x3F800001))[0]
    assert ulp_distance(one, next_up) == 1
    assert ulp_distance(-1e-45, 1e-45) == 2  # crosses zero through both subnormals
    assert ulp_distance(0.0, -0.0) == 0


def test_ulp_distance_in_low_precision() -> None:
    assert ulp_distance(1.0, 1.0078125, "bf16") == 1
    assert ulp_distance(448.0, -448.0, "fp8") == 252
    assert ulp_distance(math.nan, 1.0, "fp16") == sys.maxsize
    assert ulp_distance(math.nan, math.nan) == 0


def test_dtype_defaults_scale_with_precision() -> None:
    actual, expected = [1.0, 2.0], [1.0, 2.02]  # 1% error
    assert compare(actual, expected, dtype="bf16").ok
    assert not compare(actual, expected, dtype="fp16").ok
    assert not compare(actual, expected, dtype="fp32").ok
    assert DEFAULT_TOLERANCES["fp8_e4m3"][0] == 0.125


def test_failure_message_points_at_the_worst_element() -> None:
    actual = Tensorish([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.2]]
    with pytest.raises(AssertionError) as info:
        assert_close(actual, expected, dtype="bf16")
    text = str(info.value)
    assert "1 / 6 elements (16.67%) out of tolerance, shape=(2, 3)" in text
    assert "worst at index (1, 2): actual=6.0 expected=6.2" in text
    assert "6 ULP(s) in bf16" in text
    assert "[2, 10)" in text
    assert "<- fails" in text


def test_histogram_shows_how_widespread_the_error_is() -> None:
    report = compare([float(i) for i in range(100)], [i * (1 + 1e-5) for i in range(100)])
    assert sum(report.histogram) == 100
    assert len(report.mismatches) == 98
    assert report.render().count("<- fails") == 2


def test_nonfinite_disagreement_is_counted_separately() -> None:
    report = compare([1.0, math.inf, math.nan], [1.0, 1.0, math.nan])
    assert report.nonfinite_mismatches == 2
    assert compare([math.nan], [math.nan], equal_nan=True).ok
    assert "(non-finite)" in report.render()


def test_max_ulp_allows_a_bounded_rounding_difference() -> None:
    a, b = 1.0, 1.0 + 2**-22  # 2 fp32 ULPs, outside rtol=0 / atol=0
    assert not compare([a], [b], rtol=0, atol=0).ok
    assert compare([a], [b], rtol=0, atol=0, max_ulp=2).ok


def test_shape_and_type_errors_are_explicit() -> None:
    with pytest.raises(AssertionError, match=r"shape mismatch: actual \(2,\) vs expected \(3,\)"):
        compare([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="ragged"):
        compare([[1.0], [1.0, 2.0]], [[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="unsupported dtype"):
        compare([1.0], [1.0], dtype="int8")
    with pytest.raises(TypeError, match="expected a number"):
        compare({"a": 1.0}, [1.0])
