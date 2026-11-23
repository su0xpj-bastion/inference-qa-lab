from __future__ import annotations

# Parser for nccl-tests stdout (all_reduce_perf, all_gather_perf, alltoall_perf,
# sendrecv_perf, ...). Handles the current 13-column table (size count type redop
# root, then time/algbw/busbw/#wrong for out-of-place and in-place), the older
# 12-column table without `root`, and `#wrong` printed as N/A when run with -c 0.

import re
from dataclasses import dataclass, field
from pathlib import Path

from fabval.model import busbw_factor

ENCODING = "utf-8"

COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "alltoall", "sendrecv", "broadcast", "reduce")
HEADER_RE = re.compile(r"^#\s*nThread\s+(?P<nthread>\d+)\s+nGpus\s+(?P<ngpus>\d+)\s+minBytes\s+(?P<min>\d+)\s+maxBytes\s+(?P<max>\d+)")
ITERS_RE = re.compile(r"warmup iters:\s*(?P<warmup>\d+)\s+iters:\s*(?P<iters>\d+)")
DEVICE_RE = re.compile(
    r"^#\s+Rank\s+(?P<rank>\d+)\s+(?:Group\s+\d+\s+)?Pid\s+(?P<pid>\d+)\s+on\s+(?P<host>\S+)\s+device\s+(?P<dev>\d+)\s+\[(?P<bus>[^\]]+)\]\s+(?P<name>.+?)\s*$"
)
AVG_RE = re.compile(r"^#\s*Avg bus bandwidth\s*:\s*(?P<v>[0-9.]+)")
OOB_RE = re.compile(r"^#\s*Out of bounds values\s*:\s*(?P<n>\d+)\s+(?P<status>\w+)")


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Measurement:
    time_us: float
    algbw: float
    busbw: float
    wrong: int | None  # None when validation was off (-c 0)


@dataclass(frozen=True)
class Row:
    size: int
    count: int
    dtype: str
    redop: str
    root: int
    oop: Measurement
    ip: Measurement

    @property
    def best_busbw(self) -> float:
        return max(self.oop.busbw, self.ip.busbw)

    @property
    def wrong(self) -> int:
        return (self.oop.wrong or 0) + (self.ip.wrong or 0)


@dataclass
class NcclTestsResult:
    collective: str | None
    rows: list[Row] = field(default_factory=list)
    devices: list[dict] = field(default_factory=list)
    header: dict = field(default_factory=dict)
    avg_busbw: float | None = None
    oob_values: int | None = None
    oob_ok: bool | None = None

    @property
    def nranks(self) -> int:
        return len(self.devices)

    @property
    def hosts(self) -> list[str]:
        return sorted({d["host"] for d in self.devices})

    @property
    def complete(self) -> bool:
        """nccl-tests prints the Avg line last; without it the run died or was cut off."""
        return self.avg_busbw is not None and bool(self.rows)

    @property
    def total_wrong(self) -> int:
        return sum(r.wrong for r in self.rows)

    def peak_busbw(self, min_bytes: int) -> float | None:
        vals = [r.best_busbw for r in self.rows if r.size >= min_bytes]
        return max(vals) if vals else None

    def small_times_us(self, max_bytes: int) -> list[float]:
        return [min(r.oop.time_us, r.ip.time_us) for r in self.rows if r.size <= max_bytes]

    def consistency_problems(self, rel_tol: float = 0.02, abs_tol: float = 0.02) -> list[str]:
        """busbw must equal algbw x factor(collective, nranks); algbw must equal size/time.

        A mismatch means the wrong collective was assumed, the rank count is off, or the
        file was edited by hand. All three make every downstream gate meaningless.
        """
        problems: list[str] = []
        if not self.collective or not self.nranks:
            return problems
        factor = busbw_factor(self.collective, self.nranks)
        for r in self.rows:
            for tag, m in (("oop", r.oop), ("ip", r.ip)):
                want_bus = m.algbw * factor
                if abs(m.busbw - want_bus) > max(abs_tol, rel_tol * want_bus):
                    problems.append(f"size {r.size} {tag}: busbw {m.busbw} != algbw {m.algbw} x {factor:.4f}")
                if m.time_us > 0:
                    want_alg = r.size / m.time_us / 1e3
                    if abs(m.algbw - want_alg) > max(abs_tol, rel_tol * want_alg):
                        problems.append(f"size {r.size} {tag}: algbw {m.algbw} != size/time {want_alg:.2f}")
        return problems


def _wrong(tok: str) -> int | None:
    if tok.upper() == "N/A":
        return None
    return int(float(tok))  # older builds print the max error as a float, e.g. 0e+00


def _measurement(toks: list[str]) -> Measurement:
    return Measurement(time_us=float(toks[0]), algbw=float(toks[1]), busbw=float(toks[2]), wrong=_wrong(toks[3]))


def infer_collective(name: str) -> str | None:
    base = Path(name).name
    for c in sorted(COLLECTIVES, key=len, reverse=True):
        if c in base:
            return c
    return None


def parse(text: str, collective: str | None = None) -> NcclTestsResult:
    result = NcclTestsResult(collective=collective)
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            if m := HEADER_RE.match(line):
                result.header = {k: int(v) for k, v in m.groupdict().items()}
                if it := ITERS_RE.search(line):
                    result.header["warmup"] = int(it["warmup"])
                    result.header["iters"] = int(it["iters"])
            elif m := DEVICE_RE.match(line):
                result.devices.append({
                    "rank": int(m["rank"]), "host": m["host"], "device": int(m["dev"]),
                    "bus_id": m["bus"], "name": m["name"],
                })
            elif m := AVG_RE.match(line):
                result.avg_busbw = float(m["v"])
            elif m := OOB_RE.match(line):
                result.oob_values = int(m["n"])
                result.oob_ok = m["status"].upper() == "OK"
            continue
        toks = line.split()
        if not toks[0].isdigit():
            # nccl-tests prints NCCL failures inline, e.g. "node-01: Test NCCL failure ..."
            continue
        try:
            if len(toks) == 13:
                row = Row(int(toks[0]), int(toks[1]), toks[2], toks[3], int(toks[4]),
                          _measurement(toks[5:9]), _measurement(toks[9:13]))
            elif len(toks) == 12:
                row = Row(int(toks[0]), int(toks[1]), toks[2], toks[3], -1,
                          _measurement(toks[4:8]), _measurement(toks[8:12]))
            else:
                raise ParseError(f"line {lineno}: expected 12 or 13 columns, got {len(toks)}: {line!r}")
        except ValueError as exc:
            if isinstance(exc, ParseError):
                raise
            raise ParseError(f"line {lineno}: {exc}: {line!r}") from exc
        result.rows.append(row)
    return result


def parse_file(path: Path, collective: str | None = None) -> NcclTestsResult:
    with path.open("r", encoding=ENCODING) as fh:
        return parse(fh.read(), collective or infer_collective(path.name))
