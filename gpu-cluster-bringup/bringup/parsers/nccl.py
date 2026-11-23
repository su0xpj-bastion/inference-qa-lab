from __future__ import annotations

# Parser for nccl-tests output (all_reduce_perf and friends).
#
#   #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#   #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
#     8589934592    2147483648     float     sum      -1  23510.1  365.36  639.38      0  23498.4  365.55  639.71      0
#   # Out of bounds values : 0 OK
#   # Avg bus bandwidth    : 412.663
#
# Rows give out-of-place then in-place results. #wrong > 0 is a correctness
# failure no matter how fast the run was.

import re
from pathlib import Path

from bringup.parsers.common import ENCODING

_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s*$"
)
_AVG = re.compile(r"#\s*Avg bus bandwidth\s*:\s*([\d.]+)")


def parse_nccl_tests(text: str) -> dict:
    rows = []
    for line in text.splitlines():
        if m := _ROW.match(line):
            wrong_oop, wrong_ip = m.group(9), m.group(13)
            rows.append(
                {
                    "size_bytes": int(m.group(1)),
                    "busbw_gbps": max(float(m.group(8)), float(m.group(12))),
                    "wrong": sum(int(w) for w in (wrong_oop, wrong_ip) if w.isdigit()),
                }
            )
    if not rows:
        raise ValueError("no nccl-tests result rows found")
    avg = _AVG.search(text)
    return {
        "rows": rows,
        "avg_busbw_gbps": float(avg.group(1)) if avg else None,
        "peak_busbw_gbps": max(r["busbw_gbps"] for r in rows),
        "wrong": sum(r["wrong"] for r in rows),
    }


def load(path: Path) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_nccl_tests(fh.read())
