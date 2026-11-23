from __future__ import annotations

# Parser for the text table printed by `dcgmi diag -r 3`.
#
# Shape (DCGM 3.x):
#   +-----  Hardware  ----------+-------------------------------------------+
#   | GPU Memory                | Pass - All                                |
#   | Targeted Power            | Pass - GPUs: 0, 1, 2, 3, 4, 6, 7          |
#   |                           | Fail - GPU: 5                             |
#   | Warning                   | GPU 5 Error: ... power ... below target   |
#
# A continuation row (blank name) belongs to the test above it. "Warning" and
# "Info" rows are messages for the test above them, not tests.

import re
from pathlib import Path

from bringup.model import DiagTest
from bringup.parsers.common import ENCODING

_SECTION = re.compile(r"^[+|]-+\s+([A-Za-z ]+?)\s+-+")
_ROW = re.compile(r"^\|\s(.*?)\s*\|\s(.*?)\s*\|\s*$")
_RESULT = re.compile(r"^(Pass|Fail|Warn|Skip)\b(?:\s*-\s*(.*))?$")
_GPUS = re.compile(r"GPUs?:\s*([\d,\s]+)")
_MESSAGE_ROWS = ("Warning", "Info")
_METADATA = ("Metadata",)


def parse_dcgmi_diag(text: str) -> list[DiagTest]:
    tests: list[DiagTest] = []
    section = ""
    current: DiagTest | None = None
    for line in text.splitlines():
        sec = _SECTION.match(line)
        if sec:
            section = sec.group(1).strip()
            current = None
            continue
        row = _ROW.match(line)
        if not row or section in _METADATA or not section:
            continue
        name, value = row.group(1).strip(), row.group(2).strip()
        if name in _MESSAGE_ROWS:
            if current is not None:
                current.messages.append(value)
            continue
        res = _RESULT.match(value)
        if res is None:
            if current is not None and not name and value:
                # dcgmi word-wraps long warnings onto blank-name continuation rows.
                if current.messages:
                    current.messages[-1] += " " + value
                else:
                    current.messages.append(value)
            continue
        result, detail = res.group(1), res.group(2) or ""
        if name:
            current = DiagTest(name=name, category=section, result=result)
            tests.append(current)
        elif current is None:
            continue
        elif result == "Fail" or (result == "Warn" and current.result == "Pass"):
            current.result = result
        if result == "Fail":
            gm = _GPUS.search(detail)
            if gm:
                current.failed_gpus.extend(int(x) for x in gm.group(1).replace(",", " ").split())
    if not tests:
        raise ValueError("no dcgmi diag result rows found")
    return tests


def load(path: Path) -> list[DiagTest]:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_dcgmi_diag(fh.read())
