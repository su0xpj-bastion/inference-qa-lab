"""Parse NVIDIA Xid events out of kernel log lines.

The driver logs GPU faults to the kernel ring buffer as::

    [ 9174.162839] NVRM: Xid (PCI:0000:3b:00): 79, pid=0, GPU has fallen off the bus.

Some Xids mean the hardware or fabric is sick (the run is not evidence about the
product); others are usually the application's own fault (a bad kernel touching
bad memory). The split below follows NVIDIA's Xid catalogue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

XID_PATTERN = re.compile(
    r"NVRM: Xid \(PCI:(?P<pci>[0-9a-fA-F:.]+)\): (?P<code>\d+)"
    r"(?:, pid=(?P<pid>[^,\s]+))?(?:, name=(?P<name>[^,]+))?,?\s*(?P<detail>.*)$"
)

# Hardware, driver or fabric faults: a failure alongside one of these is infra.
HARDWARE_XIDS: dict[int, str] = {
    48: "double-bit ECC error",
    61: "internal micro-controller breakpoint",
    62: "internal micro-controller halt",
    63: "ECC page retirement / row remap recorded",
    64: "ECC page retirement / row remap failure",
    74: "NVLink error",
    79: "GPU has fallen off the bus",
    92: "high single-bit ECC error rate",
    94: "contained ECC error",
    95: "uncontained ECC error",
    119: "GSP RPC timeout",
    120: "GSP error",
    123: "SPI PMU RPC failure",
}

# Usually caused by the workload itself: treat as product evidence.
APPLICATION_XIDS: dict[int, str] = {
    13: "graphics engine exception (often an out-of-bounds kernel access)",
    31: "GPU memory page fault (often an illegal address in a kernel)",
    43: "GPU stopped processing (application-side reset)",
    45: "preemptive cleanup after a prior error",
}


@dataclass(frozen=True)
class XidEvent:
    code: int
    pci: str
    pid: str
    detail: str

    @property
    def hardware(self) -> bool:
        return self.code in HARDWARE_XIDS

    @property
    def meaning(self) -> str:
        return HARDWARE_XIDS.get(self.code) or APPLICATION_XIDS.get(self.code) or "unclassified Xid"

    def summary(self) -> str:
        side = "hardware" if self.hardware else "application/unknown"
        return f"Xid {self.code} on PCI {self.pci} ({side}: {self.meaning})"


def parse_xids(lines: list[str]) -> list[XidEvent]:
    events: list[XidEvent] = []
    for line in lines:
        match = XID_PATTERN.search(line)
        if match is None:
            continue
        events.append(
            XidEvent(
                code=int(match.group("code")),
                pci=match.group("pci"),
                pid=match.group("pid") or "",
                detail=match.group("detail").strip(),
            )
        )
    return events
