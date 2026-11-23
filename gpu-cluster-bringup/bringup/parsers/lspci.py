from __future__ import annotations

# Parser for `lspci -vv` excerpts (run as root so LnkCap/LnkSta are visible).
#
#   18:00.0 3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev a1)
#           NUMA node: 0
#           LnkCap: Port #0, Speed 32GT/s, Width x16, ASPM not supported
#           LnkSta: Speed 32GT/s, Width x8 (downgraded)
#
# lspci is the vendor-neutral PCIe view: it still lists a GPU that has fallen
# off the driver, and it is where AMD link width comes from.

import re
from pathlib import Path

from bringup.model import PciDevice
from bringup.parsers.common import ENCODING, norm_bdf

_HEAD = re.compile(r"^((?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\s+(.*)$")
_NUMA = re.compile(r"^\s+NUMA node:\s*(-?\d+)")
_LNK = re.compile(r"^\s+Lnk(Cap|Sta):.*?Speed ([\d.]+)GT/s.*?Width x(\d+)")


def parse_lspci_vv(text: str) -> list[PciDevice]:
    devices: list[PciDevice] = []
    dev: PciDevice | None = None
    for line in text.splitlines():
        if m := _HEAD.match(line):
            dev = PciDevice(bdf=norm_bdf(m.group(1)), description=m.group(2).strip())
            devices.append(dev)
        elif dev is None:
            continue
        elif m := _NUMA.match(line):
            node = int(m.group(1))
            dev.numa_node = None if node < 0 else node
        elif m := _LNK.match(line):
            speed, width = float(m.group(2)), int(m.group(3))
            if m.group(1) == "Cap":
                dev.speed_cap_gts, dev.width_cap = speed, width
            else:
                dev.speed_sta_gts, dev.width_sta = speed, width
    if not devices:
        raise ValueError("no PCI device headers in lspci output")
    return devices


def is_gpu(dev: PciDevice) -> bool:
    d = dev.description
    return ("3D controller" in d or "Display controller" in d or "Processing accelerators" in d) and (
        "NVIDIA" in d or "AMD" in d or "Advanced Micro Devices" in d
    )


def load(path: Path) -> list[PciDevice]:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_lspci_vv(fh.read())
