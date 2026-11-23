from __future__ import annotations

# Parser for `nvidia-smi topo -m`.
#
# The matrix is tab separated. Columns are GPU0..N, NIC0..M, then CPU Affinity,
# NUMA Affinity and (newer drivers) GPU NUMA ID. A "NIC Legend" block at the end
# maps NIC0 -> mlx5_0. Cells are X (self), NV# (# NVLinks), PIX, PXB, PHB, NODE,
# SYS. Links are keyed by device name (GPU0, mlx5_0) so the affinity gate can
# look up a rail by HCA name.

import re
from pathlib import Path

from bringup.model import Topology
from bringup.parsers.common import ENCODING

_NIC_LEGEND = re.compile(r"^\s*(NIC\d+):\s*(\S+)\s*$")


def parse_topo_matrix(text: str) -> Topology:
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    header_idx = next((i for i, ln in enumerate(lines) if ln.lstrip("\t ").startswith("GPU0")), None)
    if header_idx is None:
        raise ValueError("nvidia-smi topo -m header row not found")
    header = [c.strip() for c in lines[header_idx].split("\t") if c.strip()]
    legend = {m.group(1): m.group(2) for ln in lines if (m := _NIC_LEGEND.match(ln))}

    def name(col: str) -> str:
        return legend.get(col, col)

    devices = [c for c in header if re.fullmatch(r"(GPU|NIC)\d+", c)]
    numa_col = header.index("NUMA Affinity") if "NUMA Affinity" in header else None
    links: dict[tuple[str, str], str] = {}
    gpu_numa: dict[str, int | None] = {}
    for ln in lines[header_idx + 1 :]:
        if not ln.strip() or ln.startswith("Legend"):
            break
        cells = [c.strip() for c in ln.split("\t")]
        cells = [c for c in cells if c != ""]
        row = cells[0]
        if not re.fullmatch(r"(GPU|NIC)\d+", row):
            continue
        for j, col in enumerate(devices):
            if 1 + j < len(cells):
                links[(name(row), name(col))] = cells[1 + j]
        if row.startswith("GPU") and numa_col is not None and 1 + numa_col < len(cells):
            raw = cells[1 + numa_col]
            gpu_numa[row] = int(raw) if raw.isdigit() else None
    return Topology(
        gpus=[c for c in devices if c.startswith("GPU")],
        nics=[name(c) for c in devices if c.startswith("NIC")],
        links=links,
        gpu_numa=gpu_numa,
    )


def load(path: Path) -> Topology:
    with path.open("r", encoding=ENCODING) as fh:
        return parse_topo_matrix(fh.read())
