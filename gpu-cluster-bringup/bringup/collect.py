from __future__ import annotations

# Capture directory -> NodeFacts.
#
# A capture directory is what scripts/collect.sh writes on a real node (or what
# fleetgen writes for a simulated one). Vendor is detected from which GPU tool
# output is present. A missing or unparseable file is recorded in
# NodeFacts.missing and the gate that needed it fails closed: no evidence is
# never treated as a pass.

import json
from pathlib import Path

from bringup.model import NodeFacts, Topology
from bringup.parsers import dcgm, fabric, lspci, nccl, nvidia_smi, rocm_smi, topo
from bringup.parsers.common import ENCODING, norm_bdf


def _read(path: Path) -> str:
    with path.open("r", encoding=ENCODING) as fh:
        return fh.read()


def _try(facts: NodeFacts, d: Path, name: str, fn):
    path = d / name
    if not path.exists():
        facts.missing.append(name)
        return None
    try:
        return fn(path)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        facts.missing.append(f"{name} (unparseable: {exc})")
        return None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        return json.load(fh)


def _ib_devices(path: Path) -> dict[str, str]:
    out = {}
    for line in _read(path).splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = norm_bdf(parts[1])
    return out


def _amd_topology(facts: NodeFacts, links: dict, ib_bdf: dict[str, str]) -> Topology:
    """rocm-smi has no GPU<->NIC matrix; derive one from lspci NUMA nodes.

    This is weaker than nvidia-smi topo -m (it can't see PCIe switch sharing),
    so the AMD node type allows NODE and the docs say so.
    """
    numa_by_bdf = {d.bdf: d.numa_node for d in facts.pci}
    all_links = dict(links)
    gnames = [f"GPU{g.index}" for g in facts.gpus]
    for g in facts.gpus:
        gnuma = g.numa_node if g.numa_node is not None else numa_by_bdf.get(g.bus_id)
        for nic, bdf in ib_bdf.items():
            nnuma = numa_by_bdf.get(bdf)
            all_links[(f"GPU{g.index}", nic)] = "NODE" if gnuma is not None and gnuma == nnuma else "SYS"
    return Topology(
        gpus=gnames,
        nics=list(ib_bdf),
        links=all_links,
        gpu_numa={f"GPU{g.index}": g.numa_node for g in facts.gpus},
    )


def collect_node(d: Path) -> NodeFacts:
    meta = _load_json(d / "meta.json")
    facts = NodeFacts(hostname=meta["hostname"], rack=meta["rack"], node_type=meta["node_type"])
    kernel = _try(facts, d, "uname-r.txt", _read)
    facts.kernel = kernel.strip() if kernel else None
    facts.pci = _try(facts, d, "lspci-vv.txt", lspci.load) or []
    ib_bdf = _try(facts, d, "ib-devices.txt", _ib_devices) or {}

    if (d / "fi_info-efa.txt").exists():
        facts.hcas = _try(facts, d, "fi_info-efa.txt", fabric.load_fi_info) or []
    else:
        facts.hcas = _try(facts, d, "ibstat.txt", fabric.load_ibstat) or []

    if (d / "nvidia-smi-q.xml").exists():
        facts.vendor = "nvidia"
        smi = _try(facts, d, "nvidia-smi-q.xml", nvidia_smi.load)
        if smi:
            facts.driver_version = smi["driver_version"]
            facts.runtime_version = smi["cuda_version"]
            facts.gpus = smi["gpus"]
        facts.topology = _try(facts, d, "nvidia-smi-topo.txt", topo.load)
        facts.diag = _try(facts, d, "dcgmi-diag-r3.txt", dcgm.load)
    elif (d / "rocm-smi.json").exists():
        facts.vendor = "amd"
        smi = _try(facts, d, "rocm-smi.json", rocm_smi.load)
        if smi:
            facts.driver_version = smi["driver_version"]
            facts.gpus = smi["gpus"]
        rv = _try(facts, d, "rocm-version.txt", _read)
        facts.runtime_version = rv.strip() if rv else None
        links = _try(facts, d, "rocm-smi-topo.json", rocm_smi.load_topo)
        if links is not None:
            facts.topology = _amd_topology(facts, links, ib_bdf)
        # No DCGM on AMD. RVS/AGFHC output is not parsed yet (see README).
        facts.diag = None
    else:
        facts.missing.append("nvidia-smi-q.xml or rocm-smi.json")

    # Fill GPU NUMA from lspci when the GPU tool didn't report it.
    numa_by_bdf = {p.bdf: p.numa_node for p in facts.pci}
    for g in facts.gpus:
        if g.numa_node is None:
            g.numa_node = numa_by_bdf.get(g.bus_id)

    burn = _try(facts, d, "burnin.json", _load_json)
    if burn is not None:
        xid = _try(facts, d, "dmesg-xid.txt", _read)
        burn["xid_lines"] = [ln for ln in (xid or "").splitlines() if "Xid" in ln]
    facts.burnin = burn
    ar = _try(facts, d, "all_reduce_perf.txt", nccl.load)
    facts.allreduce = [ar] if ar else None
    return facts


def collect_fleet(root: Path) -> list[NodeFacts]:
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())
    if not dirs:
        raise ValueError(f"no node capture directories (with meta.json) under {root}")
    return [collect_node(p) for p in dirs]
