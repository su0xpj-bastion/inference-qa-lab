from __future__ import annotations

# Per-node gates for the node, burn-in and domain stages.
#
# Every gate is a pure function (facts, node-type spec, thresholds) -> Check.
# Gates read only the vendor-neutral NodeFacts, so the same gate runs for
# NVIDIA and AMD nodes. Each failing Check carries a reason a technician can act
# on without opening the raw capture.

import re
from collections.abc import Callable

from bringup.model import FAIL, PASS, SKIP, WARN, Check, NodeFacts
from bringup.parsers.lspci import is_gpu

_XID = re.compile(r"Xid \(PCI:([0-9a-fA-F:]+)\):\s*(\d+)")
_LINK_RANK = {"X": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}
_LINK_MEANING = {
    "PHB": "crosses the PCIe host bridge",
    "NODE": "crosses host bridges inside the NUMA node",
    "SYS": "crosses the socket interconnect (UPI/xGMI)",
}


def median(values: list[float]) -> float:
    s = sorted(values)
    if not s:
        raise ValueError("median of empty list")
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _check(stage: str, name: str, problems: list[str], evidence: dict | None = None,
           warnings: list[str] | None = None) -> Check:
    if problems:
        return Check(stage, name, FAIL, "; ".join(problems), evidence or {})
    if warnings:
        return Check(stage, name, WARN, "; ".join(warnings), evidence or {})
    return Check(stage, name, PASS, "", evidence or {})


# ------------------------------------------------------------------ node stage


def evidence_complete(f: NodeFacts, spec: dict, th: dict) -> Check:
    return _check("node", "evidence_complete", [f"missing evidence: {m}" for m in f.missing],
                  {"missing": f.missing})


def gpu_inventory(f: NodeFacts, spec: dict, th: dict) -> Check:
    want = spec["gpu"]["count"]
    problems = []
    if f.vendor != spec["vendor"]:
        problems.append(f"vendor {f.vendor} but node type {f.node_type} expects {spec['vendor']}")
    if len(f.gpus) != want:
        problems.append(f"{len(f.gpus)}/{want} GPUs visible to the driver")
    wrong = sorted({g.product for g in f.gpus if g.product != spec["gpu"]["product"]})
    if wrong:
        problems.append(f"unexpected GPU product {wrong} (want {spec['gpu']['product']})")
    on_bus = {d.bdf: d for d in f.pci if is_gpu(d)}
    seen = {g.bus_id for g in f.gpus}
    for bdf in sorted(set(on_bus) - seen):
        rev = " (rev ff: config space unreadable)" if "rev ff" in on_bus[bdf].description else ""
        problems.append(f"{bdf} is on the PCIe bus{rev} but not enumerated by the driver: "
                        "fallen off the bus, reseat/RMA")
    return _check("node", "gpu_inventory", problems,
                  {"driver_visible": len(f.gpus), "pcie_visible": len(on_bus), "expected": want})


def gpu_memory_health(f: NodeFacts, spec: dict, th: dict) -> Check:
    problems, warnings = [], []
    max_unc = th["ecc_max_uncorrectable_volatile"]
    for g in f.gpus:
        tag = f"GPU{g.index} ({g.bus_id})"
        if g.ecc_enabled is False:
            problems.append(f"{tag} ECC disabled")
        if g.ecc_pending_enabled is False:
            problems.append(f"{tag} ECC will be disabled after next reset")
        if g.ecc_uncorrectable_volatile > max_unc:
            problems.append(f"{tag} {g.ecc_uncorrectable_volatile} uncorrectable ECC errors since boot")
        if g.remap_failure:
            problems.append(f"{tag} row-remap failure: out of spare rows, RMA")
        if g.remap_pending:
            problems.append(f"{tag} row remap pending: GPU reset required, then re-run acceptance")
        if g.retirement_pending:
            problems.append(f"{tag} page retirement pending: GPU reset required")
        if g.ecc_correctable_volatile > th["ecc_max_correctable_volatile"]:
            warnings.append(f"{tag} {g.ecc_correctable_volatile} correctable ECC errors since boot")
    return _check("node", "gpu_memory_health", problems, {
        "uncorrectable": {g.index: g.ecc_uncorrectable_volatile for g in f.gpus if g.ecc_uncorrectable_volatile},
    }, warnings)


def pcie_link(f: NodeFacts, spec: dict, th: dict) -> Check:
    """Width is checked at idle; generation is not.

    GPUs drop to gen1 when idle to save power, so current_link_gen at idle is
    meaningless. Width doesn't change with power state, so a x8 at idle is a
    real down-train. For generation the useful idle check is the slot's own
    capability (max_host_link_gen) against the device's.
    """
    want_w = spec["gpu"]["pcie_width"]
    want_gen = spec["gpu"].get("pcie_gen")
    lspci_by_bdf = {d.bdf: d for d in f.pci}
    problems = []
    for g in f.gpus:
        tag = f"GPU{g.index} ({g.bus_id})"
        width = g.pcie_width_current
        if width is None and g.bus_id in lspci_by_bdf:
            width = lspci_by_bdf[g.bus_id].width_sta
        if width is None:
            problems.append(f"{tag} PCIe link width not reported")
        elif width < want_w:
            problems.append(f"{tag} PCIe link x{width}, expected x{want_w}: down-trained, reseat/check riser")
        if want_gen and g.pcie_gen_host_max is not None and g.pcie_gen_host_max < want_gen:
            problems.append(f"{tag} slot capped at gen{g.pcie_gen_host_max}, expected gen{want_gen}: BIOS setting or riser")
    for d in f.pci:
        if ("Infiniband" in d.description or "Ethernet" in d.description) and d.width_sta and d.width_cap \
                and d.width_sta < d.width_cap:
            problems.append(f"NIC {d.bdf} PCIe link x{d.width_sta} of x{d.width_cap}")
    return _check("node", "pcie_link", problems)


def gpu_config(f: NodeFacts, spec: dict, th: dict) -> Check:
    gs = spec["gpu"]
    problems, warnings = [], []
    for g in f.gpus:
        tag = f"GPU{g.index}"
        if gs.get("persistence_required") and g.persistence is False:
            problems.append(f"{tag} persistence mode off (enable nvidia-persistenced)")
        if gs.get("power_limit_w") and g.power_limit_w is not None and abs(g.power_limit_w - gs["power_limit_w"]) > 1:
            problems.append(f"{tag} power limit {g.power_limit_w:.0f} W, expected {gs['power_limit_w']} W")
        if gs.get("max_sm_clock_mhz") and g.max_sm_clock_mhz is not None and g.max_sm_clock_mhz < gs["max_sm_clock_mhz"]:
            problems.append(f"{tag} max SM clock {g.max_sm_clock_mhz} MHz, expected {gs['max_sm_clock_mhz']}")
        if g.active_throttle:
            problems.append(f"{tag} throttling at idle: {', '.join(g.active_throttle)}")
        if g.temp_c is not None and g.temp_c > th["idle_max_temp_c"]:
            warnings.append(f"{tag} idle temperature {g.temp_c:.0f} C")
    return _check("node", "gpu_config", problems, warnings=warnings)


def software_versions(f: NodeFacts, spec: dict, th: dict) -> Check:
    sw = spec["software"]
    problems = []
    for key, have in (("driver", f.driver_version), ("runtime", f.runtime_version), ("kernel", f.kernel)):
        if have != sw[key]:
            problems.append(f"{key} {have} != golden {sw[key]}")
    vbios = sorted({g.vbios for g in f.gpus if g.vbios not in spec["gpu"]["vbios"]}, key=str)
    if vbios:
        problems.append(f"VBIOS {vbios} not in golden {spec['gpu']['vbios']}")
    fw_want = spec["fabric"].get("firmware")
    fw = sorted({h.firmware for h in _compute_hcas(f, spec) if fw_want and h.firmware != fw_want}, key=str)
    if fw:
        problems.append(f"HCA firmware {fw} != golden {fw_want}")
    return _check("node", "software_versions", problems, {
        "driver": f.driver_version, "runtime": f.runtime_version, "kernel": f.kernel,
    })


def _compute_hcas(f: NodeFacts, spec: dict) -> list:
    kind = spec["fabric"]["kind"]
    if kind == "efa":
        return [h for h in f.hcas if h.kind == "efa"]
    return [h for h in f.hcas if h.kind == "ib" and any(p.link_layer == "InfiniBand" for p in h.ports)]


def fabric_ports(f: NodeFacts, spec: dict, th: dict) -> Check:
    fab = spec["fabric"]
    hcas = _compute_hcas(f, spec)
    problems = []
    if len(hcas) != fab["count"]:
        problems.append(f"{len(hcas)}/{fab['count']} compute {fab['kind'].upper()} devices")
    for h in hcas:
        for p in h.ports:
            if p.state != "Active" or p.physical_state != "LinkUp":
                problems.append(f"{h.name} port {p.number} State={p.state} PhysState={p.physical_state}: "
                                "check cable/transceiver and switch port")
            elif fab["kind"] == "ib" and p.rate_gbps != fab["rate_gbps"]:
                problems.append(f"{h.name} port {p.number} rate {p.rate_gbps} Gb/s, expected {fab['rate_gbps']}")
    return _check("node", "fabric_ports", problems, {"compute_hcas": [h.name for h in hcas]})


def gpu_nic_affinity(f: NodeFacts, spec: dict, th: dict) -> Check:
    """Every GPU needs its own compute NIC within the allowed PCIe distance.

    Matching is by topology, not by index, so a renumbered GPU (one fell off the
    bus) doesn't create false affinity failures; that is gpu_inventory's job.
    """
    if f.topology is None:
        return Check("node", "gpu_nic_affinity", FAIL, "no topology evidence")
    allowed = set(spec["affinity"]["allowed"])
    nics = [h.name for h in _compute_hcas(f, spec)]
    taken: set[str] = set()
    pairs, problems = {}, []
    for gpu in f.topology.gpus:
        ranked = sorted(
            ((f.topology.link(gpu, n) or "SYS", n) for n in nics),
            key=lambda ln: (_LINK_RANK.get(ln[0], 9), ln[1]),
        )
        choice = next(((ln, n) for ln, n in ranked if ln in allowed and n not in taken), None)
        if choice is None:
            best_link, best_nic = ranked[0] if ranked else ("none", "none")
            why = _LINK_MEANING.get(best_link, "")
            problems.append(f"{gpu} has no dedicated compute NIC within {'/'.join(sorted(allowed))} "
                            f"(closest {best_nic} is {best_link}{': ' + why if why else ''}); "
                            "its GPUDirect RDMA traffic would leave the local PCIe switch")
            continue
        taken.add(choice[1])
        pairs[gpu] = f"{choice[1]} ({choice[0]})"
    if problems and len(f.topology.gpus) == spec["gpu"]["count"]:
        for n in nics:
            links = {f.topology.link(g, n) or "SYS" for g in f.topology.gpus}
            if not links & allowed:
                problems.append(f"{n} is not within {'/'.join(sorted(allowed))} of any GPU: "
                                "wrong slot, or rail cabling/numbering swapped")
    return _check("node", "gpu_nic_affinity", problems, {"pairs": pairs})


NODE_GATES: list[Callable[[NodeFacts, dict, dict], Check]] = [
    evidence_complete, gpu_inventory, gpu_memory_health, pcie_link,
    gpu_config, software_versions, fabric_ports, gpu_nic_affinity,
]


# ---------------------------------------------------------------- burn-in stage


def dcgm_diag(f: NodeFacts, spec: dict, th: dict) -> Check:
    if f.vendor != "nvidia":
        return Check("burnin", "dcgm_diag", SKIP, "no DCGM on this vendor; RVS/AGFHC results not parsed")
    if f.diag is None:
        return Check("burnin", "dcgm_diag", FAIL, "no dcgmi diag -r 3 evidence")
    problems = []
    for t in f.diag:
        if t.result == "Fail":
            gpus = ",".join(str(g) for g in t.failed_gpus) or "?"
            msg = f" ({t.messages[0]})" if t.messages else ""
            problems.append(f"dcgmi diag -r 3 '{t.name}' failed on GPU {gpus}{msg}")
    warnings = [f"dcgmi '{t.name}' warned" for t in f.diag if t.result == "Warn"]
    return _check("burnin", "dcgm_diag", problems, {"tests": len(f.diag)}, warnings)


def thermal(f: NodeFacts, spec: dict, th: dict) -> Check:
    b = spec["burnin"]
    problems = []
    for g in (f.burnin or {}).get("gpus", []):
        hot = g["max_temp_c"] > b["max_temp_c"]
        throttled = sum(g["throttle_seconds"].values()) > b["max_throttle_s"]
        if hot or throttled:
            secs = ", ".join(f"{k} {v:.0f}s" for k, v in g["throttle_seconds"].items() if v)
            problems.append(f"GPU{g['index']} peaked at {g['max_temp_c']:.0f} C (limit {b['max_temp_c']})"
                            f"{', ' + secs if secs else ''}: check cold plate, fans, airflow")
    if f.burnin is None:
        problems.append("no burn-in evidence")
    return _check("burnin", "thermal", problems)


def xid_events(f: NodeFacts, spec: dict, th: dict) -> Check:
    if f.burnin is None:
        return Check("burnin", "xid_events", FAIL, "no burn-in evidence")
    critical = set(th["critical_xids"])
    problems = []
    for line in f.burnin.get("xid_lines", []):
        m = _XID.search(line)
        if m and int(m.group(2)) in critical:
            problems.append(f"Xid {m.group(2)} on PCI {m.group(1)} during burn-in: {line.split(',')[-1].strip()}")
    return _check("burnin", "xid_events", problems)


def burnin_perf(f: NodeFacts, spec: dict, th: dict) -> Check:
    if f.burnin is None:
        return Check("burnin", "burnin_perf", FAIL, "no burn-in evidence")
    floor = spec["burnin"]["min_gemm_tflops"]
    gpus = f.burnin.get("gpus", [])
    problems = [f"GPU{g['index']} {g['errors']} compute errors" for g in gpus if g["errors"]]
    problems += [f"GPU{g['index']} {g['gemm_tflops']:.0f} TFLOPS < floor {floor}" for g in gpus
                 if g["gemm_tflops"] < floor]
    if gpus:
        med = median([g["gemm_tflops"] for g in gpus])
        problems += [f"GPU{g['index']} {g['gemm_tflops']:.0f} TFLOPS is >5% below its siblings ({med:.0f})"
                     for g in gpus if g["gemm_tflops"] < 0.95 * med]
    node_tflops = median([g["gemm_tflops"] for g in gpus]) if gpus else None
    return _check("burnin", "burnin_perf", problems, {"node_median_tflops": node_tflops})


BURNIN_GATES = [dcgm_diag, thermal, xid_events, burnin_perf]


# ----------------------------------------------------------------- domain stage


def nvlink_topology(f: NodeFacts, spec: dict, th: dict) -> Check:
    """Scale-up domain: every GPU pair must be on the full NVLink/XGMI fabric."""
    su = spec["scaleup"]
    if f.topology is None:
        return Check("domain", "nvlink_topology", FAIL, "no topology evidence")
    problems = []
    gpus = f.topology.gpus
    for i, a in enumerate(gpus):
        for b in gpus[i + 1:]:
            link = f.topology.link(a, b) or "missing"
            if su["link"] == "NV":
                m = re.fullmatch(r"NV(\d+)", link)
                if not m or int(m.group(1)) < su["min_links"]:
                    problems.append(f"{a}<->{b} {link}, expected NV{su['min_links']}")
            elif link != su["link"]:
                problems.append(f"{a}<->{b} {link}, expected {su['link']}")
    if spec["gpu"].get("nvswitch_fabric"):
        for g in f.gpus:
            if g.fabric_state not in (None, "Completed"):
                problems.append(f"GPU{g.index} NVSwitch fabric state {g.fabric_state}: check nvidia-fabricmanager")
    # Collapse the per-pair list when one GPU is the common factor.
    if len(problems) > 3:
        names = [p.split(" ")[0].split("<->") for p in problems if "<->" in p]
        common = set(names[0]).intersection(*map(set, names[1:])) if names else set()
        if common:
            problems = [f"{common.pop()} has degraded links to all {len(names)} peers "
                        f"(e.g. {problems[0]}): NVLink/NVSwitch fault on that GPU's baseboard position"]
    return _check("domain", "nvlink_topology", problems)


def allreduce_single_node(f: NodeFacts, spec: dict, th: dict) -> Check:
    floor = spec["allreduce"]["single_node_min_busbw_gbps"]
    if not f.allreduce:
        return Check("domain", "allreduce_single_node", FAIL, "no single-node all-reduce evidence")
    r = f.allreduce[0]
    problems = []
    if r["wrong"]:
        problems.append(f"all_reduce_perf reported {r['wrong']} wrong results: data corruption")
    if r["peak_busbw_gbps"] < floor:
        problems.append(f"single-node all_reduce busbw {r['peak_busbw_gbps']:.0f} GB/s < {floor}")
    return _check("domain", "allreduce_single_node", problems, {"peak_busbw_gbps": r["peak_busbw_gbps"]})


DOMAIN_GATES = [nvlink_topology, allreduce_single_node]
