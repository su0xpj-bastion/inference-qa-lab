from __future__ import annotations

# Simulated fleet generator.
#
# Writes one capture directory per node, in the exact layout that
# scripts/collect.sh produces on a real node, with every file in the real
# tool's output format. Defects are injected into the spec before rendering,
# so a defect shows up the way it would on hardware: a missing GPU is absent
# from nvidia-smi but still on the PCIe bus, a down-trained link shows x8 in
# both nvidia-smi and lspci, and so on.

import json
import random
import textwrap
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from bringup.parsers.common import ENCODING

SEED = 20261114

# DGX/HGX H100-style layout: 8 GPUs, 8 compute rails (CX-7, NDR 400), one
# frontend Ethernet NIC. GPUs 0-3 on socket 0, 4-7 on socket 1.
H100_GPU_BDF = ["18", "2a", "3a", "5d", "9a", "ab", "ba", "db"]
H100_NIC_BDF = ["1a", "2c", "3c", "5e", "9c", "ad", "bc", "dc"]
MI300_GPU_BDF = ["05", "26", "46", "65", "85", "a6", "c6", "e5"]
MI300_NIC_BDF = ["07", "28", "48", "67", "87", "a8", "c8", "e7"]

DEFECTS = {
    # name: (stage it must be caught at, check that must catch it)
    "gpu_missing": ("node", "gpu_inventory"),
    "ecc_uncorrectable": ("node", "gpu_memory_health"),
    "pcie_downtrain": ("node", "pcie_link"),
    "driver_drift": ("node", "software_versions"),
    "ib_port_down": ("node", "fabric_ports"),
    "numa_affinity": ("node", "gpu_nic_affinity"),
    "dcgm_power_fail": ("burnin", "dcgm_diag"),
    "thermal_throttle": ("burnin", "thermal"),
    "xid_79": ("burnin", "xid_events"),
    "nvlink_degraded": ("domain", "nvlink_topology"),
    "straggler": ("cluster", "performance_outlier"),
}

# Checks that are expected to fail *as well*, downstream of the root cause, and
# must be reported after it. Six dead NVLinks on one GPU really do cut
# single-node all-reduce bandwidth; hiding that would be lying about evidence.
CONSEQUENCES = {"nvlink_degraded": ["allreduce_single_node"]}

# The default scenario: one of each defect, on fixed nodes, so reports diff cleanly.
DEFAULT_INJECTIONS = {
    "gpu-r01-n02": "gpu_missing",
    "gpu-r01-n04": "ecc_uncorrectable",
    "gpu-r01-n06": "pcie_downtrain",
    "gpu-r01-n08": "nvlink_degraded",
    "gpu-r02-n01": "driver_drift",
    "gpu-r02-n03": "ib_port_down",
    "gpu-r02-n05": "numa_affinity",
    "gpu-r02-n07": "dcgm_power_fail",
    "gpu-r03-n03": "thermal_throttle",
    "gpu-r04-n02": "straggler",
    "gpu-r04-n05": "xid_79",
    "amd-r05-n02": "ecc_uncorrectable",
}


def _h(*parts: object) -> int:
    """Stable across processes, unlike the built-in salted hash."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())


@dataclass
class GpuSpec:
    index: int
    bdf: str
    numa: int
    present: bool = True
    width: int = 16
    gen_max: int = 5
    host_gen_max: int = 5
    ecc_unc: int = 0
    remap_pending: bool = False
    nvlinks: int = 18
    max_temp_c: float = 74.0
    throttle_s: float = 0.0
    tflops: float = 0.0
    xid: int | None = None


@dataclass
class NicSpec:
    name: str
    bdf: str
    numa: int
    state: str = "Active"
    phys: str = "LinkUp"
    rate: int = 400
    layer: str = "InfiniBand"


@dataclass
class NodeSpec:
    hostname: str
    rack: str
    node_type: str
    vendor: str
    driver: str
    runtime: str
    kernel: str
    vbios: str
    nic_fw: str
    gpus: list[GpuSpec]
    nics: list[NicSpec]
    rail_override: dict[int, int] = field(default_factory=dict)  # gpu -> numa of its rail NIC
    diag_fail: dict[str, tuple[int, str]] = field(default_factory=dict)
    allreduce_busbw: float = 0.0
    defect: str | None = None


def _h100(hostname: str, rack: str, rng: random.Random) -> NodeSpec:
    gpus = [
        GpuSpec(i, f"0000:{b}:00.0", 0 if i < 4 else 1, tflops=round(rng.gauss(715.0, 4.0), 1),
                max_temp_c=round(rng.uniform(70, 79), 1))
        for i, b in enumerate(H100_GPU_BDF)
    ]
    nics = [NicSpec(f"mlx5_{i}", f"0000:{b}:00.0", 0 if i < 4 else 1) for i, b in enumerate(H100_NIC_BDF)]
    nics.append(NicSpec("mlx5_8", "0000:aa:00.0", 1, rate=100, layer="Ethernet"))
    return NodeSpec(
        hostname=hostname, rack=rack, node_type="hgx-h100-ndr", vendor="nvidia",
        driver="550.127.08", runtime="12.4", kernel="5.15.0-125-generic",
        vbios="96.00.99.00.01", nic_fw="28.39.2048", gpus=gpus, nics=nics,
        allreduce_busbw=round(rng.uniform(452.0, 468.0), 2),
    )


def _mi300x(hostname: str, rack: str, rng: random.Random) -> NodeSpec:
    gpus = [
        GpuSpec(i, f"0000:{b}:00.0", 0 if i < 4 else 1, tflops=round(rng.gauss(1180.0, 8.0), 1),
                max_temp_c=round(rng.uniform(72, 82), 1))
        for i, b in enumerate(MI300_GPU_BDF)
    ]
    nics = [NicSpec(f"mlx5_{i}", f"0000:{b}:00.0", 0 if i < 4 else 1) for i, b in enumerate(MI300_NIC_BDF)]
    return NodeSpec(
        hostname=hostname, rack=rack, node_type="mi300x-ndr", vendor="amd",
        driver="6.8.5", runtime="6.2.4", kernel="5.15.0-125-generic",
        vbios="113-M3000100-102", nic_fw="28.39.2048", gpus=gpus, nics=nics,
        allreduce_busbw=round(rng.uniform(318.0, 332.0), 2),
    )


def inject(node: NodeSpec, defect: str) -> None:
    node.defect = defect
    g = node.gpus
    if defect == "gpu_missing":
        g[4].present = False
    elif defect == "ecc_uncorrectable":
        g[3].ecc_unc = 2
        g[3].remap_pending = True
    elif defect == "pcie_downtrain":
        g[5].width = 8
    elif defect == "driver_drift":
        node.driver = "550.90.07"
    elif defect == "ib_port_down":
        nic = node.nics[4]
        nic.state, nic.phys, nic.rate = "Down", "Polling", 10
    elif defect == "numa_affinity":
        node.rail_override[6] = 0  # mlx5_6 sits on socket 0, GPU6 on socket 1
        node.nics[6].numa = 0
    elif defect == "dcgm_power_fail":
        node.diag_fail["Targeted Power"] = (
            5, "GPU 5 Error: Target power 700 W could not be reached: max observed 552 W. "
               "Check power delivery and cooling.",
        )
    elif defect == "thermal_throttle":
        g[2].max_temp_c, g[2].throttle_s = 92.0, 340.0
    elif defect == "xid_79":
        g[1].xid = 79
    elif defect == "nvlink_degraded":
        g[1].nvlinks = 12
        node.allreduce_busbw = 291.4
    elif defect == "straggler":
        for gpu in g:
            gpu.tflops = round(gpu.tflops * 0.92, 1)  # above the absolute floor
    else:
        raise ValueError(f"unknown defect {defect!r}; known: {sorted(DEFECTS)}")


def build_fleet(scenario: str = "defects", seed: int = SEED) -> list[NodeSpec]:
    rng = random.Random(seed)
    nodes = [_h100(f"gpu-r{r:02d}-n{n:02d}", f"r{r:02d}", rng) for r in range(1, 5) for n in range(1, 9)]
    nodes += [_mi300x(f"amd-r05-n{n:02d}", "r05", rng) for n in range(1, 5)]
    by_name = {n.hostname: n for n in nodes}
    if scenario == "healthy":
        return nodes
    if scenario == "defects":
        for host, defect in DEFAULT_INJECTIONS.items():
            inject(by_name[host], defect)
        return nodes
    if scenario == "leaf-down":
        # A leaf switch in rack r03 loses its downlinks: the same rail goes dark on
        # 5 of 8 nodes. Five node-level failures, one rack-level cause.
        for n in range(1, 6):
            inject(by_name[f"gpu-r03-n{n:02d}"], "ib_port_down")
        return nodes
    raise ValueError(f"unknown scenario {scenario!r}; known: healthy, defects, leaf-down")


# ---------------------------------------------------------------- renderers


def render_nvidia_smi_xml(node: NodeSpec) -> str:
    present = [g for g in node.gpus if g.present]
    parts = [
        '<?xml version="1.0" ?>',
        '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">',
        "<nvidia_smi_log>",
        "\t<timestamp>Thu Nov 12 09:14:03 2026</timestamp>",
        f"\t<driver_version>{node.driver}</driver_version>",
        f"\t<cuda_version>{node.runtime}</cuda_version>",
        f"\t<attached_gpus>{len(present)}</attached_gpus>",
    ]
    for minor, g in enumerate(present):
        bus = g.bdf.split(":")[1]
        unc_dram = g.ecc_unc
        remap_unc = 1 if g.remap_pending else 0
        temp = 33 + g.index % 3
        thermal = "Not Active"
        parts.append(f"""\t<gpu id="0000{g.bdf}">
\t\t<product_name>NVIDIA H100 80GB HBM3</product_name>
\t\t<product_brand>NVIDIA</product_brand>
\t\t<product_architecture>Hopper</product_architecture>
\t\t<persistence_mode>Enabled</persistence_mode>
\t\t<mig_mode>
\t\t\t<current_mig>Disabled</current_mig>
\t\t\t<pending_mig>Disabled</pending_mig>
\t\t</mig_mode>
\t\t<serial>16549220{_h(node.hostname, g.index) % 100000:05d}</serial>
\t\t<uuid>GPU-{_h(node.hostname, g.index, "u") % 16**8:08x}-5c1e-3b6a-9f0e-6d2c8a4b7e10</uuid>
\t\t<minor_number>{minor}</minor_number>
\t\t<vbios_version>{node.vbios}</vbios_version>
\t\t<module_id>{g.index + 1}</module_id>
\t\t<pci>
\t\t\t<pci_bus>{bus}</pci_bus>
\t\t\t<pci_device>00</pci_device>
\t\t\t<pci_domain>0000</pci_domain>
\t\t\t<pci_device_id>233010DE</pci_device_id>
\t\t\t<pci_bus_id>0000{g.bdf.upper()}</pci_bus_id>
\t\t\t<pci_gpu_link_info>
\t\t\t\t<pcie_gen>
\t\t\t\t\t<max_link_gen>{g.gen_max}</max_link_gen>
\t\t\t\t\t<current_link_gen>1</current_link_gen>
\t\t\t\t\t<device_current_link_gen>1</device_current_link_gen>
\t\t\t\t\t<max_device_link_gen>{g.gen_max}</max_device_link_gen>
\t\t\t\t\t<max_host_link_gen>{g.host_gen_max}</max_host_link_gen>
\t\t\t\t</pcie_gen>
\t\t\t\t<link_widths>
\t\t\t\t\t<max_link_width>16x</max_link_width>
\t\t\t\t\t<current_link_width>{g.width}x</current_link_width>
\t\t\t\t</link_widths>
\t\t\t</pci_gpu_link_info>
\t\t</pci>
\t\t<fabric>
\t\t\t<state>Completed</state>
\t\t\t<status>Success</status>
\t\t</fabric>
\t\t<clocks_event_reasons>
\t\t\t<clocks_event_reason_gpu_idle>Active</clocks_event_reason_gpu_idle>
\t\t\t<clocks_event_reason_applications_clocks_setting>Not Active</clocks_event_reason_applications_clocks_setting>
\t\t\t<clocks_event_reason_sw_power_cap>Not Active</clocks_event_reason_sw_power_cap>
\t\t\t<clocks_event_reason_hw_slowdown>Not Active</clocks_event_reason_hw_slowdown>
\t\t\t<clocks_event_reason_hw_thermal_slowdown>{thermal}</clocks_event_reason_hw_thermal_slowdown>
\t\t\t<clocks_event_reason_hw_power_brake_slowdown>Not Active</clocks_event_reason_hw_power_brake_slowdown>
\t\t\t<clocks_event_reason_sync_boost>Not Active</clocks_event_reason_sync_boost>
\t\t\t<clocks_event_reason_sw_thermal_slowdown>Not Active</clocks_event_reason_sw_thermal_slowdown>
\t\t\t<clocks_event_reason_display_clocks_setting>Not Active</clocks_event_reason_display_clocks_setting>
\t\t</clocks_event_reasons>
\t\t<ecc_mode>
\t\t\t<current_ecc>Enabled</current_ecc>
\t\t\t<pending_ecc>Enabled</pending_ecc>
\t\t</ecc_mode>
\t\t<ecc_errors>
\t\t\t<volatile>
\t\t\t\t<sram_correctable>0</sram_correctable>
\t\t\t\t<sram_uncorrectable_parity>0</sram_uncorrectable_parity>
\t\t\t\t<sram_uncorrectable_secded>0</sram_uncorrectable_secded>
\t\t\t\t<dram_correctable>{3 if unc_dram else 0}</dram_correctable>
\t\t\t\t<dram_uncorrectable>{unc_dram}</dram_uncorrectable>
\t\t\t</volatile>
\t\t\t<aggregate>
\t\t\t\t<sram_correctable>0</sram_correctable>
\t\t\t\t<sram_uncorrectable_parity>0</sram_uncorrectable_parity>
\t\t\t\t<sram_uncorrectable_secded>0</sram_uncorrectable_secded>
\t\t\t\t<dram_correctable>{3 if unc_dram else 0}</dram_correctable>
\t\t\t\t<dram_uncorrectable>{unc_dram}</dram_uncorrectable>
\t\t\t\t<sram_threshold_exceeded>No</sram_threshold_exceeded>
\t\t\t</aggregate>
\t\t</ecc_errors>
\t\t<retired_pages>
\t\t\t<multiple_single_bit_retirement>
\t\t\t\t<retired_count>N/A</retired_count>
\t\t\t\t<retired_pagelist>N/A</retired_pagelist>
\t\t\t</multiple_single_bit_retirement>
\t\t\t<double_bit_retirement>
\t\t\t\t<retired_count>N/A</retired_count>
\t\t\t\t<retired_pagelist>N/A</retired_pagelist>
\t\t\t</double_bit_retirement>
\t\t\t<pending_retirement>N/A</pending_retirement>
\t\t</retired_pages>
\t\t<remapped_rows>
\t\t\t<remapped_row_corr>0</remapped_row_corr>
\t\t\t<remapped_row_unc>{remap_unc}</remapped_row_unc>
\t\t\t<remapped_row_pending>{"Yes" if g.remap_pending else "No"}</remapped_row_pending>
\t\t\t<remapped_row_failure>No</remapped_row_failure>
\t\t</remapped_rows>
\t\t<temperature>
\t\t\t<gpu_temp>{temp} C</gpu_temp>
\t\t\t<gpu_temp_max_threshold>92 C</gpu_temp_max_threshold>
\t\t\t<gpu_temp_slow_threshold>89 C</gpu_temp_slow_threshold>
\t\t\t<memory_temp>{temp + 6} C</memory_temp>
\t\t</temperature>
\t\t<gpu_power_readings>
\t\t\t<power_state>P0</power_state>
\t\t\t<power_draw>{68.9 + g.index:.2f} W</power_draw>
\t\t\t<current_power_limit>700.00 W</current_power_limit>
\t\t\t<requested_power_limit>700.00 W</requested_power_limit>
\t\t\t<default_power_limit>700.00 W</default_power_limit>
\t\t\t<min_power_limit>200.00 W</min_power_limit>
\t\t\t<max_power_limit>700.00 W</max_power_limit>
\t\t</gpu_power_readings>
\t\t<clocks>
\t\t\t<graphics_clock>345 MHz</graphics_clock>
\t\t\t<sm_clock>345 MHz</sm_clock>
\t\t\t<mem_clock>2619 MHz</mem_clock>
\t\t</clocks>
\t\t<max_clocks>
\t\t\t<graphics_clock>1980 MHz</graphics_clock>
\t\t\t<sm_clock>1980 MHz</sm_clock>
\t\t\t<mem_clock>2619 MHz</mem_clock>
\t\t\t<video_clock>1545 MHz</video_clock>
\t\t</max_clocks>
\t</gpu>""")
    parts.append("</nvidia_smi_log>")
    return "\n".join(parts) + "\n"


def _gpu_nic_link(node: NodeSpec, gi: int, ni: int) -> str:
    gpu = node.gpus[gi]
    nic = node.nics[ni]
    if gi == ni and gi not in node.rail_override:
        return "PIX"
    return "NODE" if gpu.numa == nic.numa else "SYS"


def render_topo(node: NodeSpec) -> str:
    gpus = [g for g in node.gpus if g.present]
    gnames = [f"GPU{i}" for i in range(len(gpus))]
    nnames = [f"NIC{i}" for i in range(len(node.nics))]
    cpu = {0: "0-55,112-167", 1: "56-111,168-223"}
    lines = ["\t" + "\t".join(gnames + nnames + ["CPU Affinity", "NUMA Affinity", "GPU NUMA ID"])]
    for r, g in enumerate(gpus):
        cells = []
        for c, h in enumerate(gpus):
            cells.append(" X " if r == c else f"NV{min(g.nvlinks, h.nvlinks)}")
        for ni in range(len(node.nics)):
            cells.append(_gpu_nic_link(node, g.index, ni))
        cells += [cpu[g.numa], str(g.numa), "N/A"]
        lines.append(f"GPU{r}\t" + "\t".join(cells))
    for ni, nic in enumerate(node.nics):
        cells = [_gpu_nic_link(node, g.index, ni) for g in gpus]
        for nj, other in enumerate(node.nics):
            cells.append(" X " if ni == nj else ("NODE" if nic.numa == other.numa else "SYS"))
        lines.append(f"NIC{ni}\t" + "\t".join(cells))
    lines += [
        "",
        "Legend:",
        "",
        "  X    = Self",
        "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)",
        "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node",
        "  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)",
        "  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)",
        "  PIX  = Connection traversing at most a single PCIe bridge",
        "  NV#  = Connection traversing a bonded set of # NVLinks",
        "",
        "NIC Legend:",
        "",
    ]
    lines += [f"  NIC{i}: {nic.name}" for i, nic in enumerate(node.nics)]
    return "\n".join(lines) + "\n"


_DIAG_ROWS = [
    ("Deployment", ["Denylist", "NVML Library", "CUDA Main Library", "Permissions and OS Blocks",
                    "Persistence Mode", "Environment Variables", "Page Retirement/Row Remap",
                    "Graphics Processes", "Inforom"]),
    ("Integration", ["PCIe"]),
    ("Hardware", ["GPU Memory"]),
    ("Stress", ["Targeted Stress", "Targeted Power", "Memory Bandwidth"]),
]


def render_dcgmi_diag(node: NodeSpec) -> str:
    w1, w2 = 27, 48

    def row(a: str, b: str) -> str:
        return f"| {a:<{w1 - 2}} | {b:<{w2 - 2}} |"

    def sec(name: str, lead: str = "+") -> str:
        left = f"-----  {name}  "
        return f"{lead}{left:-<{w1}}+{'':-<{w2}}+"

    n = sum(1 for g in node.gpus if g.present)
    out = [
        "Successfully ran diagnostic for group.",
        f"+{'':-<{w1}}+{'':-<{w2}}+",
        row("Diagnostic", "Result"),
        f"+{'':=<{w1}}+{'':=<{w2}}+",
        sec("Metadata", "|"),
        row("DCGM Version", "3.3.9"),
        row("Driver Version Detected", node.driver),
        row("GPU Device IDs Detected", ",".join(["2330"] * n)),
    ]
    for i, (section, tests) in enumerate(_DIAG_ROWS):
        out.append(sec(section, "|" if i == 0 else "+"))
        for t in tests:
            if i == 0:
                out.append(row(t, "Pass"))
            elif t in node.diag_fail:
                gpu, msg = node.diag_fail[t]
                ok = ", ".join(str(g.index) for g in node.gpus if g.present and g.index != gpu)
                out.append(row(t, f"Pass - GPUs: {ok}"))
                out.append(row("", f"Fail - GPU: {gpu}"))
                chunks = textwrap.wrap(msg, w2 - 2)
                out.append(row("Warning", chunks[0]))
                out += [row("", c) for c in chunks[1:]]
            else:
                out.append(row(t, "Pass - All"))
    out.append(f"+{'':-<{w1}}+{'':-<{w2}}+")
    return "\n".join(out) + "\n"


def render_ibstat(node: NodeSpec) -> str:
    blocks = []
    for i, nic in enumerate(node.nics):
        guid = f"0x946dae0300{0x4f2a10 + i + 16 * (_h(node.hostname) % 4096):06x}"
        ca_type = "MT4129" if nic.layer == "InfiniBand" else "MT4125"
        blocks.append(f"""CA '{nic.name}'
\tCA type: {ca_type}
\tNumber of ports: 1
\tFirmware version: {node.nic_fw if nic.layer == "InfiniBand" else "22.39.1002"}
\tHardware version: 0
\tNode GUID: {guid}
\tSystem image GUID: {guid}
\tPort 1:
\t\tState: {nic.state}
\t\tPhysical state: {nic.phys}
\t\tRate: {nic.rate}
\t\tBase lid: {0 if nic.layer == "Ethernet" or nic.state != "Active" else 100 + i}
\t\tLMC: 0
\t\tSM lid: {0 if nic.layer == "Ethernet" else 1}
\t\tCapability mask: {"0x00010000" if nic.layer == "Ethernet" else "0xa651e848"}
\t\tPort GUID: {guid}
\t\tLink layer: {nic.layer}""")
    return "\n".join(blocks) + "\n"


def render_lspci(node: NodeSpec) -> str:
    out = []
    for g in node.gpus:
        if node.vendor == "nvidia":
            desc = "3D controller: NVIDIA Corporation GH100 [H100 SXM5 80GB] (rev " + ("a1)" if g.present else "ff)")
            speed = "32GT/s"
        else:
            desc = "Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X] (rev 00)"
            speed = "32GT/s"
        out.append(f"{g.bdf[5:]} {desc}")
        out.append(f"\tSubsystem: {'NVIDIA Corporation Device 16c1' if node.vendor == 'nvidia' else 'Advanced Micro Devices, Inc. [AMD/ATI] Device 74a1'}")
        out.append(f"\tNUMA node: {g.numa}")
        out.append(f"\t\tLnkCap:\tPort #0, Speed {speed}, Width x16, ASPM not supported")
        sta = f"Speed {speed}, Width x{g.width}" + (" (downgraded)" if g.width < 16 else "")
        out.append(f"\t\tLnkSta:\t{sta}")
        out.append("")
    for nic in node.nics:
        model = "MT2910 Family [ConnectX-7]" if nic.layer == "InfiniBand" else "MT2892 Family [ConnectX-6 Dx]"
        nic_type = "Infiniband controller" if nic.layer == "InfiniBand" else "Ethernet controller"
        out.append(f"{nic.bdf[5:]} {nic_type}: Mellanox Technologies {model}")
        out.append(f"\tNUMA node: {nic.numa}")
        out.append("\t\tLnkCap:\tPort #0, Speed 32GT/s, Width x16, ASPM not supported")
        out.append("\t\tLnkSta:\tSpeed 32GT/s, Width x16")
        out.append("")
    return "\n".join(out)


def render_rocm_smi(node: NodeSpec) -> dict:
    doc: dict = {}
    for g in node.gpus:
        doc[f"card{g.index}"] = {
            "Device Name": "AMD Instinct MI300X OAM",
            "Device ID": "0x74a1",
            "Card Series": "AMD Instinct MI300X OAM",
            "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
            "Card SKU": "M3000100",
            "VBIOS version": node.vbios,
            "Serial Number": f"692{_h(node.hostname, g.index) % 10**9:09d}",
            "PCI Bus": g.bdf.upper(),
            "Temperature (Sensor edge) (C)": "N/A",
            "Temperature (Sensor junction) (C)": f"{40 + g.index % 3}.0",
            "Temperature (Sensor memory) (C)": f"{34 + g.index % 3}.0",
            "Current Socket Graphics Package Power (W)": f"{136 + g.index}.0",
            "Max Graphics Package Power (W)": "750.0",
            "(Topology) Numa Node": str(g.numa),
            "(Topology) Numa Affinity": str(g.numa),
            "RAS UMC Correctable Error Count": str(4 if g.ecc_unc else 0),
            "RAS UMC Uncorrectable Error Count": str(g.ecc_unc),
            "RAS GFX Correctable Error Count": "0",
            "RAS GFX Uncorrectable Error Count": "0",
        }
    doc["system"] = {"Driver version": node.driver}
    return doc


def render_rocm_topo(node: NodeSpec) -> dict:
    system = {}
    for a in node.gpus:
        for b in node.gpus:
            if a.index < b.index:
                link = "XGMI" if a.nvlinks >= 18 and b.nvlinks >= 18 else "PCIE"
                system[f"(Topology) Link type between DRM devices {a.index} and {b.index}"] = link
    return {"system": system}


def render_dmesg_xid(node: NodeSpec) -> str:
    lines = []
    for g in node.gpus:
        if g.xid == 79:
            bus = g.bdf[5:].split(".")[0]
            lines.append(
                f"[ 9112.483201] NVRM: Xid (PCI:0000:{bus}): 79, pid=48213, name=gpu_burn, "
                "GPU has fallen off the bus."
            )
    return "\n".join(lines) + ("\n" if lines else "")


def burnin_summary(node: NodeSpec) -> dict:
    return {
        "schema": "burnin/v1",
        "plan": "gemm-soak-4h",
        "duration_s": 14400,
        "gpus": [
            {
                "index": g.index,
                "max_temp_c": g.max_temp_c,
                "throttle_seconds": {"hw_thermal_slowdown": g.throttle_s, "hw_power_brake_slowdown": 0.0},
                "gemm_tflops": g.tflops,
                "errors": 0,
            }
            for g in node.gpus
            if g.present
        ],
    }


def render_all_reduce(busbw: float, nranks: int, label: str) -> str:
    sizes = [2**k for k in range(23, 34, 2)]  # 8 MiB .. 8 GiB
    lines = [
        "# nThread 1 nGpus 1 minBytes 8388608 maxBytes 8589934592 step: 4(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0",
        "#",
        "# Using devices",
        f"#  {label}: {nranks} ranks",
        "#",
        "#                                                              out-of-place                       in-place          ",
        "#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong",
        "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       ",
    ]
    factor = 2 * (nranks - 1) / nranks
    total = 0.0
    for i, size in enumerate(sizes):
        frac = 0.35 + 0.65 * (i / (len(sizes) - 1)) ** 0.5
        bus = busbw * frac
        alg = bus / factor
        t = size / (alg * 1e3)
        total += bus
        lines.append(
            f"{size:>12} {size // 4:>13} {'float':>9} {'sum':>7} {-1:>7} {t:>8.1f} {alg:>7.2f} {bus:>7.2f} {0:>6} "
            f"{t * 0.998:>8.1f} {alg * 1.002:>7.2f} {bus * 1.002:>7.2f} {0:>6}"
        )
    lines += [
        "# Out of bounds values : 0 OK",
        f"# Avg bus bandwidth    : {total / len(sizes):.3f}",
        "#",
    ]
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding=ENCODING) as fh:
        fh.write(text)


def _write_json(path: Path, doc: dict) -> None:
    with path.open("w", encoding=ENCODING) as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def write_node(node: NodeSpec, root: Path) -> Path:
    d = root / node.hostname
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "meta.json", {"hostname": node.hostname, "rack": node.rack, "node_type": node.node_type})
    _write(d / "uname-r.txt", node.kernel + "\n")
    _write(d / "lspci-vv.txt", render_lspci(node))
    _write(d / "ibstat.txt", render_ibstat(node))
    _write(d / "ib-devices.txt", "".join(f"{n.name} {n.bdf}\n" for n in node.nics))
    _write(d / "dmesg-xid.txt", render_dmesg_xid(node))
    _write_json(d / "burnin.json", burnin_summary(node))
    _write(d / "all_reduce_perf.txt", render_all_reduce(node.allreduce_busbw, 8, node.hostname))
    if node.vendor == "nvidia":
        _write(d / "nvidia-smi-q.xml", render_nvidia_smi_xml(node))
        _write(d / "nvidia-smi-topo.txt", render_topo(node))
        _write(d / "dcgmi-diag-r3.txt", render_dcgmi_diag(node))
    else:
        _write_json(d / "rocm-smi.json", render_rocm_smi(node))
        _write_json(d / "rocm-smi-topo.json", render_rocm_topo(node))
        _write(d / "rocm-version.txt", node.runtime + "\n")
    return d


def write_cluster_results(root: Path) -> None:
    """Multi-node all-reduce output per pool, as the cluster stage's NCCL/RCCL stub returns it."""
    cl = root / "_cluster"
    cl.mkdir(parents=True, exist_ok=True)
    _write(cl / "all_reduce_perf.hgx-h100-ndr.txt", render_all_reduce(371.8, 8 * 32, "hgx-h100-ndr pool"))
    _write(cl / "all_reduce_perf.mi300x-ndr.txt", render_all_reduce(338.6, 8 * 4, "mi300x-ndr pool"))


def write_fleet(root: Path, scenario: str = "defects", seed: int = SEED) -> list[NodeSpec]:
    nodes = build_fleet(scenario, seed)
    root.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        write_node(node, root)
    write_cluster_results(root)
    _write_json(root / "_scenario.json", {
        "scenario": scenario,
        "seed": seed,
        "injected": {n.hostname: n.defect for n in nodes if n.defect},
    })
    return nodes
