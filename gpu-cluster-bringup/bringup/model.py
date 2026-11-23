from __future__ import annotations

# Vendor-neutral facts about a node.
#
# Parsers turn tool output (nvidia-smi, rocm-smi, ibstat, lspci, ...) into these
# records. Gates only ever read these records, so a gate never knows or cares
# whether the GPU came from NVIDIA or AMD.

from dataclasses import dataclass, field

PASS = "pass"
FAIL = "fail"
WARN = "warn"
SKIP = "skip"


@dataclass
class Gpu:
    index: int
    bus_id: str  # normalised to dddd:bb:dd.f, lower case
    product: str
    vbios: str | None = None
    serial: str | None = None
    persistence: bool | None = None
    pcie_gen_current: int | None = None
    pcie_gen_max: int | None = None  # what the device can do
    pcie_gen_host_max: int | None = None  # what the slot can do
    pcie_width_current: int | None = None
    pcie_width_max: int | None = None
    ecc_enabled: bool | None = None
    ecc_pending_enabled: bool | None = None
    ecc_correctable_volatile: int = 0
    ecc_uncorrectable_volatile: int = 0
    ecc_uncorrectable_aggregate: int = 0
    retired_pages_sbe: int | None = None
    retired_pages_dbe: int | None = None
    retirement_pending: bool | None = None
    remapped_rows_uncorrectable: int | None = None
    remap_pending: bool | None = None
    remap_failure: bool | None = None
    power_limit_w: float | None = None
    default_power_limit_w: float | None = None
    max_sm_clock_mhz: int | None = None
    max_mem_clock_mhz: int | None = None
    temp_c: float | None = None
    active_throttle: list[str] = field(default_factory=list)
    fabric_state: str | None = None  # NVSwitch fabric registration (Hopper+)
    numa_node: int | None = None


@dataclass
class Port:
    number: int
    state: str  # Active / Down / Init / Armed
    physical_state: str  # LinkUp / Polling / Disabled
    rate_gbps: int
    link_layer: str  # InfiniBand / Ethernet


@dataclass
class Hca:
    name: str  # mlx5_0, rdmap16s27
    kind: str  # "ib" | "efa"
    firmware: str | None = None
    ports: list[Port] = field(default_factory=list)


@dataclass
class PciDevice:
    bdf: str
    description: str
    numa_node: int | None = None
    width_cap: int | None = None
    width_sta: int | None = None
    speed_cap_gts: float | None = None
    speed_sta_gts: float | None = None


@dataclass
class Topology:
    """GPU<->GPU and GPU<->NIC connectivity (nvidia-smi topo -m or rocm-smi --showtopo)."""

    gpus: list[str]
    nics: list[str]  # topo column names (NIC0...) resolved to device names (mlx5_0...)
    links: dict[tuple[str, str], str]  # (a, b) -> "NV18" / "PXB" / "SYS" / "XGMI"
    gpu_numa: dict[str, int | None] = field(default_factory=dict)

    def link(self, a: str, b: str) -> str | None:
        return self.links.get((a, b))


@dataclass
class DiagTest:
    name: str
    category: str
    result: str  # Pass / Fail / Warn / Skip
    failed_gpus: list[int] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


@dataclass
class NodeFacts:
    hostname: str
    rack: str
    node_type: str
    vendor: str | None = None
    driver_version: str | None = None
    runtime_version: str | None = None  # CUDA or ROCm
    kernel: str | None = None
    gpus: list[Gpu] = field(default_factory=list)
    hcas: list[Hca] = field(default_factory=list)
    pci: list[PciDevice] = field(default_factory=list)
    topology: Topology | None = None
    diag: list[DiagTest] | None = None
    burnin: dict | None = None
    allreduce: list[dict] | None = None
    missing: list[str] = field(default_factory=list)  # evidence files that were absent or unparseable


@dataclass
class Check:
    stage: str
    name: str
    status: str
    reason: str = ""
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "evidence": self.evidence,
        }
