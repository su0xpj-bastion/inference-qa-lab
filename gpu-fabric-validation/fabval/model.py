from __future__ import annotations

# Topology-aware expected bandwidth for nccl-tests results.
#
# Bus bandwidth (busbw) is what nccl-tests reports so that numbers from different
# rank counts compare against the bottleneck link. The factors below are the ones
# nccl-tests itself uses (see its doc/PERFORMANCE.md):
#
#     all_reduce              busbw = algbw * 2(n-1)/n
#     all_gather, reduce_scatter, alltoall
#                             busbw = algbw * (n-1)/n
#     sendrecv, broadcast, reduce
#                             busbw = algbw
#
# The expected value is the bottleneck link times an efficiency from topologies.json.

import json
from dataclasses import dataclass
from pathlib import Path

ENCODING = "utf-8"
TOPOLOGIES_PATH = Path(__file__).with_name("topologies.json")
THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")

BUSBW_FACTORS = {
    "all_reduce": lambda n: 2 * (n - 1) / n,
    "all_gather": lambda n: (n - 1) / n,
    "reduce_scatter": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
    "sendrecv": lambda n: 1.0,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}


def busbw_factor(collective: str, nranks: int) -> float:
    if collective not in BUSBW_FACTORS:
        raise ValueError(f"unknown collective {collective!r}")
    if nranks < 1:
        raise ValueError("nranks must be >= 1")
    return BUSBW_FACTORS[collective](nranks)


def load_json(path: Path) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        return json.load(fh)


def load_topologies(path: Path = TOPOLOGIES_PATH) -> dict[str, dict]:
    return {k: v for k, v in load_json(path).items() if not k.startswith("_")}


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return load_json(path)


@dataclass(frozen=True)
class NodeType:
    name: str
    gpus_per_node: int
    hcas: tuple[str, ...]
    link_layer: str
    nic_rate_gbps: float
    nvlink_GBps_per_gpu: float
    efficiency: dict
    latency_budget_us: dict

    @property
    def nic_GBps(self) -> float:
        """One NIC, one direction, in GB/s (nccl-tests units: 1e9 bytes/s)."""
        return self.nic_rate_gbps / 8.0

    @property
    def is_roce(self) -> bool:
        return self.link_layer == "Ethernet"

    @classmethod
    def load(cls, name: str, path: Path = TOPOLOGIES_PATH) -> NodeType:
        topologies = load_topologies(path)
        if name not in topologies:
            raise KeyError(f"unknown node type {name!r}; known: {sorted(topologies)}")
        t = topologies[name]
        return cls(
            name=name,
            gpus_per_node=t["gpus_per_node"],
            hcas=tuple(t["hcas"]),
            link_layer=t["link_layer"],
            nic_rate_gbps=t["nic_rate_gbps"],
            nvlink_GBps_per_gpu=t["nvlink_GBps_per_gpu"],
            efficiency=t["efficiency"],
            latency_budget_us=t["latency_budget_us"],
        )


@dataclass(frozen=True)
class JobShape:
    collective: str
    nnodes: int
    ranks_per_node: int
    algo: str = "ring"

    @property
    def nranks(self) -> int:
        return self.nnodes * self.ranks_per_node


def expected_busbw(node: NodeType, shape: JobShape) -> float:
    """Expected large-message busbw in GB/s for this node type and job shape."""
    if shape.algo not in ("ring", "tree"):
        raise ValueError(f"algo must be ring or tree, got {shape.algo!r}")
    if shape.ranks_per_node > node.gpus_per_node:
        raise ValueError("ranks_per_node exceeds GPUs per node")
    eff = node.efficiency
    intra = node.nvlink_GBps_per_gpu * eff["intra_ring"]
    nics_used = min(len(node.hcas), shape.ranks_per_node)

    if shape.nnodes == 1:
        if shape.collective == "sendrecv":
            return node.nvlink_GBps_per_gpu * eff["intra_ring"]
        if shape.collective == "alltoall":
            return node.nvlink_GBps_per_gpu * eff["alltoall"]
        return intra

    if shape.collective in ("sendrecv", "broadcast", "reduce"):
        # Point-to-point across the network: one rank's traffic leaves through one NIC.
        return node.nic_GBps * eff["net_p2p"]
    if shape.collective == "alltoall":
        # Only the remote share of each rank's data crosses its NIC.
        n, g = shape.nranks, shape.ranks_per_node
        per_gpu = node.nic_GBps * eff["alltoall"] * (len(node.hcas) / node.gpus_per_node)
        return per_gpu * (n - 1) / (n - g)

    # Ring/tree collectives: every channel crosses the node boundary through one NIC,
    # so the inter-node part is (NICs in use) x (one NIC) x efficiency, capped by NVLink.
    net = nics_used * node.nic_GBps * eff["net_ring" if shape.algo == "ring" else "net_tree"]
    return min(intra, net)


def latency_budget_us(node: NodeType, key: str) -> float:
    return float(node.latency_budget_us[key])
