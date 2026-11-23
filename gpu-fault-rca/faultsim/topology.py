"""Simulated deployment: 4 nodes x 8 H100, two model replicas, a router in front.

Each replica is TP=8 inside a node (NVLink all-reduce) and PP=2 across a node
pair (NCCL send/recv over InfiniBand). Global rank inside a replica group is
`stage * 8 + local_gpu`, so gpu-node-1 GPU 3 is rank 11 of replica-a.
"""
from __future__ import annotations

import hashlib

GPUS_PER_NODE = 8
TP = 8
PP = 2
NODES = ("gpu-node-0", "gpu-node-1", "gpu-node-2", "gpu-node-3")
REPLICAS = {
    "replica-a": ("gpu-node-0", "gpu-node-1"),
    "replica-b": ("gpu-node-2", "gpu-node-3"),
}
PCI_BUS = ("0000:18:00", "0000:2a:00", "0000:3a:00", "0000:5d:00",
           "0000:9a:00", "0000:ab:00", "0000:ba:00", "0000:db:00")
NODE_IP = {node: f"10.0.0.{11 + i}" for i, node in enumerate(NODES)}
IB_DEVICE = "mlx5_1"
GPU_MODEL = "NVIDIA H100 80GB HBM3"
FB_TOTAL_MIB = 81559
DRIVER_VERSION = "550.90.07"
UPGRADE_DRIVER_VERSION = "560.35.03"
ENGINE_IMAGE = "registry.local/serving/engine:v0.9.0"
BAD_ENGINE_IMAGE = "registry.local/serving/engine:v0.9.1"


def replica_of(node: str) -> str:
    for replica, nodes in REPLICAS.items():
        if node in nodes:
            return replica
    raise KeyError(node)


def rank_of(node: str, gpu: int) -> int:
    return REPLICAS[replica_of(node)].index(node) * GPUS_PER_NODE + gpu


def gpu_uuid(node: str, gpu: int) -> str:
    h = hashlib.sha256(f"{node}/{gpu}".encode()).hexdigest()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def pid_of(node: str, gpu: int) -> int:
    return 2200 + NODES.index(node) * 100 + gpu


def manifest_topology() -> dict:
    """What a real incident bundle carries from the job spec: rank map, PCI map, IPs."""
    return {
        "gpus_per_node": GPUS_PER_NODE,
        "tp": TP,
        "pp": PP,
        "replicas": {r: list(n) for r, n in REPLICAS.items()},
        "node_ip": dict(NODE_IP),
        "pci_bus": list(PCI_BUS),
    }
