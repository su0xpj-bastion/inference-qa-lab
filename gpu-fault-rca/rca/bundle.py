"""Load an incident bundle into one in-memory view.

Reads only the telemetry a real collection job would gather. `ground_truth.json`
(written by the chaos harness) is never opened here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from rca import parsers as P

ENCODING = "utf-8"
BUNDLE_FILES = ("kern.log", "dcgm.prom", "node_exporter.prom", "worker.log", "engine.log")


def norm_pci(pci: str) -> str:
    """'00000000:3A:00.0' / '0000:3a:00.0' / '0000:3a:00' -> '0000:3a:00'."""
    p = pci.lower()
    if p.endswith(".0"):
        p = p[:-2]
    head, _, rest = p.partition(":")
    if len(head) == 8:
        head = head[4:]
    return f"{head}:{rest}"


@dataclass
class Bundle:
    root: Path
    manifest: dict
    start: float
    end: float
    events: list[P.LogEvent] = field(default_factory=list)
    nccl_warns: list[P.LogEvent] = field(default_factory=list)
    dcgm: dict[str, list[P.Sample]] = field(default_factory=dict)
    nodex: dict[str, list[P.Sample]] = field(default_factory=dict)
    requests: list[P.Request] = field(default_factory=list)

    # topology helpers ---------------------------------------------------
    @property
    def topology(self) -> dict:
        return self.manifest["topology"]

    @property
    def nodes(self) -> list[str]:
        return [n for nodes in self.topology["replicas"].values() for n in nodes]

    def replica_of(self, node: str | None) -> str | None:
        for replica, nodes in self.topology["replicas"].items():
            if node in nodes:
                return replica
        return None

    def rank_of(self, node: str, gpu: int) -> int:
        replica = self.replica_of(node)
        return self.topology["replicas"][replica].index(node) * self.topology["gpus_per_node"] + gpu

    def locate(self, replica: str, rank: int) -> tuple[str, int]:
        per = self.topology["gpus_per_node"]
        return self.topology["replicas"][replica][rank // per], rank % per

    def gpu_of_pci(self, pci: str) -> int | None:
        want = norm_pci(pci)
        for i, p in enumerate(self.topology["pci_bus"]):
            if norm_pci(p) == want:
                return i
        return None

    def node_of_ip(self, ip: str | None) -> str | None:
        for node, addr in self.topology["node_ip"].items():
            if addr == ip:
                return node
        return None

    @property
    def slo(self) -> dict:
        return self.manifest.get("slo", {"p99_latency_s": 3.0, "error_rate": 0.01})


def load(root: Path) -> Bundle:
    root = Path(root)
    with (root / "manifest.json").open("r", encoding=ENCODING) as fh:
        manifest = json.load(fh)
    start = float(manifest["start_epoch"])
    b = Bundle(root, manifest, start, start + float(manifest["duration_s"]))
    year = int(manifest["start"][:4])
    for node in b.nodes:
        d = root / "nodes" / node
        rel = f"nodes/{node}"
        if (d / "kern.log").exists():
            b.events += P.parse_kern(d / "kern.log", f"{rel}/kern.log")
        if (d / "worker.log").exists():
            for ev in P.parse_worker(d / "worker.log", f"{rel}/worker.log", node, year):
                (b.nccl_warns if ev.kind == "nccl_warn" else b.events).append(ev)
        if (d / "engine.log").exists():
            b.events += P.parse_engine(d / "engine.log", f"{rel}/engine.log", node, year)
        if (d / "dcgm.prom").exists():
            b.dcgm[node] = P.parse_prom(d / "dcgm.prom", f"{rel}/dcgm.prom")
        if (d / "node_exporter.prom").exists():
            b.nodex[node] = P.parse_prom(d / "node_exporter.prom", f"{rel}/node_exporter.prom")
    if (root / "k8s" / "events.json").exists():
        b.events += P.parse_events(root / "k8s" / "events.json", "k8s/events.json")
    if (root / "router" / "access.log").exists():
        b.requests = P.parse_access(root / "router" / "access.log")
    b.events.sort(key=lambda e: (e.ts, e.line.source, e.line.lineno))
    return b
