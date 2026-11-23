from __future__ import annotations

# Parser for NCCL_DEBUG=INFO logs (NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH is enough).
#
# Every NCCL line is prefixed "<host>:<pid>:<tid> [<cudaDev>] NCCL INFO|WARN <msg>".
# What it extracts per host:
#   - network plugin chosen ("Using network IB" vs "Using network Socket")
#   - per-connection transport from channel lines, e.g.
#       "Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA"
#     NET/IB/<n>/GDRDMA = InfiniBand/RoCE verbs with GPUDirect RDMA on HCA n
#     NET/IB/<n>        = verbs, but bounced through host memory (GDR off)
#     NET/Socket/<n>    = TCP sockets: the fabric is bypassed entirely
#     P2P/... SHM/...   = intra-node paths
#   - ring and tree channel counts
#   - GDR decisions ("GPU Direct RDMA Enabled|Disabled for GPU x / HCA y (distance a <= b)")
#   - NCCL_* settings echoed as "<VAR> set by environment to <value>."
#   - WARN lines

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ENCODING = "utf-8"

PREFIX_RE = re.compile(r"^(?P<host>[^:\s]+):(?P<pid>\d+):(?P<tid>\d+)\s+\[(?P<dev>\d+)\]\s+NCCL\s+(?P<level>INFO|WARN)\s+(?P<msg>.*)$")
VERSION_RE = re.compile(r"^NCCL version (?P<v>\S+)")
ENV_RE = re.compile(r"^(?P<k>NCCL_[A-Z0-9_]+) set by environment to (?P<v>.*?)\.?$")
NETWORK_RE = re.compile(r"^Using network (?P<net>\S+)")
CONN_RE = re.compile(
    r"^Channel (?P<ch>\d+)/(?P<nch>\d+) : (?P<src>\d+)\[(?P<srcdev>\d+)\] -> (?P<dst>\d+)\[(?P<dstdev>\d+)\] "
    r"(?:\[(?P<dir>send|receive)\] )?via (?P<via>\S+)"
)
RING_RE = re.compile(r"^Channel (?P<ch>\d+)/(?P<nch>\d+) :\s+(?P<order>(?:\d+\s*)+)$")
TREE_RE = re.compile(r"^Trees(?P<trees>(?: \[\d+\] \S+)+)")
GDR_RE = re.compile(r"GPU Direct RDMA (?P<state>Enabled|Disabled) for GPU (?P<gpu>\w+) / HCA (?P<hca>\d+)")
NO_IB_RE = re.compile(r"^NET/IB : No device found")
IB_DEVS_RE = re.compile(r"^NET/IB : Using (?P<devs>(?:\[\d+\]\S+\s*)+)")
RANK_RE = re.compile(r"\brank (?P<rank>\d+) nr?anks (?P<nranks>\d+)", re.IGNORECASE)


@dataclass
class HostSummary:
    host: str
    version: str | None = None
    networks: set[str] = field(default_factory=set)
    env: dict[str, str] = field(default_factory=dict)
    transports: set[str] = field(default_factory=set)  # "IB", "Socket", "P2P", "SHM", ...
    net_conns: list[dict] = field(default_factory=list)
    gdr: dict[int, bool] = field(default_factory=dict)  # HCA index -> GDR used on its connections
    gdr_decisions: list[dict] = field(default_factory=list)
    ring_channels: set[int] = field(default_factory=set)
    tree_channels: int = 0
    ib_devices: list[str] = field(default_factory=list)
    ib_not_found: bool = False
    ranks: set[int] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    @property
    def socket_fallback(self) -> bool:
        return "Socket" in self.networks or "Socket" in self.transports or self.ib_not_found

    @property
    def gdr_off_hcas(self) -> list[int]:
        return sorted(h for h, on in self.gdr.items() if not on)

    def to_dict(self) -> dict:
        return {
            "host": self.host, "version": self.version, "networks": sorted(self.networks),
            "transports": sorted(self.transports), "socket_fallback": self.socket_fallback,
            "gdr_by_hca": {str(k): v for k, v in sorted(self.gdr.items())},
            "ring_channels": len(self.ring_channels), "tree_channels": self.tree_channels,
            "env": dict(sorted(self.env.items())), "ranks": sorted(self.ranks),
            "warnings": self.warnings,
        }


def _transport(via: str) -> tuple[str, int | None, bool]:
    parts = via.split("/")
    if parts[0] == "NET" and len(parts) >= 2:
        idx = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
        return parts[1], idx, "GDRDMA" in parts[3:]
    return parts[0], None, False


def parse(text: str) -> dict[str, HostSummary]:
    hosts: dict[str, HostSummary] = {}
    gdr_seen: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for raw in text.splitlines():
        m = PREFIX_RE.match(raw.strip())
        if not m:
            continue
        host, msg = m["host"], m["msg"].strip()
        h = hosts.setdefault(host, HostSummary(host=host))
        if m["level"] == "WARN":
            h.warnings.append(msg)
            continue
        if v := VERSION_RE.match(msg):
            h.version = v["v"]
        elif e := ENV_RE.match(msg):
            h.env[e["k"]] = e["v"]
        elif n := NETWORK_RE.match(msg):
            h.networks.add(n["net"])
        elif NO_IB_RE.match(msg):
            h.ib_not_found = True
        elif d := IB_DEVS_RE.match(msg):
            h.ib_devices = re.findall(r"\[\d+\](\S+)", d["devs"])
        elif c := CONN_RE.match(msg):
            kind, idx, gdr = _transport(c["via"])
            h.transports.add(kind)
            if c["via"].startswith("NET/"):
                h.net_conns.append({"channel": int(c["ch"]), "src": int(c["src"]), "dst": int(c["dst"]),
                                    "dev": int(m["dev"]), "kind": kind, "hca": idx, "gdr": gdr})
                if kind == "IB" and idx is not None:
                    gdr_seen[host][idx].append(gdr)
        elif r := RING_RE.match(msg):
            h.ring_channels.add(int(r["ch"]))
        elif t := TREE_RE.match(msg):
            h.tree_channels = max(h.tree_channels, len(re.findall(r"\[\d+\]", t["trees"])))
        if g := GDR_RE.search(msg):
            h.gdr_decisions.append({"state": g["state"], "gpu": g["gpu"], "hca": int(g["hca"]), "line": msg})
        if rk := RANK_RE.search(msg):
            h.ranks.add(int(rk["rank"]))
    for host, per_hca in gdr_seen.items():
        hosts[host].gdr = {idx: all(vals) for idx, vals in per_hca.items()}
    return hosts


def parse_file(path: Path) -> dict[str, HostSummary]:
    with path.open("r", encoding=ENCODING) as fh:
        return parse(fh.read())
