from __future__ import annotations

# Simulated cluster: writes the same files a real capture produces (see docs/RUNBOOK.md and
# scripts/collect_node.sh) for N nodes, with faults injected at chosen nodes / rails.
#
# Everything is generated from one fault table, so the pieces agree with each other:
# a down-trained HCA shows Rate 200 in ibstat, link errors in perfquery, half bandwidth in
# every pair test on that rail, and a slower full collective. The analyzer never imports
# this module; it reads only the text files, exactly as it would from a real cluster.
#
# Fault kinds (spec "kind:node[:rail]"):
#   downtrain:node-03:2   HCA 2 trained at 200G instead of 400G (+ symbol errors, link downed)
#   pfc:node-05:4         RoCE port 4 without PFC on the RoCE priority: drops, retransmits, tail latency
#   numa:node-02:6        NIC 6 on the other socket: GPU6<->NIC6 is SYS, GDR off, ~60% bandwidth
#   gid:node-07           GID table shifted on one node: NCCL_IB_GID_INDEX hits a non-RoCEv2 GID,
#                         NCCL falls back to NET/Socket on that node
#   corrupt:node-04       silent data corruption on one node's GPUs (#wrong > 0)

import json
import random
from dataclasses import dataclass
from pathlib import Path

from fabval.model import JobShape, NodeType, busbw_factor, expected_busbw

ENCODING = "utf-8"
NCCL_VERSION = "2.23.4+cuda12.6"
GPU_BUS_IDS = ("18", "2a", "3a", "5d", "9a", "ab", "ba", "db")
GPU_NAME = "NVIDIA H100 80GB HBM3"
PAIR_SIZES = (8, 1 << 30)
FULL_SIZES = (8, 8 << 30)
SOCKET_GBPS = 2.6  # what NCCL's socket transport typically manages over the OOB interface
FAULT_KINDS = ("downtrain", "pfc", "numa", "gid", "corrupt")

BASE_ENV = {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET,GRAPH",
    "NCCL_IB_HCA": "^=mlx5_bond_0",
    "NCCL_NET_GDR_LEVEL": "PXB",
    "NCCL_CROSS_NIC": "0",
    "NCCL_SOCKET_IFNAME": "eth0",
}
ROCE_ENV = {"NCCL_IB_GID_INDEX": "3", "NCCL_IB_TC": "106", "NCCL_IB_QPS_PER_CONNECTION": "4"}


@dataclass(frozen=True)
class Fault:
    kind: str
    node: str
    rail: int | None = None

    @classmethod
    def parse(cls, spec: str) -> Fault:
        parts = spec.split(":")
        if parts[0] not in FAULT_KINDS or len(parts) not in (2, 3):
            raise ValueError(f"bad fault spec {spec!r}; want kind:node[:rail], kind in {FAULT_KINDS}")
        rail = int(parts[2]) if len(parts) == 3 else None
        if parts[0] in ("downtrain", "pfc", "numa") and rail is None:
            raise ValueError(f"{parts[0]} needs a rail: {parts[0]}:<node>:<rail>")
        return cls(parts[0], parts[1], rail)

    def __str__(self) -> str:
        return f"{self.kind}:{self.node}" + (f":{self.rail}" if self.rail is not None else "")


def node_names(n: int) -> list[str]:
    return [f"node-{i:02d}" for i in range(1, n + 1)]


def netdev(rail: int) -> str:
    return f"ens{rail + 1}np0"


def sizes(lo: int, hi: int) -> list[int]:
    out, s = [], lo
    while s <= hi:
        out.append(s)
        s *= 2
    return out


def _fmt_time(us: float) -> str:
    if us >= 10000.0:
        return f"{us:7.0f}"
    if us >= 100.0:
        return f"{us:7.1f}"
    return f"{us:7.2f}"


def format_nccl_tests(collective: str, devices: list[tuple[str, int]], peak_busbw: float, lat_us: float,
                      size_range: tuple[int, int], rng: random.Random, lat_jitter: tuple[float, float] = (1.0, 1.0),
                      wrong_sizes: frozenset[int] = frozenset(), iters: int = 20) -> str:
    """Render nccl-tests stdout in the exact layout of nccl-tests 2.13 common.cu."""
    n = len(devices)
    factor = busbw_factor(collective, n)
    peak_alg = peak_busbw / factor
    redop = "sum" if collective in ("all_reduce", "reduce_scatter", "sendrecv", "reduce") else "none"
    lines = [
        f"# nThread 1 nGpus 1 minBytes {size_range[0]} maxBytes {size_range[1]} step: 2(factor) "
        f"warmup iters: 5 iters: {iters} agg iters: 1 validation: 1 graph: 0",
        "#",
        "# Using devices",
    ]
    for rank, (host, dev) in enumerate(devices):
        lines.append(f"#  Rank {rank:2d} Group  0 Pid {40000 + rank * 7:6d} on {host:>10s} device {dev:2d} "
                     f"[0x{GPU_BUS_IDS[dev]}] {GPU_NAME}")
    lines += [
        "#",
        "#                                                              out-of-place                       in-place          ",
        "#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong",
        "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       ",
    ]
    busbws: list[float] = []
    total_wrong = 0
    for size in sizes(*size_range):
        count = size // 4 if collective in ("all_reduce", "sendrecv", "broadcast", "reduce") else size // 4 // n
        cells = []
        for _ in ("oop", "ip"):
            lat = lat_us * rng.uniform(*lat_jitter) * rng.uniform(0.97, 1.03)
            t = lat + size / (peak_alg * rng.uniform(0.985, 1.015) * 1e3)
            ts = _fmt_time(t)
            t_printed = float(ts)
            alg = size / t_printed / 1e3
            bus = alg * factor
            wrong = rng.randint(1, 4096) if size in wrong_sizes else 0
            total_wrong += wrong
            busbws.append(round(bus, 2))
            cells.append(f"  {ts}  {alg:6.2f}  {bus:6.2f}  {wrong:5d}")
        lines.append(f"{size:12d}  {count:12d}  {'float':>8s}  {redop:>6s}  {-1:6d}" + "".join(cells))
    lines.append(f"# Out of bounds values : {total_wrong} {'OK' if total_wrong == 0 else 'FAILED'}")
    lines.append(f"# Avg bus bandwidth    : {sum(busbws) / len(busbws):.4f} ")
    lines.append("#")
    return "\n".join(lines) + "\n"


class SimCluster:
    def __init__(self, node_type: NodeType, nnodes: int, faults: list[Fault], seed: int = 0):
        self.nt = node_type
        self.nodes = node_names(nnodes)
        self.faults = faults
        self.rng = random.Random(seed)
        self.env = dict(BASE_ENV, **(ROCE_ENV if node_type.is_roce else {}))
        for f in faults:
            if f.node not in self.nodes:
                raise ValueError(f"fault {f} names unknown node; have {self.nodes[0]}..{self.nodes[-1]}")
            if f.rail is not None and not 0 <= f.rail < len(node_type.hcas):
                raise ValueError(f"fault {f}: rail out of range")
            if f.kind in ("pfc", "gid") and not node_type.is_roce:
                raise ValueError(f"fault {f} only applies to RoCE node types")

    # ---------------------------------------------------------------- fault lookup
    def _has(self, kind: str, node: str, rail: int | None = None) -> bool:
        return any(f.kind == kind and f.node == node and (rail is None or f.rail in (None, rail)) for f in self.faults)

    def endpoint_bw_factor(self, node: str, rail: int) -> float:
        if self._has("gid", node):
            return SOCKET_GBPS / (self.nt.nic_GBps * self.nt.efficiency["net_p2p"])
        factor = 1.0
        if self._has("downtrain", node, rail):
            factor = min(factor, 0.5)
        if self._has("numa", node, rail):
            factor = min(factor, 0.62)
        if self._has("pfc", node, rail):
            factor = min(factor, 0.92)
        return factor

    def endpoint_lat(self, node: str, rail: int) -> tuple[float, tuple[float, float]]:
        if self._has("gid", node):
            return 3.0, (1.0, 1.2)
        if self._has("pfc", node, rail):
            return 1.0, (5.0, 9.0)  # go-back-N retransmits after drops: small messages stall
        return 1.0, (1.0, 1.0)

    # ---------------------------------------------------------------- writers
    def write(self, out: Path) -> Path:
        out.mkdir(parents=True, exist_ok=True)
        for node_idx, node in enumerate(self.nodes):
            nd = out / "nodes" / node
            nd.mkdir(parents=True, exist_ok=True)
            _write(nd / "ibstat.txt", self.ibstat(node))
            _write(nd / "topo.txt", self.topo(node))
            _write(nd / "intra_all_reduce_perf.txt", self.intra_all_reduce(node))
            if self.nt.is_roce:
                _write(nd / "ibdev2netdev.txt", self.ibdev2netdev())
                _write(nd / "show_gids.txt", self.show_gids(node, node_idx))
                for rail in range(len(self.nt.hcas)):
                    _write(nd / f"ethtool-{netdev(rail)}.txt", self.ethtool(node, rail))
            else:
                for rail, hca in enumerate(self.nt.hcas):
                    _write(nd / f"perfquery-{hca}.txt", self.perfquery(node, rail, node_idx))
        pd = out / "pairs"
        pd.mkdir(exist_ok=True)
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1:]:
                for rail in range(len(self.nt.hcas)):
                    _write(pd / f"{a}__{b}__rail{rail}.sendrecv_perf.txt", self.pair_sendrecv(a, b, rail))
        cd = out / "collective"
        cd.mkdir(exist_ok=True)
        _write(cd / "all_reduce_perf.txt", self.full_all_reduce())
        _write(cd / "nccl_debug.log", self.nccl_debug_log())
        manifest = {
            "node_type": self.nt.name,
            "nodes": self.nodes,
            "nccl_env": self.env,
            "source": "simulated",
            "injected_faults": [str(f) for f in self.faults],
        }
        _write(out / "cluster.json", json.dumps(manifest, indent=2) + "\n")
        return out

    # ---------------------------------------------------------------- nccl-tests
    def pair_sendrecv(self, a: str, b: str, rail: int) -> str:
        shape = JobShape("sendrecv", nnodes=2, ranks_per_node=1)
        exp = expected_busbw(self.nt, shape)
        factor = min(self.endpoint_bw_factor(a, rail), self.endpoint_bw_factor(b, rail))
        lat_mult, jitter = max(self.endpoint_lat(a, rail), self.endpoint_lat(b, rail))
        base_lat = 11.0 * lat_mult
        peak = exp * factor * self.rng.uniform(0.97, 1.03)
        return format_nccl_tests("sendrecv", [(a, rail), (b, rail)], peak, base_lat, PAIR_SIZES, self.rng, jitter)

    def intra_all_reduce(self, node: str) -> str:
        exp = expected_busbw(self.nt, JobShape("all_reduce", nnodes=1, ranks_per_node=self.nt.gpus_per_node))
        wrong = frozenset(sizes(1 << 20, 8 << 30)[::3]) if self._has("corrupt", node) else frozenset()
        devices = [(node, d) for d in range(self.nt.gpus_per_node)]
        return format_nccl_tests("all_reduce", devices, exp * self.rng.uniform(0.97, 1.03), 18.0,
                                 FULL_SIZES, self.rng, wrong_sizes=wrong)

    def full_all_reduce(self) -> str:
        shape = JobShape("all_reduce", nnodes=len(self.nodes), ranks_per_node=self.nt.gpus_per_node)
        exp = expected_busbw(self.nt, shape)
        # NCCL splits every collective evenly across channels, one channel per rail, so the
        # slowest endpoint anywhere in the ring sets the pace for the whole job.
        worst = min(self.endpoint_bw_factor(n, r) for n in self.nodes for r in range(len(self.nt.hcas)))
        wrong = frozenset(sizes(1 << 20, 8 << 30)[::4]) if any(f.kind == "corrupt" for f in self.faults) else frozenset()
        devices = [(n, d) for n in self.nodes for d in range(self.nt.gpus_per_node)]
        return format_nccl_tests("all_reduce", devices, exp * worst * self.rng.uniform(0.97, 1.02),
                                 25.0 + 4.0 * len(self.nodes), FULL_SIZES, self.rng, wrong_sizes=wrong)

    # ---------------------------------------------------------------- NCCL_DEBUG=INFO
    def nccl_debug_log(self) -> str:
        g = self.nt.gpus_per_node
        nranks = g * len(self.nodes)
        lines: list[str] = []
        for ni, node in enumerate(self.nodes):
            pid = 51000 + ni * 100
            ip = f"10.0.0.{ni + 1}"
            pre = lambda dev: f"{node}:{pid + dev}:{pid + dev} [{dev}] NCCL"  # noqa: E731
            socket = self._has("gid", node)
            lines.append(f"{pre(0)} INFO Bootstrap : Using eth0:{ip}<0>")
            for k, v in self.env.items():
                if k != "NCCL_DEBUG":
                    lines.append(f"{pre(0)} INFO {k} set by environment to {v}")
            lines.append(f"{pre(0)} INFO NCCL version {NCCL_VERSION}")
            if socket:
                lines.append(f"{pre(0)} INFO NET/IB : No device found.")
                lines.append(f"{pre(0)} INFO NET/Socket : Using [0]eth0:{ip}<0>")
                lines.append(f"{pre(0)} INFO Using network Socket")
            else:
                tag = "RoCE" if self.nt.is_roce else "IB"
                devs = " ".join(f"[{i}]{h}:1/{tag}" for i, h in enumerate(self.nt.hcas))
                lines.append(f"{pre(0)} INFO NET/IB : Using {devs} [RO]; OOB eth0:{ip}<0>")
                lines.append(f"{pre(0)} INFO Using network IB")
            for d in range(g):
                rank = ni * g + d
                lines.append(f"{pre(d)} INFO comm 0x7f3a{ni:02x}{d:02x}00 rank {rank} nRanks {nranks} "
                             f"nNodes {len(self.nodes)} localRanks {g} localRank {d} MNNVL 0")
                if not socket:
                    if self._has("numa", node, d):
                        lines.append(f"{pre(d)} INFO GPU Direct RDMA Disabled for GPU {GPU_BUS_IDS[d]}000 / HCA {d} (distance 7 > 4)")
                    else:
                        lines.append(f"{pre(d)} INFO GPU Direct RDMA Enabled for GPU {GPU_BUS_IDS[d]}000 / HCA {d} (distance 4 <= 4), read 1")
                if rank == 0:
                    for ch in range(g * 2):
                        order = " ".join(f"{(r + ch) % nranks:3d}" for r in range(nranks))
                        lines.append(f"{pre(d)} INFO Channel {ch:02d}/{g * 2} : {order}")
                tree = " ".join(f"[{c}] {(rank + 1) % nranks}/-1/-1->{rank}->{(rank - 1) % nranks}" for c in range(g * 2))
                lines.append(f"{pre(d)} INFO Trees {tree}")
                peer_next = ((ni + 1) % len(self.nodes)) * g + d
                peer_prev = ((ni - 1) % len(self.nodes)) * g + d
                for ch in (d, d + g):
                    if socket:
                        via = "NET/Socket/0"
                    elif self._has("numa", node, d):
                        via = f"NET/IB/{d}"
                    else:
                        via = f"NET/IB/{d}/GDRDMA"
                    lines.append(f"{pre(d)} INFO Channel {ch:02d}/0 : {rank}[{d}] -> {peer_next}[{d}] [send] via {via}")
                    lines.append(f"{pre(d)} INFO Channel {ch:02d}/0 : {peer_prev}[{d}] -> {rank}[{d}] [receive] via {via}")
                    lines.append(f"{pre(d)} INFO Channel {ch:02d}/0 : {rank}[{d}] -> {ni * g + (d + 1) % g}[{(d + 1) % g}] via P2P/CUMEM/read")
                lines.append(f"{pre(d)} INFO comm 0x7f3a{ni:02x}{d:02x}00 rank {rank} nranks {nranks} cudaDev {d} "
                             f"nvmlDev {d} busId {GPU_BUS_IDS[d]}000 commId 0x5e1f00d - Init COMPLETE")
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------------- host tools
    def ibstat(self, node: str) -> str:
        out = []
        for rail, hca in enumerate(self.nt.hcas):
            rate = self.nt.nic_rate_gbps / 2 if self._has("downtrain", node, rail) else self.nt.nic_rate_gbps
            guid = f"0x946dae0300{self.nodes.index(node):02x}{rail:02x}c0"
            lid = 0 if self.nt.is_roce else 100 + self.nodes.index(node) * 8 + rail
            out += [
                f"CA '{hca}'",
                "\tCA type: MT4129",
                "\tNumber of ports: 1",
                "\tFirmware version: 28.42.1000",
                "\tHardware version: 0",
                f"\tNode GUID: {guid}",
                f"\tSystem image GUID: {guid}",
                "\tPort 1:",
                "\t\tState: Active",
                "\t\tPhysical state: LinkUp",
                f"\t\tRate: {rate:g}",
                f"\t\tBase lid: {lid}",
                "\t\tLMC: 0",
                f"\t\tSM lid: {0 if self.nt.is_roce else 1}",
                "\t\tCapability mask: 0xa651e848",
                f"\t\tPort GUID: {guid}",
                f"\t\tLink layer: {self.nt.link_layer}",
            ]
        return "\n".join(out) + "\n"

    def perfquery(self, node: str, rail: int, node_idx: int) -> str:
        bad = self._has("downtrain", node, rail)
        lid = 100 + node_idx * 8 + rail
        c = {
            "PortSelect": "1", "CounterSelect": "0x0000",
            "SymbolErrorCounter": 1874 if bad else 0,
            "LinkErrorRecoveryCounter": 5 if bad else 0,
            "LinkDownedCounter": 2 if bad else 0,
            "PortRcvErrors": 0, "PortRcvRemotePhysicalErrors": 0, "PortRcvSwitchRelayErrors": 0,
            "PortXmitDiscards": 0, "PortXmitConstraintErrors": 0, "PortRcvConstraintErrors": 0,
            "CounterSelect2": "0x00", "LocalLinkIntegrityErrors": 0, "ExcessiveBufferOverrunErrors": 0,
            "QP1Dropped": 0, "VL15Dropped": 0,
            "PortXmitData": 3841873024, "PortRcvData": 3841902112,
            "PortXmitPkts": 15007312, "PortRcvPkts": 15007431, "PortXmitWait": 120443,
        }
        lines = [f"# Port counters: Lid {lid} port 1 (CapMask: 0x5A00)"]
        for k, v in c.items():
            lines.append(f"{k + ':':.<33}{v}")
        return "\n".join(lines) + "\n"

    def ethtool(self, node: str, rail: int) -> str:
        pfc_bad = self._has("pfc", node, rail)
        down = self._has("downtrain", node, rail)
        r = self.rng
        stats = {
            "rx_packets": 98123771 + r.randint(0, 9999), "tx_packets": 98120012 + r.randint(0, 9999),
            "rx_bytes": 402184467712, "tx_bytes": 402170112000,
            "rx_vport_rdma_unicast_packets": 97001234, "tx_vport_rdma_unicast_packets": 96998812,
            "rx_ecn_mark": 0 if pfc_bad else 1500 + r.randint(0, 900),
            "rx_out_of_buffer": 0,
            "rx_crc_errors_phy": 0,
            "rx_symbol_err_phy": 312 if down else 0,
            "rx_discards_phy": 2148331 if pfc_bad else 0,
            "tx_discards_phy": 0,
            "rx_pause_ctrl_phy": 0, "tx_pause_ctrl_phy": 0,
        }
        for prio in range(8):
            lossless = prio == 3 and not pfc_bad
            stats[f"rx_prio{prio}_pause"] = 4000 + r.randint(0, 2000) if lossless else 0
            stats[f"rx_prio{prio}_pause_duration"] = 52000 + r.randint(0, 9000) if lossless else 0
            stats[f"tx_prio{prio}_pause"] = 3500 + r.randint(0, 2000) if lossless else 0
            stats[f"tx_prio{prio}_pause_duration"] = 48000 + r.randint(0, 9000) if lossless else 0
        return "NIC statistics:\n" + "".join(f"     {k}: {v}\n" for k, v in stats.items())

    def ibdev2netdev(self) -> str:
        return "".join(f"{hca} port 1 ==> {netdev(r)} (Up)\n" for r, hca in enumerate(self.nt.hcas))

    def show_gids(self, node: str, node_idx: int) -> str:
        lines = ["DEV\tPORT\tINDEX\tGID\t\t\t\t\tIPv4  \t\tVER\tDEV",
                 "---\t----\t-----\t---\t\t\t\t\t------------  \t---\t---"]
        n = 0
        for rail, hca in enumerate(self.nt.hcas):
            mac = f"0a5c:3eff:fe{node_idx:02x}:{rail:02x}01"
            ip = f"10.{100 + rail}.{node_idx // 250}.{node_idx % 250 + 1}"
            ip_hex = "{:02x}{:02x}:{:02x}{:02x}".format(*(int(x) for x in ip.split(".")))
            entries = [(f"fe80:0000:0000:0000:{mac}", None, "v1"), (f"fe80:0000:0000:0000:{mac}", None, "v2")]
            if self._has("gid", node):
                # A stray global IPv6 address on the netdev lands before the IPv4 entries.
                entries += [(f"fd00:0000:0000:00{rail:02x}:{mac}", None, "v1"), (f"fd00:0000:0000:00{rail:02x}:{mac}", None, "v2")]
            entries += [(f"0000:0000:0000:0000:0000:ffff:{ip_hex}", ip, "v1"), (f"0000:0000:0000:0000:0000:ffff:{ip_hex}", ip, "v2")]
            for idx, (gid, ipv4, ver) in enumerate(entries):
                ipcol = f"{ipv4}  " if ipv4 else ""
                lines.append(f"{hca}\t1\t{idx}\t{gid}\t{ipcol}\t{ver}\t{netdev(rail)}")
                n += 1
        lines.append(f"n_gids_found={n}")
        return "\n".join(lines) + "\n"

    def topo(self, node: str) -> str:
        g, k = self.nt.gpus_per_node, len(self.nt.hcas)
        gpu_sock = [i * 2 // g for i in range(g)]
        nic_sock = [(1 - (i * 2 // k)) if self._has("numa", node, i) else i * 2 // k for i in range(k)]
        cols = [f"GPU{i}" for i in range(g)] + [f"NIC{i}" for i in range(k)]
        head = "\t" + "\t".join(cols) + "\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID"
        rows = [head]

        def gpu_nic(i: int, j: int) -> str:
            if i == j and nic_sock[j] == gpu_sock[i]:
                return "PIX"
            return "NODE" if nic_sock[j] == gpu_sock[i] else "SYS"

        for i in range(g):
            cells = [" X " if j == i else "NV18" for j in range(g)] + [gpu_nic(i, j) for j in range(k)]
            cpus = "0-55,112-167" if gpu_sock[i] == 0 else "56-111,168-223"
            rows.append(f"GPU{i}\t" + "\t".join(cells) + f"\t{cpus}\t{gpu_sock[i]}\t\tN/A")
        for j in range(k):
            cells = [gpu_nic(i, j) for i in range(g)]
            cells += [" X " if m == j else ("NODE" if nic_sock[m] == nic_sock[j] else "SYS") for m in range(k)]
            rows.append(f"NIC{j}\t" + "\t".join(cells))
        rows += ["", "Legend:", "", "  X    = Self",
                 "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)",
                 "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node",
                 "  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)",
                 "  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)",
                 "  PIX  = Connection traversing at most a single PCIe bridge",
                 "  NV#  = Connection traversing a bonded set of # NVLinks", "", "NIC Legend:", ""]
        rows += [f"  NIC{j}: {h}" for j, h in enumerate(self.nt.hcas)]
        return "\n".join(rows) + "\n"


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding=ENCODING) as fh:
        fh.write(text)
