from __future__ import annotations

# Parsers and threshold checks for host-side fabric tools:
#   ibstat, ibstatus, perfquery, ibdev2netdev   (rdma-core / infiniband-diags)
#   ethtool -S                                   (mlx5 RoCE counters)
#   show_gids                                    (MLNX_OFED / DOCA helper)
#   nvidia-smi topo -m                           (GPU <-> NIC PCIe/NUMA distance)

import re
from dataclasses import dataclass, field

ENCODING = "utf-8"


@dataclass(frozen=True)
class Problem:
    severity: str  # FAIL | WARN
    check: str
    hca: str | None
    message: str


# ---------------------------------------------------------------- ibstat / ibstatus

@dataclass
class IbPort:
    hca: str
    port: int
    state: str = ""
    physical_state: str = ""
    rate_gbps: float = 0.0
    link_layer: str = ""
    extra: dict = field(default_factory=dict)


def parse_ibstat(text: str) -> dict[str, IbPort]:
    """`ibstat` output -> {"mlx5_0": IbPort, ...}. Only port 1 of each CA is kept as the
    dataplane port (HCAs here are single-port); further ports are keyed "mlx5_0:2"."""
    ports: dict[str, IbPort] = {}
    ca: str | None = None
    cur: IbPort | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if m := re.match(r"^CA '([^']+)'", line):
            ca, cur = m.group(1), None
        elif (m := re.match(r"^Port (\d+):$", line)) and ca:
            port = int(m.group(1))
            cur = IbPort(hca=ca, port=port)
            ports[ca if port == 1 else f"{ca}:{port}"] = cur
        elif cur and ":" in line:
            key, val = (s.strip() for s in line.split(":", 1))
            if key == "State":
                cur.state = val
            elif key == "Physical state":
                cur.physical_state = val
            elif key == "Rate":
                cur.rate_gbps = float(val)
            elif key == "Link layer":
                cur.link_layer = val
            else:
                cur.extra[key] = val
    return ports


def parse_ibstatus(text: str) -> dict[str, IbPort]:
    """`ibstatus` output -> same shape as parse_ibstat."""
    ports: dict[str, IbPort] = {}
    cur: IbPort | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if m := re.match(r"^Infiniband device '([^']+)' port (\d+) status:", line):
            hca, port = m.group(1), int(m.group(2))
            cur = IbPort(hca=hca, port=port)
            ports[hca if port == 1 else f"{hca}:{port}"] = cur
        elif cur and ":" in line:
            key, val = (s.strip() for s in line.split(":", 1))
            if key == "state":
                cur.state = val.split(":", 1)[-1].strip().title()  # "4: ACTIVE" -> "Active"
            elif key == "phys state":
                cur.physical_state = val.split(":", 1)[-1].strip()
            elif key == "rate":
                cur.rate_gbps = float(val.split()[0])
                cur.extra["rate_text"] = val
            elif key == "link_layer":
                cur.link_layer = val
            else:
                cur.extra[key] = val
    return ports


def check_ib_ports(ports: dict[str, IbPort], hcas: tuple[str, ...], rate_gbps: float, link_layer: str) -> list[Problem]:
    out: list[Problem] = []
    for hca in hcas:
        p = ports.get(hca)
        if p is None:
            out.append(Problem("FAIL", "port_present", hca, f"{hca} missing from ibstat"))
            continue
        if p.state != "Active":
            out.append(Problem("FAIL", "port_state", hca, f"{hca} state {p.state or '?'} (want Active)"))
        if p.physical_state != "LinkUp":
            out.append(Problem("FAIL", "port_phys", hca, f"{hca} physical state {p.physical_state or '?'} (want LinkUp)"))
        if p.rate_gbps and p.rate_gbps < rate_gbps:
            out.append(Problem("FAIL", "port_rate", hca, f"{hca} rate {p.rate_gbps:g} Gb/s, expected {rate_gbps:g} (link down-trained)"))
        if p.link_layer and p.link_layer != link_layer:
            out.append(Problem("FAIL", "link_layer", hca, f"{hca} link layer {p.link_layer}, expected {link_layer}"))
    return out


def parse_ibdev2netdev(text: str) -> dict[str, dict]:
    """`ibdev2netdev` lines: "mlx5_0 port 1 ==> ens1f0np0 (Up)"."""
    out: dict[str, dict] = {}
    for line in text.splitlines():
        if m := re.match(r"^\s*(\S+) port (\d+) ==> (\S+) \((\w+)\)", line):
            out[m.group(1)] = {"port": int(m.group(2)), "netdev": m.group(3), "state": m.group(4)}
    return out


# ---------------------------------------------------------------- perfquery

def parse_perfquery(text: str) -> dict[str, int]:
    """`perfquery [-x] <lid> <port>` -> {"SymbolErrorCounter": 0, ...}. Dotted padding is stripped."""
    counters: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip(". \t")
        if not key.strip() or not val:
            continue
        try:
            counters[key.strip()] = int(val, 0)
        except ValueError:
            continue
    return counters


def check_perfquery(hca: str, counters: dict[str, int], limits: dict[str, int]) -> list[Problem]:
    out: list[Problem] = []
    for name, limit in limits.items():
        val = counters.get(name)
        if val is not None and val > limit:
            out.append(Problem("FAIL", f"perfquery:{name}", hca, f"{hca} {name}={val} (limit {limit})"))
    return out


# ---------------------------------------------------------------- ethtool -S (RoCE)

PHY_ERROR_COUNTERS = ("rx_crc_errors_phy", "rx_symbol_err_phy")


def parse_ethtool_stats(text: str) -> dict[str, int]:
    """`ethtool -S <netdev>` -> {"rx_prio3_pause": 123, ...}."""
    stats: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line or line.strip().startswith("NIC statistics"):
            continue
        key, val = line.split(":", 1)
        try:
            stats[key.strip()] = int(val.strip())
        except ValueError:
            continue
    return stats


def check_ethtool(hca: str, stats: dict[str, int], limits: dict[str, int], priority: int) -> list[Problem]:
    """Drop counters over the limit fail the port. The PFC/ECN counters do not fail on their
    own (zero pause frames on an idle port is normal) but are attached as the explanation:
    on a lossless RoCE priority, drops with no pause frames mean PFC is not in effect, and
    drops with no ECN marks mean the switch is not marking before its buffers overflow."""
    out: list[Problem] = []
    drops, phy = [], []
    for name, limit in limits.items():
        val = stats.get(name)
        if val is not None and val > limit:
            (phy if name in PHY_ERROR_COUNTERS else drops).append(f"{name}={val}")
    if phy:
        out.append(Problem("FAIL", "ethtool:phy", hca, f"{hca} physical-layer errors: {', '.join(phy)}"))
    if drops:
        pause_rx = stats.get(f"rx_prio{priority}_pause", 0)
        pause_tx = stats.get(f"tx_prio{priority}_pause", 0)
        ecn = stats.get("rx_ecn_mark", 0)
        why = []
        if pause_rx == 0 and pause_tx == 0:
            why.append(f"no PFC pause frames on priority {priority}")
        if ecn == 0:
            why.append("no ECN marks")
        hint = f"; {', '.join(why)} -> lossless config (PFC/ECN) not in effect" if why else ""
        out.append(Problem("FAIL", "ethtool:drops", hca, f"{hca} {', '.join(drops)}{hint}"))
    return out


# ---------------------------------------------------------------- show_gids (RoCE)

@dataclass(frozen=True)
class Gid:
    dev: str
    port: int
    index: int
    gid: str
    ipv4: str | None
    version: str
    netdev: str


def parse_show_gids(text: str) -> list[Gid]:
    out: list[Gid] = []
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 5 or not toks[1].isdigit() or not toks[2].isdigit():
            continue
        if len(toks) == 7:
            dev, port, idx, gid, ipv4, ver, netdev = toks
        elif len(toks) == 6:
            dev, port, idx, gid, ver, netdev = toks
            ipv4 = None
        else:
            continue
        out.append(Gid(dev, int(port), int(idx), gid, ipv4, ver, netdev))
    return out


def check_gid_index(gids: list[Gid], hcas: tuple[str, ...], gid_index: int, want_version: str = "v2") -> list[Problem]:
    """NCCL_IB_GID_INDEX is one number for the whole job, but GID tables are per host. An extra
    VLAN or IPv6 address on one node shifts its table, and the cluster-wide index then points
    at a RoCEv1 or link-local GID on that node only."""
    out: list[Problem] = []
    table = {(g.dev, g.index): g for g in gids}
    for hca in hcas:
        g = table.get((hca, gid_index))
        good = [x.index for x in gids if x.dev == hca and x.version == want_version and x.ipv4]
        if g is None or g.version != want_version or not g.ipv4:
            got = "missing" if g is None else f"{g.version} {'IPv4 ' + g.ipv4 if g.ipv4 else 'link-local/IPv6'}"
            want = f"RoCE {want_version} IPv4 GID is at index {good[0]}" if good else "no RoCE v2 IPv4 GID present"
            out.append(Problem("FAIL", "gid_index", hca, f"{hca} GID index {gid_index} is {got}; {want}"))
    return out


# ---------------------------------------------------------------- nvidia-smi topo -m

def parse_topo(text: str) -> dict:
    """`nvidia-smi topo -m` -> {"matrix": {row: {col: rel}}, "nic_names": {"NIC0": "mlx5_0"},
    "numa": {"GPU0": "0"}}."""
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    header: list[str] | None = None
    matrix: dict[str, dict[str, str]] = {}
    numa: dict[str, str] = {}
    nic_names: dict[str, str] = {}
    for line in lines:
        cells = [c.strip() for c in line.split("\t")]
        if header is None:
            if cells and cells[0] == "" and any(c.startswith("GPU") for c in cells):
                header = [c for c in cells[1:] if c]
            continue
        if m := re.match(r"^\s*(NIC\d+):\s*(\S+)", line):
            nic_names[m.group(1)] = m.group(2)
            continue
        if not cells or not re.match(r"^(GPU|NIC)\d+$", cells[0]):
            continue
        row = cells[0]
        vals = [c for c in cells[1:] if c != ""]
        matrix[row] = {}
        for i, col in enumerate(header):
            if i < len(vals) and re.match(r"^(GPU|NIC)\d+$", col):
                matrix[row][col] = vals[i]
            elif col == "NUMA Affinity" and i < len(vals):
                numa[row] = vals[i]
    return {"matrix": matrix, "nic_names": nic_names, "numa": numa}


def check_gpu_nic_affinity(topo: dict, hcas: tuple[str, ...], allowed: list[str]) -> list[Problem]:
    """Rail-optimized nodes pair GPU i with HCA i behind the same PCIe switch (PIX/PXB). NODE or
    SYS means the traffic crosses the CPU root complex or the socket interconnect, and NCCL's
    default NCCL_NET_GDR_LEVEL (PXB) turns GPUDirect RDMA off for that pair."""
    out: list[Problem] = []
    by_name = {name: nic for nic, name in topo["nic_names"].items()}
    for i, hca in enumerate(hcas):
        gpu, nic = f"GPU{i}", by_name.get(hca)
        if nic is None:
            out.append(Problem("WARN", "topo", hca, f"{hca} not in nvidia-smi topo NIC legend"))
            continue
        rel = topo["matrix"].get(gpu, {}).get(nic)
        if rel not in allowed:
            out.append(Problem("FAIL", "gpu_nic_affinity", hca,
                               f"{gpu} <-> {nic} ({hca}) is {rel} (want {'/'.join(allowed)}): NIC on the wrong PCIe root / NUMA node"))
    return out
