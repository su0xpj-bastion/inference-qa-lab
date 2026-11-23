from __future__ import annotations

# Fabric analysis over a run directory (real capture or simulated; same layout):
#
#   cluster.json                       node_type, nodes, nccl_env
#   nodes/<node>/ibstat.txt            + perfquery-<hca>.txt (IB) or ethtool-<netdev>.txt,
#                                        ibdev2netdev.txt, show_gids.txt (RoCE)
#   nodes/<node>/topo.txt              nvidia-smi topo -m
#   nodes/<node>/intra_all_reduce_perf.txt
#   pairs/<a>__<b>__rail<r>.sendrecv_perf.txt
#   collective/all_reduce_perf.txt     full job, all nodes
#   collective/nccl_debug.log          NCCL_DEBUG=INFO from the full job
#
# Stages follow the runbook order: node health -> single node -> pairs -> full collective.
# Then the localized suspects from the pair matrix are joined with the node evidence so a
# finding says what is wrong, not just where.

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from fabval import fabric, nccl_log, nccl_tests
from fabval.localize import PairTest, localize
from fabval.model import JobShape, NodeType, expected_busbw, latency_budget_us, load_thresholds

ENCODING = "utf-8"
PAIR_RE = re.compile(r"^(?P<a>.+?)__(?P<b>.+?)__rail(?P<rail>\d+)\.sendrecv_perf\.txt$")
EXIT_PASS, EXIT_FAIL, EXIT_INCONCLUSIVE = 0, 1, 2


@dataclass
class Finding:
    severity: str           # FAIL | WARN
    category: str           # bandwidth | latency | link | transport | affinity | lossless | gid | corruption | counters
    node: str | None
    hca: str | None
    message: str
    evidence: list[str] = field(default_factory=list)
    stage: str = ""

    @property
    def location(self) -> str:
        if self.node and self.hca:
            return f"{self.node}/{self.hca}"
        return self.node or "cluster"

    def to_dict(self) -> dict:
        return {"severity": self.severity, "category": self.category, "stage": self.stage,
                "node": self.node, "hca": self.hca, "location": self.location,
                "message": self.message, "evidence": self.evidence}


@dataclass
class Report:
    node_type: str
    nodes: list[str]
    source: str
    findings: list[Finding] = field(default_factory=list)
    root_causes: list[Finding] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    pair_matrix: dict = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if any(f.severity == "FAIL" for f in self.findings):
            return "FAIL"
        if self.errors:
            return "INCONCLUSIVE"
        return "PASS"

    @property
    def exit_code(self) -> int:
        return {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL, "INCONCLUSIVE": EXIT_INCONCLUSIVE}[self.verdict]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "exit_code": self.exit_code, "node_type": self.node_type,
            "nodes": self.nodes, "source": self.source, "stages": self.stages,
            "root_causes": [f.to_dict() for f in self.root_causes],
            "findings": [f.to_dict() for f in self.findings],
            "pair_matrix": self.pair_matrix, "errors": self.errors,
        }


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _read(path: Path) -> str:
    with path.open("r", encoding=ENCODING) as fh:
        return fh.read()


class Analyzer:
    def __init__(self, run_dir: Path, thresholds: dict | None = None):
        self.run = run_dir
        self.th = thresholds or load_thresholds()
        with (run_dir / "cluster.json").open("r", encoding=ENCODING) as fh:
            self.cluster = json.load(fh)
        self.nt = NodeType.load(self.cluster["node_type"])
        self.nodes: list[str] = self.cluster["nodes"]
        self.env: dict = self.cluster.get("nccl_env", {})
        self.report = Report(self.nt.name, self.nodes, self.cluster.get("source", "capture"))
        self.gates = self.th["gates"]

    # ------------------------------------------------------------------ helpers
    def _add(self, f: Finding) -> None:
        self.report.findings.append(f)

    def _load_tests(self, path: Path, collective: str) -> nccl_tests.NcclTestsResult | None:
        if not path.exists():
            self.report.errors.append(f"missing {path.relative_to(self.run)}")
            return None
        try:
            res = nccl_tests.parse(_read(path), collective)
        except nccl_tests.ParseError as exc:
            self.report.errors.append(f"{path.relative_to(self.run)}: {exc}")
            return None
        if not res.complete:
            self.report.errors.append(f"{path.relative_to(self.run)}: truncated (no 'Avg bus bandwidth' line)")
            return None
        if problems := res.consistency_problems():
            self.report.errors.append(f"{path.relative_to(self.run)}: busbw/algbw inconsistent: {problems[0]}")
            return None
        return res

    def _gate_bw(self, res: nccl_tests.NcclTestsResult, expected: float) -> tuple[float, float]:
        peak = res.peak_busbw(self.gates["large_min_bytes"]) or 0.0
        return peak, peak / expected if expected else 0.0

    # ------------------------------------------------------------------ stage 0: node health
    def stage_node_health(self) -> None:
        nt, failed_nodes = self.nt, set()
        for node in self.nodes:
            nd = self.run / "nodes" / node
            problems: list[fabric.Problem] = []
            if (nd / "ibstat.txt").exists():
                ports = fabric.parse_ibstat(_read(nd / "ibstat.txt"))
                problems += fabric.check_ib_ports(ports, nt.hcas, nt.nic_rate_gbps, nt.link_layer)
            else:
                self.report.errors.append(f"missing nodes/{node}/ibstat.txt")
            if nt.is_roce:
                n2n = fabric.parse_ibdev2netdev(_read(nd / "ibdev2netdev.txt")) if (nd / "ibdev2netdev.txt").exists() else {}
                for hca in nt.hcas:
                    nd_name = n2n.get(hca, {}).get("netdev")
                    path = nd / f"ethtool-{nd_name}.txt" if nd_name else None
                    if path is None or not path.exists():
                        self.report.errors.append(f"missing ethtool -S for {node}/{hca}")
                        continue
                    stats = fabric.parse_ethtool_stats(_read(path))
                    problems += fabric.check_ethtool(hca, stats, self.th["ethtool"], self.th["roce"]["priority"])
                gid_index = self.env.get("NCCL_IB_GID_INDEX")
                if gid_index is not None and (nd / "show_gids.txt").exists():
                    gids = fabric.parse_show_gids(_read(nd / "show_gids.txt"))
                    problems += fabric.check_gid_index(gids, nt.hcas, int(gid_index), self.th["roce"]["gid_version"])
            else:
                for hca in nt.hcas:
                    path = nd / f"perfquery-{hca}.txt"
                    if not path.exists():
                        self.report.errors.append(f"missing nodes/{node}/perfquery-{hca}.txt")
                        continue
                    problems += fabric.check_perfquery(hca, fabric.parse_perfquery(_read(path)), self.th["perfquery"])
            if (nd / "topo.txt").exists():
                topo = fabric.parse_topo(_read(nd / "topo.txt"))
                problems += fabric.check_gpu_nic_affinity(topo, nt.hcas, self.th["topo"]["allowed_gpu_nic_paths"])
            else:
                self.report.errors.append(f"missing nodes/{node}/topo.txt")
            category = {"port_rate": "link", "port_state": "link", "port_phys": "link", "port_present": "link",
                        "link_layer": "link", "ethtool:drops": "lossless", "ethtool:phy": "counters", "gid_index": "gid",
                        "gpu_nic_affinity": "affinity", "topo": "affinity"}
            for p in problems:
                cat = "counters" if p.check.startswith("perfquery") else category.get(p.check, p.check)
                self._add(Finding(p.severity, cat, node, p.hca, p.message, stage="node health"))
                if p.severity == "FAIL":
                    failed_nodes.add(node)
        self.report.stages["node_health"] = {"nodes": len(self.nodes), "nodes_failed": sorted(failed_nodes)}

    # ------------------------------------------------------------------ stage 1: single node
    def stage_single_node(self) -> None:
        exp = expected_busbw(self.nt, JobShape("all_reduce", 1, self.nt.gpus_per_node))
        budget = latency_budget_us(self.nt, "all_reduce_intra")
        rows = {}
        for node in self.nodes:
            res = self._load_tests(self.run / "nodes" / node / "intra_all_reduce_perf.txt", "all_reduce")
            if res is None:
                continue
            peak, ratio = self._gate_bw(res, exp)
            small = _median(res.small_times_us(self.gates["small_max_bytes"]))
            rows[node] = {"peak_busbw": round(peak, 1), "pct_expected": round(100 * ratio, 1),
                          "small_median_us": round(small, 1), "wrong": res.total_wrong}
            if res.total_wrong:
                bad = [r.size for r in res.rows if r.wrong]
                self._add(Finding("FAIL", "corruption", node, None,
                                  f"intra-node all_reduce returned wrong results ({res.total_wrong} elements across {len(bad)} sizes)",
                                  [f"#wrong > 0 at sizes {bad[:4]}{'...' if len(bad) > 4 else ''}"], "single node"))
            if ratio < self.gates["busbw_pct_of_expected"]:
                self._add(Finding("FAIL", "bandwidth", node, None,
                                  f"NVLink all_reduce busbw {peak:.0f} GB/s = {100 * ratio:.0f}% of expected {exp:.0f}",
                                  stage="single node"))
            if small > budget:
                self._add(Finding("FAIL", "latency", node, None,
                                  f"intra-node small-message all_reduce median {small:.1f} us > budget {budget:.0f} us",
                                  stage="single node"))
        self.report.stages["single_node"] = {"expected_busbw": round(exp, 1), "nodes": rows}

    # ------------------------------------------------------------------ stage 2: pairs
    def stage_pairs(self) -> None:
        exp = expected_busbw(self.nt, JobShape("sendrecv", 2, 1))
        budget = latency_budget_us(self.nt, "sendrecv_pair")
        pct = self.gates["busbw_pct_of_expected"]
        bw_tests: list[PairTest] = []
        lat_tests: list[PairTest] = []
        matrix: dict[str, dict] = {}
        pdir = self.run / "pairs"
        files = sorted(pdir.glob("*.sendrecv_perf.txt")) if pdir.exists() else []
        want = {(a, b, r) for i, a in enumerate(self.nodes) for b in self.nodes[i + 1:] for r in range(len(self.nt.hcas))}
        seen = set()
        for f in files:
            m = PAIR_RE.match(f.name)
            if not m:
                continue
            a, b, rail = m["a"], m["b"], int(m["rail"])
            res = self._load_tests(f, "sendrecv")
            if res is None:
                continue
            seen.add((a, b, rail))
            peak, ratio = self._gate_bw(res, exp)
            small = _median(res.small_times_us(self.gates["small_max_bytes"]))
            bw_fail = ratio < pct
            lat_fail = small > budget
            bw_tests.append(PairTest(a, b, rail, bw_fail, f"{peak:.1f} GB/s ({100 * ratio:.0f}%)"))
            lat_tests.append(PairTest(a, b, rail, lat_fail, f"{small:.1f} us"))
            key = f"{a}|{b}"
            matrix.setdefault(key, {"min_pct": 101.0, "max_small_us": 0.0, "rails_bw_fail": [], "rails_lat_fail": []})
            cell = matrix[key]
            cell["min_pct"] = round(min(cell["min_pct"], 100 * ratio), 1)
            cell["max_small_us"] = round(max(cell["max_small_us"], small), 1)
            if bw_fail:
                cell["rails_bw_fail"].append(rail)
            if lat_fail:
                cell["rails_lat_fail"].append(rail)
        if missing := sorted(want - seen):
            self.report.errors.append(f"{len(missing)} of {len(want)} pair tests missing or unreadable, e.g. {missing[0]}")
        self.report.pair_matrix = matrix

        rails = len(self.nt.hcas)
        for kind, tests, limit in (("bandwidth", bw_tests, f"{pct:.0%} of {exp:.1f} GB/s"), ("latency", lat_tests, f"{budget:.0f} us")):
            loc = localize(tests, rails)
            for s in loc.suspects:
                hca = None if s.rail is None else self.nt.hcas[s.rail]
                where = "every rail" if s.rail is None else f"rail {s.rail} ({hca})"
                details = [t.detail for t in tests if t.failed and s.node in (t.a, t.b) and (s.rail is None or t.rail == s.rail)]
                self._add(Finding("FAIL", kind, s.node, hca,
                                  f"{where} fails the pair {kind} gate in {s.failed}/{s.tested} tests (limit {limit})",
                                  [f"partners: {', '.join(s.partners[:6])}{'...' if len(s.partners) > 6 else ''}",
                                   f"measured: {', '.join(details[:4])}"], "pairs"))
            for t in loc.unexplained_links:
                hca = self.nt.hcas[t.rail]
                self._add(Finding("FAIL", kind, None, hca,
                                  f"link {t.a}/{hca} <-> {t.b}/{hca} fails the pair {kind} gate ({t.detail}); "
                                  "only this pair, so suspect the cable, optics or switch port between them",
                                  stage="pairs"))
        self.report.stages["pairs"] = {
            "expected_busbw": round(exp, 1), "latency_budget_us": budget, "tests": len(bw_tests),
            "bw_failed": sum(t.failed for t in bw_tests), "lat_failed": sum(t.failed for t in lat_tests),
        }

    # ------------------------------------------------------------------ stage 3: full collective
    def stage_collective(self) -> None:
        cdir = self.run / "collective"
        shape = JobShape("all_reduce", len(self.nodes), self.nt.gpus_per_node)
        exp = expected_busbw(self.nt, shape)
        res = self._load_tests(cdir / "all_reduce_perf.txt", "all_reduce")
        stage: dict = {"expected_busbw": round(exp, 1)}
        if res is not None:
            if res.nranks != shape.nranks:
                self.report.errors.append(f"collective ran on {res.nranks} ranks, expected {shape.nranks}")
            peak, ratio = self._gate_bw(res, exp)
            stage.update({"peak_busbw": round(peak, 1), "pct_expected": round(100 * ratio, 1),
                          "avg_busbw": res.avg_busbw, "wrong": res.total_wrong, "nranks": res.nranks})
            if ratio < self.gates["busbw_pct_of_expected"]:
                self._add(Finding("FAIL", "bandwidth", None, None,
                                  f"full all_reduce busbw {peak:.1f} GB/s = {100 * ratio:.0f}% of expected {exp:.0f} GB/s "
                                  f"({len(self.nodes)} nodes x {self.nt.gpus_per_node} GPUs, ring)",
                                  ["the slowest channel gates every rank: one bad rail anywhere caps the whole job"],
                                  "full collective"))
            if res.total_wrong:
                self._add(Finding("FAIL", "corruption", None, None,
                                  f"full all_reduce returned wrong results ({res.total_wrong} elements)",
                                  ["see single-node stage for the node"], "full collective"))

        log_path = cdir / "nccl_debug.log"
        if not log_path.exists():
            self.report.errors.append("missing collective/nccl_debug.log")
        else:
            hosts = nccl_log.parse(_read(log_path))
            stage["nccl"] = {h: s.to_dict() for h, s in sorted(hosts.items())}
            for node in self.nodes:
                if node not in hosts:
                    self.report.errors.append(f"no NCCL_DEBUG lines from {node}")
            env_vals: dict[str, Counter] = {}
            for s in hosts.values():
                for k, v in s.env.items():
                    env_vals.setdefault(k, Counter())[v] += 1
            for host, s in sorted(hosts.items()):
                if s.socket_fallback:
                    ev = ["'NET/IB : No device found.'"] if s.ib_not_found else []
                    ev.append(f"networks={sorted(s.networks)} transports={sorted(s.transports)}")
                    self._add(Finding("FAIL", "transport", host, None,
                                      "NCCL fell back to TCP sockets (NET/Socket): the RDMA fabric is bypassed", ev,
                                      "full collective"))
                for idx in s.gdr_off_hcas:
                    hca = self.nt.hcas[idx] if idx < len(self.nt.hcas) else f"HCA {idx}"
                    lines = [d["line"] for d in s.gdr_decisions if d["hca"] == idx and d["state"] == "Disabled"]
                    self._add(Finding("WARN", "affinity", host, hca,
                                      f"GPUDirect RDMA off on {hca}: connections run via NET/IB/{idx} without /GDRDMA",
                                      lines[:1], "full collective"))
                for k, v in s.env.items():
                    common, _ = env_vals[k].most_common(1)[0]
                    if v != common:
                        self._add(Finding("FAIL", "config", host, None,
                                          f"{k}={v} differs from the rest of the job ({common})", stage="full collective"))
                for w in s.warnings[:3]:
                    self._add(Finding("WARN", "nccl", host, None, f"NCCL WARN: {w}", stage="full collective"))
        self.report.stages["collective"] = stage

    # ------------------------------------------------------------------ correlation
    CAUSES = {
        # (symptom category) -> ordered node-evidence categories that explain it
        "bandwidth": ("transport", "gid", "link", "affinity", "lossless", "counters"),
        "latency": ("transport", "gid", "lossless", "link", "counters"),
    }
    CAUSE_TEXT = {
        "transport": "transport fallback to TCP sockets",
        "gid": "RoCE GID index mismatch",
        "link": "link down-trained / port not healthy",
        "affinity": "NIC on the wrong NUMA node (GDR off)",
        "lossless": "RoCE lossless config (PFC/ECN) not in effect",
        "counters": "physical-layer errors",
    }

    def correlate(self) -> None:
        fs = self.report.findings
        evidence = [f for f in fs if f.stage in ("node health", "full collective") and f.node]
        used: set[int] = set()
        causes: list[Finding] = []
        for sym in [f for f in fs if f.stage == "pairs" and f.node]:
            matched: list[Finding] = []
            for cat in self.CAUSES[sym.category]:
                for e in evidence:
                    if e.category == cat and e.node == sym.node and (sym.hca is None or e.hca in (None, sym.hca)):
                        matched.append(e)
                if matched:
                    break
            if matched:
                cat = matched[0].category
                if cat == "transport" and any(e.category == "gid" and e.node == sym.node for e in evidence):
                    cat = "gid"  # GID mismatch is the reason the transport fell back
                    matched += [e for e in evidence if e.category == "gid" and e.node == sym.node and e not in matched]
                if cat == "affinity":
                    matched += [e for e in evidence if e.category == "affinity" and e.node == sym.node
                                and e.hca == sym.hca and e not in matched]
                text = self.CAUSE_TEXT[cat]
            else:
                cat = sym.category
                text = f"unexplained {sym.category} degradation (no host-side evidence)"
            key = (sym.node, sym.hca, text)
            if any((c.node, c.hca, c.message) == key for c in causes):
                continue
            used.update(id(e) for e in matched)
            causes.append(Finding("FAIL", cat, sym.node, sym.hca, text,
                                  [sym.message] + [e.message for e in matched], "root cause"))
        # Node evidence with no matching pair symptom still matters (e.g. corruption, a port the
        # pair sweep never exercised): promote FAIL-level node findings that nothing explained.
        for e in fs:
            if e.severity == "FAIL" and e.node and e.stage in ("node health", "single node", "full collective") \
                    and id(e) not in used and not any(c.node == e.node and (c.hca in (None, e.hca)) for c in causes):
                causes.append(Finding("FAIL", e.category, e.node, e.hca, e.message, [], "root cause"))
        causes.sort(key=lambda c: (c.node or "", c.hca or ""))
        self.report.root_causes = causes

    def run_all(self) -> Report:
        self.stage_node_health()
        self.stage_single_node()
        self.stage_pairs()
        self.stage_collective()
        self.correlate()
        return self.report


def analyze(run_dir: Path) -> Report:
    return Analyzer(run_dir).run_all()
