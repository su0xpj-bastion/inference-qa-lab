from __future__ import annotations

# Staged acceptance: node -> burn-in -> domain (NVLink/rack) -> cluster.
#
# A node that fails a stage is cordoned with the stage, the failing checks and
# their reasons, and does not enter later stages. Nothing is dropped silently:
# every node in the input appears in the verdict, accepted or cordoned.

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from bringup.allreduce import AllReduceRunner
from bringup.gates import BURNIN_GATES, DOMAIN_GATES, NODE_GATES, median
from bringup.model import FAIL, PASS, Check, NodeFacts
from bringup.parsers import nccl
from bringup.parsers.common import ENCODING

STAGES = ("node", "burnin", "domain", "cluster")
ACCEPTED, DEGRADED, REJECTED = "ACCEPTED", "ACCEPTED_DEGRADED", "REJECTED"
EXIT_CODES = {ACCEPTED: 0, REJECTED: 1, DEGRADED: 3}  # 2 is argparse's usage error


@dataclass
class NodeVerdict:
    hostname: str
    rack: str
    node_type: str
    vendor: str | None
    status: str = "accepted"
    cordon_stage: str | None = None
    reasons: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)

    def cordon(self, stage: str, failed: list[Check]) -> None:
        self.status = "cordoned"
        self.cordon_stage = stage
        self.failed_checks = [c.name for c in failed]
        self.reasons = [f"{c.name}: {c.reason}" for c in failed]

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname, "rack": self.rack, "node_type": self.node_type,
            "vendor": self.vendor, "status": self.status, "cordon_stage": self.cordon_stage,
            "failed_checks": self.failed_checks, "reasons": self.reasons,
            "stages_run": self.stages_run, "checks": [c.to_dict() for c in self.checks],
        }


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        doc = json.load(fh)
    if doc.get("schema") != "golden-manifest/v1":
        raise ValueError(f"{path}: unsupported manifest schema {doc.get('schema')!r}")
    return doc


def _run_stage(stage: str, gates, facts: NodeFacts, spec: dict, th: dict, v: NodeVerdict) -> bool:
    v.stages_run.append(stage)
    results = [g(facts, spec, th) for g in gates]
    v.checks.extend(results)
    failed = [c for c in results if c.status == FAIL]
    if failed:
        v.cordon(stage, failed)
        return False
    return True


def rack_findings(verdicts: list[NodeVerdict], th: dict) -> list[dict]:
    """Same check failing on many nodes of one rack points at a shared cause."""
    by_rack: dict[str, list[NodeVerdict]] = defaultdict(list)
    for v in verdicts:
        by_rack[v.rack].append(v)
    findings = []
    for rack, vs in sorted(by_rack.items()):
        counts = Counter(c for v in vs for c in v.failed_checks)
        for check, n in sorted(counts.items()):
            if n >= th["rack_correlation_min_nodes"] and n / len(vs) >= th["rack_correlation_min_fraction"]:
                hosts = [v.hostname for v in vs if check in v.failed_checks]
                findings.append({
                    "rack": rack, "check": check, "nodes": hosts, "rack_size": len(vs),
                    "reason": f"{n}/{len(vs)} nodes in rack {rack} fail {check}: suspect a shared cause "
                              "(leaf switch, cabling bundle, PDU, CDU) before RMAing nodes one by one",
                })
    return findings


def run_pipeline(fleet: list[NodeFacts], manifest: dict, runner: AllReduceRunner) -> dict:
    th = manifest["thresholds"]
    types = manifest["node_types"]
    verdicts: list[NodeVerdict] = []
    facts_by_host = {f.hostname: f for f in fleet}
    for f in fleet:
        v = NodeVerdict(f.hostname, f.rack, f.node_type, f.vendor)
        verdicts.append(v)
        spec = types.get(f.node_type)
        if spec is None:
            v.stages_run.append("node")
            chk = Check("node", "node_type_known", FAIL, f"node type {f.node_type!r} not in golden manifest")
            v.checks.append(chk)
            v.cordon("node", [chk])
            continue
        for stage, gates in (("node", NODE_GATES), ("burnin", BURNIN_GATES), ("domain", DOMAIN_GATES)):
            if not _run_stage(stage, gates, f, spec, th, v):
                break

    racks = rack_findings(verdicts, th)

    # ---------------------------------------------------------- cluster stage
    cluster_checks: list[dict] = []
    pools: dict[str, dict] = {}
    for pool, spec in sorted(types.items()):
        members = [v for v in verdicts if v.node_type == pool]
        alive = [v for v in members if v.status == "accepted"]
        for v in alive:
            v.stages_run.append("cluster")

        # Performance outlier: a node can pass every absolute floor and still be
        # the slowest rank of every collective. Only the fleet shows that.
        perf = {}
        for v in alive:
            ev = next((c.evidence for c in v.checks if c.name == "burnin_perf"), {})
            if ev.get("node_median_tflops"):
                perf[v.hostname] = ev["node_median_tflops"]
        pct = th["straggler_max_below_median_pct"]
        if len(perf) >= 3:
            pool_med = median(list(perf.values()))
            for v in alive:
                t = perf.get(v.hostname)
                if t is not None and t < pool_med * (1 - pct / 100):
                    chk = Check("cluster", "performance_outlier", FAIL,
                                f"node median GEMM {t:.0f} TFLOPS is {100 * (1 - t / pool_med):.1f}% below the "
                                f"{pool} pool median {pool_med:.0f} (limit {pct}%): slowest rank gates every collective",
                                {"node_tflops": t, "pool_median_tflops": pool_med})
                    v.checks.append(chk)
                    v.cordon("cluster", [chk])
                else:
                    v.checks.append(Check("cluster", "performance_outlier", PASS, "", {"node_tflops": t}))
        alive = [v for v in members if v.status == "accepted"]

        # Homogeneity among survivors (defence in depth: golden pins should already ensure it).
        stacks = {(facts_by_host[v.hostname].driver_version, facts_by_host[v.hostname].runtime_version,
                   facts_by_host[v.hostname].kernel) for v in alive}
        cluster_checks.append(Check("cluster", f"homogeneity[{pool}]", PASS if len(stacks) <= 1 else FAIL,
                                    "" if len(stacks) <= 1 else f"{len(stacks)} distinct stacks among accepted nodes",
                                    {"stacks": sorted(map(list, stacks))}).to_dict())

        need = spec["pool"]["min_accepted_nodes"]
        cap = Check("cluster", f"capacity[{pool}]", PASS if len(alive) >= need else FAIL,
                    "" if len(alive) >= need else f"{len(alive)} accepted nodes < required {need}",
                    {"accepted": len(alive), "required": need, "total": len(members)})
        cluster_checks.append(cap.to_dict())

        floor = spec["allreduce"]["multi_node_min_busbw_gbps"]
        if len(alive) >= 2:
            try:
                res = nccl.parse_nccl_tests(runner.run(pool, [v.hostname for v in alive], spec["gpu"]["count"]))
                bad = []
                if res["wrong"]:
                    bad.append(f"{res['wrong']} wrong results")
                if res["peak_busbw_gbps"] < floor:
                    bad.append(f"busbw {res['peak_busbw_gbps']:.0f} GB/s < {floor}")
                ar = Check("cluster", f"allreduce_multi_node[{pool}]", FAIL if bad else PASS, "; ".join(bad),
                           {"nodes": len(alive), "peak_busbw_gbps": res["peak_busbw_gbps"], "floor": floor})
            except (OSError, RuntimeError, ValueError) as exc:
                ar = Check("cluster", f"allreduce_multi_node[{pool}]", FAIL, f"could not run: {exc}")
        else:
            ar = Check("cluster", f"allreduce_multi_node[{pool}]", FAIL, "fewer than 2 accepted nodes")
        cluster_checks.append(ar.to_dict())
        pools[pool] = {"total": len(members), "accepted": len(alive), "required": need,
                       "gpus_accepted": len(alive) * spec["gpu"]["count"]}

    cordoned = [v for v in verdicts if v.status == "cordoned"]
    if any(c["status"] == FAIL for c in cluster_checks):
        decision = REJECTED
    elif cordoned:
        decision = DEGRADED
    else:
        decision = ACCEPTED
    return {
        "schema": "bringup-report/v1",
        "cluster": manifest["cluster"],
        "decision": decision,
        "exit_code": EXIT_CODES[decision],
        "summary": {
            "nodes": len(verdicts),
            "accepted": len(verdicts) - len(cordoned),
            "cordoned": len(cordoned),
            "cordoned_by_stage": {s: sum(1 for v in cordoned if v.cordon_stage == s) for s in STAGES},
        },
        "pools": pools,
        "rack_findings": racks,
        "cluster_checks": cluster_checks,
        "cordon": [{"hostname": v.hostname, "rack": v.rack, "stage": v.cordon_stage,
                    "checks": v.failed_checks, "reasons": v.reasons} for v in cordoned],
        "nodes": [v.to_dict() for v in verdicts],
    }
