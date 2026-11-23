from __future__ import annotations

# CLI.
#
#   python -m bringup gen-fleet --scenario defects --out fleet/
#   python -m bringup run --fleet fleet/ --manifest manifests/golden.json --out reports/
#   python -m bringup node --captures /var/lib/bringup/captures --manifest golden.json
#
# `run` exits 0 ACCEPTED, 1 REJECTED, 3 ACCEPTED_DEGRADED (2 is argparse's usage
# error). `node` is the per-node hook used by provisioning: exit 0 if the node
# passes node + burn-in + domain stages, 1 otherwise.

import argparse
import json
import sys
from pathlib import Path

from bringup.allreduce import FixtureRunner, NcclTestsRunner
from bringup.collect import collect_fleet, collect_node
from bringup.fleetgen import write_fleet
from bringup.gates import BURNIN_GATES, DOMAIN_GATES, NODE_GATES
from bringup.model import FAIL
from bringup.pipeline import load_manifest, run_pipeline
from bringup.report import write_reports


def _cmd_gen(a: argparse.Namespace) -> int:
    nodes = write_fleet(Path(a.out), a.scenario, a.seed)
    injected = sum(1 for n in nodes if n.defect)
    print(f"wrote {len(nodes)} node capture dirs to {a.out} ({injected} with injected defects)")
    return 0


def _cmd_run(a: argparse.Namespace) -> int:
    fleet_dir = Path(a.fleet)
    manifest = load_manifest(Path(a.manifest))
    runner = FixtureRunner(fleet_dir) if a.runner == "fixture" else NcclTestsRunner()
    report = run_pipeline(collect_fleet(fleet_dir), manifest, runner)
    jpath, mpath = write_reports(report, Path(a.out))
    s = report["summary"]
    print(f"{report['decision']}: {s['accepted']}/{s['nodes']} accepted, {s['cordoned']} cordoned "
          f"-> {mpath} {jpath}")
    for c in report["cordon"]:
        print(f"  CORDON {c['hostname']} [{c['stage']}] {c['reasons'][0]}")
    return report["exit_code"]


def _cmd_node(a: argparse.Namespace) -> int:
    manifest = load_manifest(Path(a.manifest))
    facts = collect_node(Path(a.captures))
    spec = manifest["node_types"].get(facts.node_type)
    if spec is None:
        print(f"FAIL node type {facts.node_type!r} not in manifest")
        return 1
    th = manifest["thresholds"]
    failed = []
    for stage, gates in (("node", NODE_GATES), ("burnin", BURNIN_GATES), ("domain", DOMAIN_GATES)):
        if a.stage and stage not in a.stage:
            continue
        for gate in gates:
            chk = gate(facts, spec, th)
            print(f"{chk.status.upper():4} {stage}/{chk.name} {chk.reason}")
            if chk.status == FAIL:
                failed.append(chk.to_dict())
        if failed:
            break
    if a.json:
        with Path(a.json).open("w", encoding="utf-8") as fh:
            json.dump({"hostname": facts.hostname, "passed": not failed, "failed": failed}, fh, indent=2)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bringup", description="GPU cluster bring-up acceptance")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-fleet", help="write a simulated fleet of node captures")
    g.add_argument("--out", required=True)
    g.add_argument("--scenario", default="defects", choices=["healthy", "defects", "leaf-down"])
    g.add_argument("--seed", type=int, default=20261114)
    g.set_defaults(fn=_cmd_gen)

    r = sub.add_parser("run", help="run all stages over a fleet of captures")
    r.add_argument("--fleet", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--runner", default="fixture", choices=["fixture", "nccl"])
    r.set_defaults(fn=_cmd_run)

    n = sub.add_parser("node", help="per-node acceptance hook (provisioning)")
    n.add_argument("--captures", required=True)
    n.add_argument("--manifest", required=True)
    n.add_argument("--stage", action="append", choices=["node", "burnin", "domain"],
                   help="limit to these stages (repeatable); default all three")
    n.add_argument("--json", help="write a machine-readable verdict here")
    n.set_defaults(fn=_cmd_node)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
