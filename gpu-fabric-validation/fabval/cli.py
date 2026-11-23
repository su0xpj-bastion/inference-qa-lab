from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from fabval import nccl_log, nccl_tests, report
from fabval.analyze import EXIT_INCONCLUSIVE, analyze
from fabval.model import JobShape, NodeType, expected_busbw, load_topologies
from fabval.sim import Fault, SimCluster


def cmd_simulate(args: argparse.Namespace) -> int:
    nt = NodeType.load(args.node_type)
    faults = [Fault.parse(s) for s in args.fault]
    out = SimCluster(nt, args.nodes, faults, seed=args.seed).write(Path(args.out))
    print(f"wrote simulated run for {args.nodes} x {nt.name} to {out} (faults: {', '.join(map(str, faults)) or 'none'})")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "cluster.json").exists():
        print(f"no cluster.json in {run_dir}", file=sys.stderr)
        return EXIT_INCONCLUSIVE
    rep = analyze(run_dir)
    report.write(rep, Path(args.md) if args.md else None, Path(args.json) if args.json else None)
    if not args.quiet:
        print(report.to_markdown(rep))
    return rep.exit_code


def cmd_expected(args: argparse.Namespace) -> int:
    nt = NodeType.load(args.node_type)
    shape = JobShape(args.collective, args.nodes, args.ranks_per_node or nt.gpus_per_node, args.algo)
    print(f"{expected_busbw(nt, shape):.1f}")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if args.kind == "nccl-tests":
        res = nccl_tests.parse_file(path, args.collective)
        out = {"collective": res.collective, "nranks": res.nranks, "hosts": res.hosts, "header": res.header,
               "avg_busbw": res.avg_busbw, "wrong": res.total_wrong, "complete": res.complete,
               "consistency_problems": res.consistency_problems(),
               "rows": [asdict(r) for r in res.rows]}
    else:
        out = {h: s.to_dict() for h, s in nccl_log.parse_file(path).items()}
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fabval", description="GPU fabric validation: nccl-tests, NCCL logs, IB/RoCE health")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write a simulated run directory with injected faults")
    s.add_argument("--node-type", default="h100-sxm-8x-400g-roce", choices=sorted(load_topologies()))
    s.add_argument("--nodes", type=int, default=8)
    s.add_argument("--fault", action="append", default=[], help="kind:node[:rail], repeatable")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_simulate)

    a = sub.add_parser("analyze", help="analyze a run directory; exit 0 pass, 1 fail, 2 inconclusive")
    a.add_argument("run_dir")
    a.add_argument("--md")
    a.add_argument("--json")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(fn=cmd_analyze)

    e = sub.add_parser("expected", help="expected large-message busbw (GB/s) for a node type and job shape")
    e.add_argument("--node-type", required=True, choices=sorted(load_topologies()))
    e.add_argument("--collective", default="all_reduce")
    e.add_argument("--nodes", type=int, default=2)
    e.add_argument("--ranks-per-node", type=int)
    e.add_argument("--algo", default="ring", choices=["ring", "tree"])
    e.set_defaults(fn=cmd_expected)

    pa = sub.add_parser("parse", help="parse one file to JSON")
    pa.add_argument("kind", choices=["nccl-tests", "nccl-log"])
    pa.add_argument("file")
    pa.add_argument("--collective")
    pa.set_defaults(fn=cmd_parse)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
