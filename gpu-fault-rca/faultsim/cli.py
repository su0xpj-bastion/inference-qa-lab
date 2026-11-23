"""chaos list | run <scenario> | matrix | recover <scenario>"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from faultsim.matrix import DEFAULT_SEEDS, evaluate, markdown, write
from faultsim.recovery import recovery_scenarios, validate
from faultsim.scenarios import SCENARIOS, run

PRECISION_GATE = 0.95
RECALL_GATE = 0.95


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chaos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run", help="inject a scenario and write an incident bundle")
    r.add_argument("scenario", choices=sorted(SCENARIOS))
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--load", choices=("idle", "steady", "ramp", "soak"), default=None)
    m = sub.add_parser("matrix", help="precision/recall of the RCA over the scenario library")
    m.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    m.add_argument("--out", type=Path, default=Path("reports"))
    v = sub.add_parser("recover", help="replay with the RCA's recommended action and check recovery SLOs")
    v.add_argument("scenario", nargs="?", choices=sorted(SCENARIOS))
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for name, sc in SCENARIOS.items():
            print(f"{name:30s} load={sc.load:6s} {','.join(sc.tags):18s} {sc.description}")
        return 0
    if args.cmd == "run":
        out = args.out or Path("bundles") / f"{args.scenario}-s{args.seed}"
        run(args.scenario, out, seed=args.seed, load=args.load)
        print(out)
        return 0
    if args.cmd == "matrix":
        with tempfile.TemporaryDirectory() as tmp:
            res = evaluate(Path(tmp), tuple(args.seeds))
        write(res, args.out)
        sys.stdout.write(markdown(res))
        return 0 if res["precision"] >= PRECISION_GATE and res["recall"] >= RECALL_GATE else 1
    names = [args.scenario] if args.scenario else recovery_scenarios()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            rows.append(validate(name, Path(tmp), seed=args.seed))
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ["## Recovery-path validation", "",
             "| Scenario | Action(s) | TTD vs impact | TTM | Requests lost | Verdict |", "|---|---|---|---|---|---|"]
    for x in rows:
        acts = ", ".join(f"{a['action']}@T+{a['at_s']}" for a in x.get("actions", []))
        ttd = "-" if x.get("ttd_s") is None else f"{x['ttd_s']}s"
        ttm = "not mitigated" if x.get("ttm_s") is None else f"{x['ttm_s']}s"
        lines.append(f"| `{x['scenario']}` | {acts} | {ttd} | {ttm} | "
                     f"{x.get('requests_lost')} | {x['verdict']} |")
    md = "\n".join(lines) + "\n"
    with (args.out / "recovery.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    with (args.out / "recovery.md").open("w", encoding="utf-8") as fh:
        fh.write(md)
    sys.stdout.write(md)
    return 0
