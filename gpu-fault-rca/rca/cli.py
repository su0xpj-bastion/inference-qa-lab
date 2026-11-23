"""rca analyze <bundle-dir> [--out DIR] [--threshold 0.7]

Exit 0 = healthy, 1 = incident with a confident root cause, 2 = degraded but unexplained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rca.engine import THRESHOLD, analyze
from rca.report import markdown

ENCODING = "utf-8"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rca")
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze", help="analyze an incident bundle")
    an.add_argument("bundle", type=Path)
    an.add_argument("--out", type=Path, default=None, help="write incident.md + incident.json here")
    an.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args(argv)
    report = analyze(args.bundle, args.threshold)
    md = markdown(report)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "incident.json").open("w", encoding=ENCODING) as fh:
            json.dump(report, fh, indent=2)
        with (args.out / "incident.md").open("w", encoding=ENCODING) as fh:
            fh.write(md)
    sys.stdout.write(md)
    return {"healthy": 0, "incident": 1, "degraded": 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
