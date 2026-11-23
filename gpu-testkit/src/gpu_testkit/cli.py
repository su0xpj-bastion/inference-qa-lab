"""``gpu-testkit``: run diffs, hardware inventory, quarantine audit, pool holders."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from gpu_testkit.backends import BACKEND_CHOICES, select_backend
from gpu_testkit.pool import DevicePool, default_lock_dir
from gpu_testkit.quarantine import QuarantineError, load, today
from gpu_testkit.report import (
    DEFAULT_SLOW_MIN_DELTA_S,
    DEFAULT_SLOW_RATIO,
    DEFAULT_TOP,
    diff_runs,
    render_markdown,
)
from gpu_testkit.results import read_run

ENCODING = "utf-8"
EXIT_OK = 0
EXIT_GATE = 1
EXIT_USAGE = 2


def _cmd_report(args: argparse.Namespace) -> int:
    head = read_run(args.head)
    base = read_run(args.base) if args.base else None
    diff = diff_runs(head, base, slow_ratio=args.slow_ratio, slow_min_delta_s=args.slow_min_delta)
    text = render_markdown(diff, top=args.top)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding=ENCODING) as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    if args.fail_on_new and diff.new_failures:
        print(f"gpu-testkit: {len(diff.new_failures)} new failure(s) vs base", file=sys.stderr)
        return EXIT_GATE
    if args.fail_on_gating and diff.gating:
        print(f"gpu-testkit: {len(diff.gating)} gating failure(s)", file=sys.stderr)
        return EXIT_GATE
    return EXIT_OK


def _cmd_hw(args: argparse.Namespace) -> int:
    host = select_backend(args.backend, args.fake_spec).discover()
    if args.json:
        print(json.dumps(asdict(host), indent=2))
    else:
        print(host.summary())
        for gpu in host.gpus:
            print(f"  [{gpu.index}] {gpu.name}  {gpu.arch_label}  {gpu.mem_gb:g} GB  {gpu.uuid}")
    return EXIT_OK


def _cmd_quarantine(args: argparse.Namespace) -> int:
    try:
        entries = load(args.file)
    except QuarantineError as exc:
        print(f"invalid quarantine file: {exc}", file=sys.stderr)
        return EXIT_USAGE
    now = args.today or today()
    expired = 0
    for entry in sorted(entries, key=lambda e: e.expires):
        state = "EXPIRED" if entry.expired(now) else "active"
        expired += entry.expired(now)
        ticket = f" {entry.ticket}" if entry.ticket else ""
        print(f"{state:8} {entry.expires}  {entry.owner:<16}{ticket}  {entry.test}")
    print(f"{len(entries)} entries, {expired} expired (today={now})")
    return EXIT_GATE if (args.fail_on_expired and expired) else EXIT_OK


def _cmd_pool(args: argparse.Namespace) -> int:
    host = select_backend(args.backend, args.fake_spec).discover()
    lock_dir = args.lock_dir or default_lock_dir(host.hostname)
    holders = DevicePool(lock_dir).holders([gpu.index for gpu in host.gpus])
    print(f"lock dir: {lock_dir}")
    for gpu in host.gpus:
        print(f"  [{gpu.index}] {gpu.name:<28} {holders.get(gpu.index, 'free')}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-testkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="summarize a run, or diff it against a base run, as markdown")
    report.add_argument("head", type=Path, help="results JSONL of the run under review")
    report.add_argument("--base", type=Path, help="results JSONL to compare against")
    report.add_argument("--output", type=Path, help="append markdown here (e.g. $GITHUB_STEP_SUMMARY)")
    report.add_argument("--top", type=int, default=DEFAULT_TOP)
    report.add_argument("--slow-ratio", type=float, default=DEFAULT_SLOW_RATIO)
    report.add_argument("--slow-min-delta", type=float, default=DEFAULT_SLOW_MIN_DELTA_S)
    report.add_argument("--fail-on-new", action="store_true", help="exit 1 if head has failures base did not")
    report.add_argument("--fail-on-gating", action="store_true", help="exit 1 on any gating failure")
    report.set_defaults(func=_cmd_report)

    for name, func, text in (
        ("hw", _cmd_hw, "print the hardware a backend discovers"),
        ("pool", _cmd_pool, "show which process holds which GPU lease"),
    ):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("--backend", choices=BACKEND_CHOICES, default="auto")
        cmd.add_argument("--fake-spec", help="preset (h100x8), inline JSON, or JSON path")
        cmd.set_defaults(func=func)
        if name == "hw":
            cmd.add_argument("--json", action="store_true")
        else:
            cmd.add_argument("--lock-dir", type=Path)

    quarantine = sub.add_parser("quarantine", help="audit a quarantine file: owners, expiry")
    quarantine.add_argument("file", type=Path)
    quarantine.add_argument("--today", help="YYYY-MM-DD (default: today)")
    quarantine.add_argument("--fail-on-expired", action="store_true")
    quarantine.set_defaults(func=_cmd_quarantine)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
