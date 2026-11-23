"""CLI: run the Inference / Accelerator / Cluster / Release scenarios against the live Cerebras API or the local mock.

  python -m ies run                      # all workflows, live (needs CEREBRAS_API_KEY)
  python -m ies run --target mock        # all workflows against an in-process mock
  python -m ies run --target mock --faults slow_decode -w accelerator
  python -m ies chat "Why is fast inference important?"   # streamed chat + usage + cost
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cerebras.cloud.sdk as cerebras
from dotenv import load_dotenv

from .client import MAX_ATTEMPTS, Budget, chat, make_client
from .core import Check, Ctx, verdict
from .workflows import DEFAULT, WORKFLOWS

MODEL = "qwen-3.8-27b"
SLO_FILE = Path(__file__).resolve().parent.parent / "slo.json"
# USD per 1K tokens, from cerebras.ai/pricing (Developer tier, 2026-11-14)
RATES = {
    "qwen-3.8-27b": {"prompt": 0.00099, "completion": 0.00149},
    "gpt-oss-120b": {"prompt": 0.00035, "completion": 0.00075},
}
# models that accept reasoning_effort="none"; others keep their default reasoning
NO_REASONING = {"qwen-3.8-27b", "gemma-4-31b"}


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    if model not in RATES:
        return None
    return (prompt_tokens / 1000) * RATES[model]["prompt"] + \
        (completion_tokens / 1000) * RATES[model]["completion"]


def load_slo(target: str) -> dict:
    with SLO_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)[target]


def to_markdown(checks: list[Check], header: str) -> str:
    icon = {"pass": "✅", "fail": "❌", "warn": "⚠️", "info": "ℹ️", "skip": "⏭️"}
    lines = [f"## {header}", "", f"**Verdict: {verdict(checks)}**", ""]
    for wf in dict.fromkeys(c.workflow for c in checks):
        group = [c for c in checks if c.workflow == wf]
        lines += [f"### {wf} — {verdict(group)}", "", "| Check | Result | Gated | Detail |", "|---|---|---|---|"]
        for c in group:
            detail = c.detail.replace("|", "\\|")
            lines.append(f"| {c.name} | {icon.get(c.status, c.status)} {c.status} | "
                         f"{'yes' if c.gated else 'no'} | {detail} |")
        lines.append("")
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace) -> int:
    srv = None
    if args.target == "mock":
        from mockserve.server import API_KEY, serve
        srv, _ = serve(0, {f for f in args.faults.split(",") if f})
        api_key, base_url = API_KEY, f"http://127.0.0.1:{srv.server_address[1]}"
    else:
        api_key, base_url = os.environ.get("CEREBRAS_API_KEY"), os.environ.get("CEREBRAS_BASE_URL")
        if not api_key:
            print("[ERROR] CEREBRAS_API_KEY is not set (add it to .env or export it)", file=sys.stderr)
            return 2
    slo = load_slo(args.target)
    budget = Budget(args.max_requests or slo["max_requests"])
    ctx = Ctx(client=make_client(api_key, base_url), model=args.model, target=args.target, slo=slo,
              budget=budget, bad_key_client=lambda: make_client("ies-invalid-key", base_url),
              base_params={"reasoning_effort": "none"} if args.model in NO_REASONING else {},
              baseline_dir=args.baselines, update_baseline=args.update_baseline,
              api_key=api_key, base_url=base_url)
    names: list[str] = []
    for part in args.workflow.split(","):
        for name in (DEFAULT if part == "all" else [part]):
            if name not in WORKFLOWS:
                print(f"[ERROR] unknown scenario {name!r}; choose from {', '.join(WORKFLOWS)}", file=sys.stderr)
                return 2
            if name not in names:
                names.append(name)
    if "schema" in names:                       # validate everything the others produced
        names = [n for n in names if n != "schema"] + ["schema"]
    checks: list[Check] = []
    try:
        for name in names:
            print(f"[RUN] {name} ...", file=sys.stderr, flush=True)
            got = WORKFLOWS[name](ctx)
            for c in got:
                print(f"  {c.status:5} {c.name}: {c.detail}", file=sys.stderr, flush=True)
            checks += got
    finally:
        if srv is not None:
            srv.shutdown()
    header = f"{args.target} · {args.model}" + (f" · faults={args.faults}" if args.faults else "")
    md = to_markdown(checks, header) + f"\n_{budget.used} API requests this run._\n"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(
        {"target": args.target, "model": args.model, "faults": args.faults, "requests": budget.used,
         "verdict": verdict(checks), "checks": [c.__dict__ for c in checks]}, indent=2, default=str))
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    print(md)
    return 1 if verdict(checks) == "FAIL" else 0


def cmd_chat(args: argparse.Namespace) -> int:
    """The quickstart, done properly: stream, retry 429s, report usage, time_info, and cost."""
    if not os.environ.get("CEREBRAS_API_KEY"):
        print("[ERROR] CEREBRAS_API_KEY is not set (add it to .env or export it)", file=sys.stderr)
        return 2
    client = make_client(os.environ.get("CEREBRAS_API_KEY"), os.environ.get("CEREBRAS_BASE_URL"))
    params = {"reasoning_effort": "none"} if args.model in NO_REASONING else {}
    try:
        call = chat(client, args.model, [{"role": "user", "content": args.prompt}], stream=True, **params)
    except cerebras.RateLimitError:
        print(f"[ERROR] rate limit exceeded after {MAX_ATTEMPTS} attempts", file=sys.stderr)
        return 1
    except cerebras.APIStatusError as exc:
        print(f"[ERROR] status {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except cerebras.APIError as exc:
        print(f"[ERROR] API error: {exc}", file=sys.stderr)
        return 1
    print(call.content, flush=True)
    if call.finish_reason and call.finish_reason != "stop":
        print(f"[WARN] stream ended with finish_reason={call.finish_reason!r}")
    if call.usage_seen:
        usd = cost(args.model, call.prompt_tokens or 0, call.completion_tokens or 0)
        print(f"[USAGE] prompt={call.prompt_tokens} completion={call.completion_tokens} "
              f"cached={call.cached_tokens} cost=" + (f"${usd:.6f}" if usd is not None else "n/a (no rate)"))
    if call.server_total is not None:
        tps = (call.completion_tokens or 0) / call.completion_time if call.completion_time else 0
        print(f"[TIME] ttft={(call.ttft or 0) * 1e3:.0f}ms wall={call.wall * 1e3:.0f}ms | server queue="
              f"{(call.queue_time or 0) * 1e3:.1f}ms prefill={(call.prompt_time or 0) * 1e3:.1f}ms "
              f"decode={(call.completion_time or 0) * 1e3:.1f}ms ({tps:,.0f} tok/s) fp={call.fingerprint}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="ies")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run scenarios")
    r.add_argument("--target", choices=("live", "mock"), default="live")
    r.add_argument("-w", "--workflow", default="all", help=f"all or comma list of {','.join(WORKFLOWS)}")
    r.add_argument("--model", default=os.environ.get("CEREBRAS_MODEL", MODEL))
    r.add_argument("--faults", default="", help="mock only: comma list of injected faults")
    r.add_argument("--out", default="reports")
    r.add_argument("--baselines", default="baselines")
    r.add_argument("--update-baseline", action="store_true")
    r.add_argument("--max-requests", type=int, default=0, help="request budget (0 = from slo.json)")
    c = sub.add_parser("chat", help="one streamed chat with usage, timing and cost")
    c.add_argument("prompt")
    c.add_argument("--model", default=os.environ.get("CEREBRAS_MODEL", MODEL))
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_chat(args)
