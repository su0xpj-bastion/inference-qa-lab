"""latency — decompose end-to-end TTFT into per-hop time and find where a gap lives.

Each request carries a fresh W3C `traceparent`. After the run, spans for
every trace are pulled from the entry point's /debug/spans (the lb merges
in the serving pod's spans), and client TTFT is split into hops:

  client_network = client TTFT        - lb.ttfb          (client stack + network to the lb)
  lb             = lb.ttfb            - server.ttfb      (proxy + lb→pod network)
  sched_wait     = server.sched_wait                     (batch scheduler queueing)
  prefill        = server.prefill                        (dispatch → first chunk)
  server_other   = server.ttfb - sched_wait - prefill    (parsing, admission, everything else)

The hops sum to client TTFT by construction, so "unexplained" time can't hide.
With --baseline, the per-hop p50 delta names the hop that owns a regression.

Usage:
    python -m inferqa.latency --base-url http://127.0.0.1:8080/v1 --requests 40 --concurrency 4 \\
        --budget slo/latency_budget.json --out reports/latency.json [--baseline reports/before.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

from inferqa.metrics import percentile
from inferqa.probe import MODEL, stream_chat
from mockserve.tracing import new_traceparent

ENCODING = "utf-8"
HOPS = ("client_network", "lb", "sched_wait", "prefill", "server_other")
PROMPT = "trace me through every hop"


def decompose(client_ttft_ms: float, spans: list[dict]) -> dict[str, float]:
    """Pure: one request's TTFT split into HOPS (ms). Missing lb spans → lb hop is 0."""
    d = {s["name"]: s["dur_ms"] for s in spans}
    srv = d["server.ttfb"]
    wait = d.get("server.sched_wait", 0.0)
    prefill = d.get("server.prefill", 0.0)
    lb_ttfb = d.get("lb.ttfb", srv)
    return {
        "client_network": client_ttft_ms - lb_ttfb,
        "lb": lb_ttfb - srv,
        "sched_wait": wait,
        "prefill": prefill,
        "server_other": srv - wait - prefill,
    }


def fetch_spans(debug_root: str, trace_id: str) -> list[dict]:
    with urllib.request.urlopen(f"{debug_root}/debug/spans?trace_id={trace_id}", timeout=5) as r:
        return json.loads(r.read())["spans"]


def run_traced(client: openai.OpenAI, debug_root: str, n: int, concurrency: int,
               model: str = MODEL) -> list[dict]:
    """Returns one row per request: client TTFT, served_by, and the hop breakdown."""
    def one(_: int) -> tuple[str, float, str | None]:
        header, trace_id = new_traceparent()
        r = stream_chat(client, PROMPT, model=model, max_tokens=8, extra_headers={"traceparent": header})
        return trace_id, r.timing.ttft * 1000, r.served_by

    # Warm-up (untraced): the first request pays client connection-pool setup,
    # which otherwise shows up as a one-off client_network p99 outlier.
    stream_chat(client, PROMPT, model=model, max_tokens=1)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        done = list(pool.map(one, range(n)))
    rows = []
    for trace_id, ttft_ms, served_by in done:
        spans = fetch_spans(debug_root, trace_id)
        rows.append({"trace_id": trace_id, "ttft_ms": ttft_ms, "served_by": served_by,
                     "hops": decompose(ttft_ms, spans)})
    return rows


def summarize(rows: list[dict]) -> dict:
    def pct(vals: list[float]) -> dict[str, float]:
        return {"p50": round(percentile(vals, 50), 2), "p99": round(percentile(vals, 99), 2)}

    hops = {h: pct([r["hops"][h] for r in rows]) for h in HOPS}
    ttft = pct([r["ttft_ms"] for r in rows])
    dominant = max(HOPS, key=lambda h: hops[h]["p50"])
    return {"requests": len(rows), "ttft_ms": ttft, "hops_ms": hops, "dominant_hop": dominant}


def diff(current: dict, baseline: dict) -> dict:
    """Per-hop p50 delta (ms) and which hop owns the TTFT change."""
    deltas = {h: round(current["hops_ms"][h]["p50"] - baseline["hops_ms"][h]["p50"], 2) for h in HOPS}
    total = current["ttft_ms"]["p50"] - baseline["ttft_ms"]["p50"]
    owner = max(HOPS, key=lambda h: abs(deltas[h]))
    share = deltas[owner] / total if total else 0.0
    return {"ttft_p50_delta_ms": round(total, 2), "hop_p50_delta_ms": deltas,
            "owner": owner, "owner_share": round(share, 3)}


def check_budget(summary: dict, budget: dict) -> list[str]:
    out = []
    for pct, limit in budget.get("ttft_ms", {}).items():
        if summary["ttft_ms"][pct] > limit:
            out.append(f"ttft {pct} {summary['ttft_ms'][pct]:.1f}ms > {limit}ms")
    for hop, limits in budget.get("hops_ms", {}).items():
        for pct, limit in limits.items():
            got = summary["hops_ms"][hop][pct]
            if got > limit:
                out.append(f"{hop} {pct} {got:.1f}ms > {limit}ms")
    return out


def to_markdown(summary: dict, delta: dict | None = None) -> str:
    lines = [f"TTFT p50 {summary['ttft_ms']['p50']:.1f}ms · p99 {summary['ttft_ms']['p99']:.1f}ms · "
             f"dominant hop: **{summary['dominant_hop']}**", "",
             "| Hop | p50 ms | p99 ms |" + (" Δp50 vs baseline |" if delta else ""),
             "|---|---|---|" + ("---|" if delta else "")]
    for h in HOPS:
        row = f"| {h} | {summary['hops_ms'][h]['p50']:.1f} | {summary['hops_ms'][h]['p99']:.1f} |"
        if delta:
            row += f" {delta['hop_p50_delta_ms'][h]:+.1f} |"
        lines.append(row)
    if delta:
        lines += ["", f"TTFT p50 changed {delta['ttft_p50_delta_ms']:+.1f}ms; "
                      f"**{delta['owner']}** accounts for {delta['owner_share']:.0%} of it."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-hop TTFT decomposition via trace spans")
    p.add_argument("--base-url", default=os.environ.get("INFERQA_BASE_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--budget", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    client = openai.OpenAI(base_url=args.base_url, api_key=os.environ.get("CEREBRAS_API_KEY", "mock"), max_retries=0)
    debug_root = args.base_url.rstrip("/").removesuffix("/v1")
    summary = summarize(run_traced(client, debug_root, args.requests, args.concurrency))

    delta = None
    if args.baseline:
        with args.baseline.open("r", encoding=ENCODING) as fh:
            delta = diff(summary, json.load(fh))
        summary["vs_baseline"] = delta
    violations: list[str] = []
    if args.budget:
        with args.budget.open("r", encoding=ENCODING) as fh:
            violations = check_budget(summary, json.load(fh))
    summary["budget_violations"] = violations

    print(to_markdown(summary, delta))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding=ENCODING)
    for v in violations:
        print(f"[BUDGET FAIL] {v}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
