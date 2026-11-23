"""loadgen — concurrent streaming load + SLO gate for an inference endpoint.

Runs N streamed completions at concurrency C, aggregates TTFT / TBT /
throughput percentiles, error rate, and replica distribution, then
checks them against an SLO file. Exit code is the gate: 0 pass, 1 fail.

Usage:
    python -m inferqa.loadgen --base-url http://127.0.0.1:8080/v1 \\
        --requests 200 --concurrency 16 --slo slo/default.json --out report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

from inferqa.metrics import summarize
from inferqa.probe import MODEL, ProbeResult, stream_chat

ENCODING = "utf-8"
PROMPT = "Explain wafer scale inference in one short paragraph please"


def run_load(client: openai.OpenAI, n: int, concurrency: int, model: str = MODEL) -> list[ProbeResult]:
    def one(_: int) -> ProbeResult:
        try:
            return stream_chat(client, PROMPT, model=model)
        except openai.RateLimitError:
            return ProbeResult(ok=False, error="429")
        except openai.APIStatusError as exc:
            return ProbeResult(ok=False, error=str(exc.status_code))
        except openai.APIError as exc:
            return ProbeResult(ok=False, error=type(exc).__name__)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, range(n)))


def build_report(results: list[ProbeResult]) -> dict:
    ok = [r for r in results if r.ok and r.timing]
    errors = Counter(r.error or "contract" for r in results if not r.ok)
    return {
        "requests": len(results),
        "succeeded": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 4) if results else 0.0,
        "errors": dict(errors),
        "ttft_s": summarize([r.timing.ttft for r in ok]),
        "tbt_s": summarize([r.timing.tbt for r in ok]),
        "output_tps": summarize([r.timing.output_tps for r in ok]),
        "retried": sum(1 for r in results if r.attempts > 1),
        "served_by": dict(Counter(r.served_by or "unknown" for r in ok)),
        "warnings": sorted({w for r in results for w in r.warnings}),
    }


def check_slo(report: dict, slo: dict) -> list[str]:
    """Return human-readable violations; empty list means the gate passes."""
    violations = []
    if report["error_rate"] > slo["max_error_rate"]:
        violations.append(f"error_rate {report['error_rate']} > {slo['max_error_rate']}")
    for metric in ("ttft_s", "tbt_s"):
        for pct, limit in slo.get(metric, {}).items():
            got = report[metric][pct]
            if got > limit:
                violations.append(f"{metric}.{pct} {got:.4f} > {limit}")
    floor = slo.get("min_output_tps_p50")
    if floor is not None and report["output_tps"]["p50"] < floor:
        violations.append(f"output_tps.p50 {report['output_tps']['p50']:.1f} < {floor}")
    min_replicas = slo.get("min_replicas_serving")
    if min_replicas is not None and len(report["served_by"]) < min_replicas:
        violations.append(f"only {len(report['served_by'])} replica(s) served traffic, want >= {min_replicas}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Concurrent streaming load + SLO gate")
    parser.add_argument("--base-url", default=os.environ.get("INFERQA_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--slo", type=Path, default=Path("slo/default.json"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    client = openai.OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get("CEREBRAS_API_KEY", "not-needed-for-mock"),
        max_retries=0,
    )
    report = build_report(run_load(client, args.requests, args.concurrency, args.model))
    with args.slo.open("r", encoding=ENCODING) as fh:
        slo = json.load(fh)
    report["slo_violations"] = check_slo(report, slo)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding=ENCODING)
    for v in report["slo_violations"]:
        print(f"[SLO FAIL] {v}", file=sys.stderr)
    return 1 if report["slo_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
