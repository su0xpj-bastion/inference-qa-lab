"""Latency-gap demo: baseline vs regression vs fix, decomposed per hop.

Starts 3 mockserve replicas behind the NGINX-semantics proxy once per scheduler
policy, traces load through each, and writes reports/latency-<policy>.json plus
reports/latency.md (also used as the CI job summary).

Exit 0 only if the story holds: fixed-wait FAILS the per-hop budget on sched_wait,
adaptive PASSES it. If either flips, the gate or the attribution has regressed.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openai  # noqa: E402

from inferqa.latency import check_budget, diff, run_traced, summarize, to_markdown  # noqa: E402
from testbed.lb import build_proxy  # noqa: E402

ENCODING = "utf-8"
REPORTS = ROOT / "reports"
POLICIES = ("immediate", "fixed-wait", "adaptive")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def up(policy: str) -> tuple[list[subprocess.Popen], object, str]:
    procs, peers = [], []
    for i in range(3):
        port = free_port()
        env = {**os.environ, "REPLICA_ID": f"{policy}-{i}", "MOCKSERVE_SCHEDULER": policy, "PYTHONPATH": str(ROOT)}
        procs.append(subprocess.Popen([sys.executable, "-m", "mockserve", "--host", "127.0.0.1", "--port", str(port)],
                                      env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        peers.append(f"127.0.0.1:{port}")
    for peer in peers:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://{peer}/healthz", timeout=0.5)
                break
            except OSError:
                time.sleep(0.05)
    lb = build_proxy("127.0.0.1", 0, peers)
    threading.Thread(target=lb.serve_forever, daemon=True).start()
    return procs, lb, f"http://127.0.0.1:{lb.server_address[1]}"


def main() -> int:
    with (ROOT / "slo/latency_budget.json").open("r", encoding=ENCODING) as fh:
        budget = json.load(fh)
    REPORTS.mkdir(exist_ok=True)
    results: dict[str, dict] = {}
    md = ["## TTFT latency attribution: baseline → regression → fix", ""]
    for policy in POLICIES:
        procs, lb, root = up(policy)
        try:
            client = openai.OpenAI(base_url=f"{root}/v1", api_key="mock", max_retries=0, timeout=10)
            s = summarize(run_traced(client, root, n=40, concurrency=4))
        finally:
            lb.shutdown()
            for p in procs:
                p.kill()
        s["budget_violations"] = check_budget(s, budget)
        if policy != "immediate":
            s["vs_baseline"] = diff(s, results["immediate"])
        results[policy] = s
        (REPORTS / f"latency-{policy}.json").write_text(json.dumps(s, indent=2), encoding=ENCODING)
        verdict = "✅ within budget" if not s["budget_violations"] else "❌ " + "; ".join(s["budget_violations"])
        md += [f"### scheduler = `{policy}` · {verdict}", "", to_markdown(s, s.get("vs_baseline")), ""]
    text = "\n".join(md)
    (REPORTS / "latency.md").write_text(text, encoding=ENCODING)
    print(text)

    ok = (any(v.startswith("sched_wait") for v in results["fixed-wait"]["budget_violations"])
          and not results["adaptive"]["budget_violations"]
          and results["fixed-wait"]["vs_baseline"]["owner"] == "sched_wait")
    print("\nlatency demo:", "story holds" if ok else "STORY BROKEN (gate or attribution regressed)", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
