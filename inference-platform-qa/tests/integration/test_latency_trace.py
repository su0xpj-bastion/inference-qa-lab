"""Latency attribution: find a 350ms TTFT gap, prove which hop owns it, prove the fix, gate it.

Scenario: a scheduler change (fixed 350ms batch window, for better batch fill) ships.
At realistic concurrency the batch never fills, so every request waits the full
window. End-to-end TTFT says "slow"; only per-hop spans say *where*.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import openai
import pytest

from inferqa.latency import check_budget, diff, fetch_spans, run_traced, summarize
from inferqa.probe import stream_chat
from mockserve.tracing import new_traceparent
from tests.conftest import _start_lb, spawn_replica

pytestmark = pytest.mark.integration
BUDGET = json.loads(Path(__file__).resolve().parents[2].joinpath("slo/latency_budget.json").read_text())


@pytest.fixture
def stack():
    """Factory: 3 replicas with a scheduler policy behind the proxy. Returns (client, debug_root)."""
    started = []

    def make(policy: str, n: int = 3):
        reps = [spawn_replica(f"{policy}-{i}", MOCKSERVE_SCHEDULER=policy) for i in range(n)]
        lb = _start_lb([f"127.0.0.1:{r.port}" for r in reps], fail_timeout=2.0)
        started.append((lb, reps))
        root = f"http://127.0.0.1:{lb.server_address[1]}"
        return openai.OpenAI(base_url=f"{root}/v1", api_key="t", max_retries=0, timeout=10), root

    yield make
    for lb, reps in started:
        lb.shutdown()
        for r in reps:
            if r.proc.poll() is None:
                r.kill()


def test_trace_context_propagates_through_proxy(stack):
    """One traceparent → spans from both the lb and the serving pod, under the same trace id."""
    client, root = stack("immediate")
    header, tid = new_traceparent()
    stream_chat(client, "hello", extra_headers={"traceparent": header})
    spans = fetch_spans(root, tid)
    names = {s["name"] for s in spans}
    assert {"lb.connect", "lb.ttfb", "lb.total", "server.sched_wait", "server.prefill", "server.ttfb"} <= names
    assert {s["trace_id"] for s in spans} == {tid}
    assert any(s["service"].startswith("mockserve/") for s in spans)


def test_hops_are_non_negative_and_explain_ttft(stack):
    client, root = stack("immediate")
    rows = run_traced(client, root, n=12, concurrency=3)
    for r in rows:
        assert all(v > -2.0 for v in r["hops"].values()), r      # 2ms tolerance for clock reads
        assert sum(r["hops"].values()) == pytest.approx(r["ttft_ms"])


def test_350ms_gap_is_attributed_to_the_scheduler(stack):
    """The headline: TTFT regresses ~350ms; >=90% of it lands on sched_wait, not network or lb."""
    base = summarize(run_traced(*stack("immediate"), n=24, concurrency=4))
    bad = summarize(run_traced(*stack("fixed-wait"), n=24, concurrency=4))
    d = diff(bad, base)
    assert 300 <= d["ttft_p50_delta_ms"] <= 450, d
    assert d["owner"] == "sched_wait" and d["owner_share"] >= 0.9, d
    assert bad["dominant_hop"] == "sched_wait"


def test_budget_gate_fails_regression_and_passes_fix(stack):
    bad = summarize(run_traced(*stack("fixed-wait"), n=16, concurrency=4))
    fixed = summarize(run_traced(*stack("adaptive"), n=16, concurrency=4))
    assert any(v.startswith("sched_wait") for v in check_budget(bad, BUDGET))
    assert check_budget(fixed, BUDGET) == [], fixed
    assert fixed["ttft_ms"]["p50"] < bad["ttft_ms"]["p50"] - 300


def test_adaptive_fix_still_batches_bursts():
    """The fix must not throw away the reason the window existed: a burst still forms full batches."""
    pod = spawn_replica("burst", MOCKSERVE_SCHEDULER="adaptive", MOCKSERVE_MAX_BATCH="8", MOCKSERVE_IDLE_GAP_MS="20")
    try:
        client = openai.OpenAI(base_url=f"{pod.url}/v1", api_key="t", max_retries=0, timeout=10)
        stream_chat(client, "warm", max_tokens=1)
        gate = threading.Barrier(16)

        def go():
            gate.wait()
            stream_chat(client, "burst", max_tokens=4)

        threads = [threading.Thread(target=go) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        metrics = urllib.request.urlopen(f"{pod.url}/metrics", timeout=2).read().decode()
        mean = float(next(l for l in metrics.splitlines() if l.startswith("mockserve_batch_size_mean")).rsplit(" ", 1)[1])
        assert mean > 2.0, metrics
    finally:
        pod.kill()
