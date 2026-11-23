from __future__ import annotations

import threading
import time

import pytest

from inferqa.latency import HOPS, check_budget, decompose, diff
from mockserve.sched import BatchScheduler
from mockserve.tracing import new_traceparent, trace_id_of

SPANS = [
    {"name": "lb.ttfb", "dur_ms": 380.0},
    {"name": "server.ttfb", "dur_ms": 372.0},
    {"name": "server.sched_wait", "dur_ms": 350.0},
    {"name": "server.prefill", "dur_ms": 20.0},
]


def summary(ttft_p50: float, **hop_p50: float) -> dict:
    hops = {h: {"p50": hop_p50.get(h, 0.0), "p99": hop_p50.get(h, 0.0)} for h in HOPS}
    return {"ttft_ms": {"p50": ttft_p50, "p99": ttft_p50}, "hops_ms": hops}


class TestDecompose:
    def test_hops_sum_to_client_ttft(self):
        hops = decompose(390.0, SPANS)
        assert sum(hops.values()) == pytest.approx(390.0)
        assert hops == pytest.approx({"client_network": 10.0, "lb": 8.0, "sched_wait": 350.0,
                                      "prefill": 20.0, "server_other": 2.0})

    def test_without_lb_spans_lb_hop_is_zero(self):
        hops = decompose(380.0, [s for s in SPANS if not s["name"].startswith("lb.")])
        assert hops["lb"] == 0.0 and hops["client_network"] == pytest.approx(8.0)


class TestDiffAndBudget:
    def test_owner_is_the_hop_that_moved(self):
        d = diff(summary(375, sched_wait=355, prefill=12), summary(15, prefill=12))
        assert d["owner"] == "sched_wait" and d["owner_share"] == pytest.approx(355 / 360, abs=1e-3)

    def test_budget_names_the_hop(self):
        v = check_budget(summary(375, sched_wait=355), {"ttft_ms": {"p99": 200}, "hops_ms": {"sched_wait": {"p99": 50}}})
        assert any(x.startswith("sched_wait p99") for x in v) and any(x.startswith("ttft p99") for x in v)


class TestTraceContext:
    def test_roundtrip(self):
        header, tid = new_traceparent()
        assert trace_id_of(header) == tid and len(tid) == 32

    @pytest.mark.parametrize("bad", [None, "", "garbage", "01-" + "a" * 32 + "-" + "b" * 16 + "-01"])
    def test_invalid_headers_are_ignored(self, bad):
        assert trace_id_of(bad) is None


class TestScheduler:
    def _waits(self, sched: BatchScheduler, n: int) -> list[float]:
        out: list[float] = []
        lock = threading.Lock()

        def go():
            w = sched.admit()
            with lock:
                out.append(w)

        threads = [threading.Thread(target=go) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        return out

    def test_fixed_wait_holds_a_lone_request_for_the_whole_window(self):
        (w,) = self._waits(BatchScheduler("fixed-wait", max_batch=8, batch_wait_s=0.2), 1)
        assert 0.18 <= w < 0.35

    def test_adaptive_releases_a_lone_request_after_the_idle_gap(self):
        (w,) = self._waits(BatchScheduler("adaptive", max_batch=8, batch_wait_s=0.2, idle_gap_s=0.005), 1)
        assert w < 0.05

    def test_full_batch_dispatches_without_waiting(self):
        s = BatchScheduler("fixed-wait", max_batch=4, batch_wait_s=2.0)
        t0 = time.monotonic()
        self._waits(s, 4)
        assert time.monotonic() - t0 < 1.0 and s.batches == [4]

    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError):
            BatchScheduler("fifo")
