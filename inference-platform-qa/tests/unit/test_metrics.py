from __future__ import annotations

import pytest

from inferqa.loadgen import build_report, check_slo
from inferqa.metrics import percentile, stream_timing
from inferqa.probe import ProbeResult
from testbed.lb import Upstreams


class TestStreamTiming:
    def test_basic(self):
        t = stream_timing(10.0, [10.2, 10.3, 10.5], 10.6)
        assert t.ttft == pytest.approx(0.2)
        assert t.tbt == pytest.approx(0.15)
        assert t.total == pytest.approx(0.6)
        assert t.tokens == 3
        assert t.output_tps == pytest.approx(2 / 0.4)

    def test_no_tokens_is_well_formed(self):
        t = stream_timing(1.0, [], 2.0)
        assert (t.ttft, t.tbt, t.tokens, t.output_tps) == (0.0, 0.0, 0, 0.0)

    def test_single_token_has_no_gap(self):
        t = stream_timing(0.0, [0.1], 0.1)
        assert t.tbt == 0.0 and t.output_tps == 0.0

    def test_clock_skew_never_negative(self):
        assert stream_timing(5.0, [4.9], 4.8).ttft == 0.0


class TestPercentile:
    @pytest.mark.parametrize("pct,want", [(50, 5), (90, 9), (99, 10), (100, 10), (1, 1)])
    def test_nearest_rank(self, pct, want):
        assert percentile(list(range(1, 11)), pct) == want

    def test_returns_observed_value(self):
        vals = [0.1, 0.9, 0.2]
        assert percentile(vals, 95) in vals

    def test_empty(self):
        assert percentile([], 99) == 0.0

    @pytest.mark.parametrize("bad", [0, -1, 101])
    def test_rejects_bad_pct(self, bad):
        with pytest.raises(ValueError):
            percentile([1.0], bad)


class TestSloGate:
    def _report(self, **over):
        ok = ProbeResult(ok=True, timing=stream_timing(0, [0.1, 0.11, 0.12], 0.12), served_by="a")
        ok2 = ProbeResult(ok=True, timing=stream_timing(0, [0.1, 0.11, 0.12], 0.12), served_by="b")
        report = build_report([ok, ok2] * 50)
        report.update(over)
        return report

    SLO = {"max_error_rate": 0.01, "ttft_s": {"p99": 0.5}, "tbt_s": {"p99": 0.05},
           "min_output_tps_p50": 10, "min_replicas_serving": 2}

    def test_pass(self):
        assert check_slo(self._report(), self.SLO) == []

    def test_error_rate_fails(self):
        assert any("error_rate" in v for v in check_slo(self._report(error_rate=0.05), self.SLO))

    def test_single_replica_fails(self):
        v = check_slo(self._report(served_by={"a": 100}), self.SLO)
        assert any("replica" in x for x in v)

    def test_errors_are_counted(self):
        report = build_report([ProbeResult(ok=False, error="503"), ProbeResult(ok=False, error="503")])
        assert report["error_rate"] == 1.0 and report["errors"] == {"503": 2}


class TestUpstreamSelection:
    def test_round_robin_rotates(self):
        u = Upstreams(["a", "b", "c"], fail_timeout=10)
        assert [u.order()[0] for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]

    def test_failed_peer_ejected(self):
        u = Upstreams(["a", "b"], fail_timeout=10)
        u.mark_failed("a")
        assert all(u.order() == ["b"] for _ in range(4))

    def test_all_down_resets(self):
        # NGINX behaviour: when every peer is marked down, try them all again.
        u = Upstreams(["a", "b"], fail_timeout=10)
        u.mark_failed("a")
        u.mark_failed("b")
        assert sorted(u.order()) == ["a", "b"]

    def test_peer_returns_after_fail_timeout(self):
        u = Upstreams(["a", "b"], fail_timeout=0)
        u.mark_failed("a")
        assert "a" in u.order() + u.order()
