from __future__ import annotations

import pytest

from fabval import nccl_tests


def test_healthy_all_reduce(fixture_text):
    """Header, devices, rows, avg line and the busbw/algbw identity are all read."""
    r = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.2node.healthy.txt"), "all_reduce")
    assert r.complete and r.nranks == 16 and r.hosts == ["gpu-a1", "gpu-a2"]
    assert r.header["iters"] == 20 and r.header["warmup"] == 5
    assert len(r.rows) == 31 and r.rows[0].size == 8 and r.rows[-1].size == 8 << 30
    last = r.rows[-1]
    assert (last.oop.time_us, last.oop.algbw, last.oop.busbw, last.oop.wrong) == (44943.0, 191.13, 358.37, 0)
    assert r.avg_busbw == pytest.approx(128.0726)
    assert r.oob_ok and r.total_wrong == 0
    assert r.consistency_problems() == []
    assert r.peak_busbw(256 << 20) == pytest.approx(366.06)


@pytest.mark.parametrize("name,coll", [
    ("all_gather_perf.2node.healthy.txt", "all_gather"),
    ("alltoall_perf.2node.healthy.txt", "alltoall"),
    ("sendrecv_perf.pair.healthy.txt", "sendrecv"),
])
def test_other_collectives_parse_consistently(fixture_text, name, coll):
    """Each collective's busbw factor is applied correctly by the identity check."""
    r = nccl_tests.parse(fixture_text(f"nccl_tests/{name}"), nccl_tests.infer_collective(name))
    assert r.collective == coll and r.complete and r.consistency_problems() == []
    assert r.rows[0].redop == ("sum" if coll == "sendrecv" else "none")


def test_degraded_vs_healthy_peak(fixture_text):
    h = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.2node.healthy.txt"), "all_reduce")
    d = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.2node.degraded.txt"), "all_reduce")
    assert d.peak_busbw(256 << 20) < 0.55 * h.peak_busbw(256 << 20)


def test_wrong_results_counted(fixture_text):
    """#wrong > 0 is surfaced per row and the OOB line says FAILED."""
    r = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.2node.wrong.txt"), "all_reduce")
    assert r.total_wrong > 0 and r.oob_ok is False
    assert {row.size for row in r.rows if row.wrong} == {1 << 28, 1 << 32}


def test_legacy_12_column_format(fixture_text):
    """Older nccl-tests: no root column, float error column, no Group in device lines."""
    r = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.legacy_12col.txt"), "all_reduce")
    assert r.nranks == 8 and len(r.rows) == 5 and r.rows[0].root == -1
    assert r.rows[-1].oop.busbw == 230.64 and r.total_wrong == 0
    assert r.consistency_problems() == []


def test_validation_off_reads_na(fixture_text):
    r = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.nocheck.txt"), "all_reduce")
    assert all(row.oop.wrong is None and row.ip.wrong is None for row in r.rows)
    assert r.consistency_problems() == []


def test_truncated_run_is_incomplete(fixture_text):
    """A run that died mid-sweep has no Avg line; the inline NCCL failure line is skipped."""
    r = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.truncated.txt"), "all_reduce")
    assert not r.complete and len(r.rows) == 3


def test_busbw_mismatch_flagged(fixture_text):
    """busbw that does not equal algbw x factor means the wrong collective or rank count."""
    r = nccl_tests.parse(fixture_text("nccl_tests/all_reduce_perf.busbw_mismatch.txt"), "all_reduce")
    assert any("busbw" in p for p in r.consistency_problems())


def test_bad_row_raises():
    with pytest.raises(nccl_tests.ParseError):
        nccl_tests.parse("     8   2  float  sum  -1  1.0  0.0  0.0\n")


def test_infer_collective():
    assert nccl_tests.infer_collective("runs/x/all_reduce_perf.txt") == "all_reduce"
    assert nccl_tests.infer_collective("reduce_scatter_perf.txt") == "reduce_scatter"
    assert nccl_tests.infer_collective("misc.txt") is None
