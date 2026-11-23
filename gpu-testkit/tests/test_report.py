"""Run diffs and the gpu-testkit CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_testkit.cli import main
from gpu_testkit.report import diff_runs, render_markdown
from gpu_testkit.results import RunRecord, TestRecord, read_run, write_run


def run(run_id: str, *records: TestRecord) -> RunRecord:
    return RunRecord(run_id=run_id, host={"summary": "fake-h100-node0: 8x H100"}, tests={r.nodeid: r for r in records})


BASE = run(
    "base",
    TestRecord("t::regressed", "passed", "passed", duration_s=1.0),
    TestRecord("t::fixed", "failed", "product", attempts=3),
    TestRecord("t::still", "failed", "product"),
    TestRecord("t::slow", "passed", "passed", duration_s=1.0),
    TestRecord("t::lost", "passed", "passed"),
    TestRecord("t::gone", "passed", "passed"),
)
HEAD = run(
    "head",
    TestRecord("t::regressed", "failed", "product", attempts=3, message="AssertionError: logits", diagnostics="d.tgz"),
    TestRecord("t::fixed", "passed", "passed"),
    TestRecord("t::still", "failed", "infra", tags={"evidence": "Xid 79"}),
    TestRecord("t::slow", "passed", "passed", duration_s=2.5),
    TestRecord("t::lost", "skipped", "skipped", message="gpu-testkit: needs 8 GPU(s)"),
    TestRecord("t::flaky", "passed", "flaky", attempts=2, tags={"evidence": "failed 1 of 2 attempts"}),
    TestRecord("t::hidden", "quarantined", "product", quarantine="quarantined GPU-1 (@o, until 2026-12-01): r"),
    TestRecord("t::qpass", "passed", "passed", quarantine="quarantined GPU-2 (@o, until 2026-12-01): r"),
)


def test_diff_buckets_every_kind_of_change() -> None:
    diff = diff_runs(HEAD, BASE)
    names = lambda records: [r.nodeid for r in records]  # noqa: E731
    assert names(diff.new_failures) == ["t::regressed"]
    assert names(diff.still_failing) == ["t::still"]
    assert names(diff.fixed) == ["t::fixed"]
    assert names(diff.flaky) == ["t::flaky"]
    assert names(diff.infra) == ["t::still"]
    assert names(diff.quarantine_hidden) == ["t::hidden"]
    assert names(diff.quarantine_passing) == ["t::qpass"]
    assert names(diff.lost_coverage) == ["t::lost"]
    assert [s.nodeid for s in diff.slower] == ["t::slow"]
    assert diff.removed == ["t::gone"]


def test_small_absolute_slowdowns_are_noise() -> None:
    base = run("b", TestRecord("t::fast", "passed", duration_s=0.01))
    head = run("h", TestRecord("t::fast", "passed", duration_s=0.05))  # 5x, but 40 ms
    assert diff_runs(head, base).slower == []


def test_markdown_has_a_section_per_bucket() -> None:
    text = render_markdown(diff_runs(HEAD, BASE))
    assert text.startswith("## gpu-testkit: head — ❌ gating failures")
    for heading in (
        "New failures (1)",
        "Still failing (1)",
        "Fixed (1)",
        "Flaky (passed on rerun) (1)",
        "Quarantine is hiding (1)",
        "Lost coverage",
        "Slowest regressions (1)",
        "Removed since base (1)",
    ):
        assert f"### {heading}" in text
    assert "| `t::regressed` | product | 3 | AssertionError: logits | `d.tgz` |" in text
    assert "| `t::slow` | 1.00s | 2.50s | x2.50 |" in text


def test_results_round_trip(tmp_path: Path) -> None:
    write_run(tmp_path / "r.jsonl", HEAD)
    loaded = read_run(tmp_path / "r.jsonl")
    assert loaded.tests == HEAD.tests
    assert loaded.run_id == "head"


def test_read_run_rejects_headerless_files(tmp_path: Path) -> None:
    (tmp_path / "r.jsonl").write_text('{"kind": "test", "nodeid": "x", "outcome": "passed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing run header"):
        read_run(tmp_path / "r.jsonl")


def test_cli_report_gates_on_new_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_run(tmp_path / "base.jsonl", BASE)
    write_run(tmp_path / "head.jsonl", HEAD)
    summary = tmp_path / "summary.md"
    code = main(
        [
            "report",
            str(tmp_path / "head.jsonl"),
            "--base",
            str(tmp_path / "base.jsonl"),
            "--output",
            str(summary),
            "--fail-on-new",
        ]
    )
    assert code == 1
    assert "1 new failure(s) vs base" in capsys.readouterr().err
    assert "### New failures (1)" in summary.read_text(encoding="utf-8")
    assert main(["report", str(tmp_path / "base.jsonl"), "--base", str(tmp_path / "base.jsonl"), "--fail-on-new"]) == 0


def test_cli_hw_prints_inventory(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["hw", "--backend", "fake", "--fake-spec", "h100x2"]) == 0
    out = capsys.readouterr().out
    assert "2x NVIDIA H100 80GB HBM3 (sm90, 85.5 GB)" in out
    assert "[1] NVIDIA H100 80GB HBM3  sm90" in out


def test_cli_quarantine_audit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "q.toml"
    path.write_text(
        '[[quarantine]]\ntest = "a::b"\nowner = "@o"\nreason = "r"\nexpires = "2026-11-01"\n', encoding="utf-8"
    )
    assert main(["quarantine", str(path), "--today", "2026-11-14", "--fail-on-expired"]) == 1
    assert "EXPIRED  2026-11-01  @o" in capsys.readouterr().out
    path.write_text("[[quarantine]]\ntest = 'a::b'\n", encoding="utf-8")
    assert main(["quarantine", str(path)]) == 2


def test_cli_pool_shows_holders(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gpu_testkit.pool import DevicePool

    with DevicePool(tmp_path).lease([1], 1, owner="tests/test_tp.py::test_tp2"):
        assert main(["pool", "--backend", "fake", "--fake-spec", "h100x2", "--lock-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "[0] NVIDIA H100 80GB HBM3        free" in out
    assert "[1] NVIDIA H100 80GB HBM3        tests/test_tp.py::test_tp2 (pid" in out
