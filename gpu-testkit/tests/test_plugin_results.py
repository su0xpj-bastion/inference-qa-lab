"""JUnit enrichment and the JSONL result store, in-process and under xdist, then diffed by the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_testkit.cli import main
from gpu_testkit.results import read_run
from tests.conftest import junit_cases

SUITE = """
import pytest

@pytest.mark.gpus(2, arch=">=sm90")
def test_tp2(gpu_devices):
    pass

@pytest.mark.gpus(16)
def test_needs_16():
    pass

def test_cpu_ref():
    assert {fail} is False
"""


def test_junit_carries_hardware_tags(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_suite=SUITE.format(fail=False))
    pytester.runpytest("--gpu-fake-spec=h100x4", "--junitxml=j.xml").assert_outcomes(passed=2, skipped=1)
    case = junit_cases(pytester.path / "j.xml")["test_tp2"]
    assert case["gpu_testkit.devices"] == "0,1"
    assert case["gpu_testkit.gpu"] == "NVIDIA H100 80GB HBM3 sm90"
    assert case["gpu_testkit.requires"] == "gpus=2, arch>=sm90"
    assert case["gpu_testkit.host"] == "fake-h100-node0"
    assert case["gpu_testkit.backend"] == "fake"
    assert (case["gpu_testkit.verdict"], case["gpu_testkit.attempts"]) == ("passed", "1")
    assert list(case).count("gpu_testkit.verdict") == 1


@pytest.mark.parametrize("workers", [[], ["-n", "2"]], ids=["in-process", "xdist"])
def test_results_store(pytester: pytest.Pytester, workers: list[str]) -> None:
    pytester.makepyfile(test_suite=SUITE.format(fail=True))
    result = pytester.runpytest(
        "--gpu-fake-spec=h100x4", "--gpu-results=out/results.jsonl", "--gpu-run-id=r1", "--gpu-no-diagnostics", *workers
    )
    result.assert_outcomes(passed=1, failed=1, skipped=1)
    run = read_run(pytester.path / "out" / "results.jsonl")
    assert run.run_id == "r1"
    assert run.host["backend"] == "fake"
    tests = {nodeid.split("::")[1]: record for nodeid, record in run.tests.items()}
    assert (tests["test_tp2"].outcome, tests["test_tp2"].tags["devices"]) == ("passed", "0,1")
    assert (tests["test_needs_16"].outcome, tests["test_needs_16"].verdict) == ("skipped", "skipped")
    assert "needs 16 GPU(s)" in tests["test_needs_16"].message
    assert (tests["test_cpu_ref"].outcome, tests["test_cpu_ref"].verdict) == ("failed", "product")
    assert tests["test_cpu_ref"].message.startswith("assert True is False")
    header = json.loads((pytester.path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert header["kind"] == "run"
    assert header["meta"]["exitstatus"] == "1"


def test_two_runs_diffed_by_the_cli(pytester: pytest.Pytester, capsys: pytest.CaptureFixture[str]) -> None:
    pytester.makepyfile(test_suite=SUITE.format(fail=False))
    pytester.runpytest("--gpu-fake-spec=h100x4", "--gpu-results=base.jsonl", "--gpu-run-id=base")
    pytester.makepyfile(test_suite=SUITE.format(fail=True))
    pytester.runpytest(
        "--gpu-fake-spec=h100x4", "--gpu-results=head.jsonl", "--gpu-run-id=head", "--gpu-no-diagnostics"
    )
    code = main(
        ["report", str(pytester.path / "head.jsonl"), "--base", str(pytester.path / "base.jsonl"), "--fail-on-new"]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "## gpu-testkit: head — ❌ gating failures" in out
    assert "### New failures (1)" in out
    assert "`test_suite.py::test_cpu_ref` | product" in out


def test_quarantined_outcome_in_store(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPU_TESTKIT_TODAY", "2026-11-14")
    pytester.makefile(
        ".toml",
        q='[[quarantine]]\ntest = "test_suite.py::test_cpu_ref"\nowner = "@o"\nreason = "r"\nexpires = "2026-12-01"\n',
    )
    pytester.makepyfile(test_suite=SUITE.format(fail=True))
    result = pytester.runpytest(
        "--gpu-fake-spec=h100x4", "--gpu-quarantine=q.toml", "--gpu-results=r.jsonl", "--gpu-no-diagnostics"
    )
    assert result.ret == pytest.ExitCode.OK
    record = read_run(pytester.path / "r.jsonl").tests["test_suite.py::test_cpu_ref"]
    assert record.outcome == "quarantined"
    assert record.quarantine.startswith("quarantined (@o")
    assert not record.gating_failure
