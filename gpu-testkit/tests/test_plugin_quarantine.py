"""Quarantined tests still run, stop gating, and the summary says what quarantine is hiding."""

from __future__ import annotations

import pytest

from tests.conftest import junit_cases

SUITE = """
def test_known_bad():
    assert False, "allreduce hangs on IB firmware 28.39"

def test_fixed_upstream():
    pass

def test_regular():
    pass
"""

QUARANTINE = """
[[quarantine]]
test = "test_suite.py::test_known_bad"
owner = "@comms"
reason = "IB firmware hang"
ticket = "GPU-1432"
expires = "{bad}"

[[quarantine]]
test = "test_suite.py::test_fixed_upstream"
owner = "@kernels"
reason = "fp8 gemm mismatch"
ticket = "GPU-1500"
expires = "2026-12-31"

[[quarantine]]
test = "test_suite.py::test_renamed_long_ago"
owner = "@nobody"
reason = "?"
expires = "2026-12-31"
"""


@pytest.fixture
def suite(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    monkeypatch.setenv("GPU_TESTKIT_TODAY", "2026-11-14")
    pytester.makepyfile(test_suite=SUITE)
    return pytester


def test_quarantined_failure_does_not_gate_but_is_reported(suite: pytest.Pytester) -> None:
    suite.makefile(".toml", quarantine=QUARANTINE.format(bad="2026-12-01"))
    result = suite.runpytest("--gpu-fake-spec=h100x1", "--gpu-quarantine=quarantine.toml", "--junitxml=j.xml")
    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=2, xfailed=1)
    result.stdout.fnmatch_lines(
        [
            "gpu-testkit: reruns=0, quarantine=3 entries (0 expired)*",
            "quarantine is hiding 1 failure(s):",
            "  test_suite.py::test_known_bad: AssertionError: allreduce hangs on IB firmware 28.39*"
            "[[]quarantined GPU-1432 (@comms, until 2026-12-01): IB firmware hang]",
            "1 quarantined test(s) passed; consider releasing:",
            "  test_suite.py::test_fixed_upstream*",
            "STALE  quarantine entry matches no test that ran: test_suite.py::test_renamed_long_ago (@nobody)",
        ]
    )
    cases = junit_cases(suite.path / "j.xml")
    assert cases["test_known_bad"]["_result"] == "skipped"
    assert cases["test_known_bad"]["gpu_testkit.quarantine"].startswith("quarantined GPU-1432")
    assert cases["test_known_bad"]["gpu_testkit.verdict"] == "product"


def test_expired_quarantine_gates_again(suite: pytest.Pytester) -> None:
    suite.makefile(".toml", quarantine=QUARANTINE.format(bad="2026-11-13"))
    result = suite.runpytest("--gpu-fake-spec=h100x1", "--gpu-quarantine=quarantine.toml")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        ["EXPIRED quarantine for test_suite.py::test_known_bad (@comms, 2026-11-13): gating again"]
    )


def test_quarantine_path_from_ini_is_relative_to_ini(suite: pytest.Pytester) -> None:
    (suite.path / "ci").mkdir()
    (suite.path / "ci" / "quarantine.toml").write_text(QUARANTINE.format(bad="2026-12-01"), encoding="utf-8")
    suite.makeini("[pytest]\ngpu_quarantine = ci/quarantine.toml\n")
    result = suite.runpytest("--gpu-fake-spec=h100x1")
    result.assert_outcomes(passed=2, xfailed=1)


def test_invalid_quarantine_file_is_a_usage_error(suite: pytest.Pytester) -> None:
    suite.makefile(".toml", quarantine='[[quarantine]]\ntest = "x"\nreason = "r"\nexpires = "2026-12-01"\n')
    result = suite.runpytest("--gpu-quarantine=quarantine.toml")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*entry 1: missing owner*"])
