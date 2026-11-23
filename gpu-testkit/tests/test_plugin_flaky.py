"""Rerun-to-classify through the real plugin: flaky vs product vs infra."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import junit_cases

FLAKY_SUITE = """
import pytest
from pathlib import Path

COUNTER = Path(__file__).with_name("counter.txt")
SETUPS = Path(__file__).with_name("setups.txt")

@pytest.fixture(scope="module")
def expensive_model():
    with SETUPS.open("a") as fh:
        fh.write("x")
    return "model"

def test_flaky(expensive_model):
    n = int(COUNTER.read_text()) if COUNTER.exists() else 0
    COUNTER.write_text(str(n + 1))
    assert n >= 1, "simulated NCCL hiccup on first attempt"

def test_product(expensive_model):
    assert 1 + 1 == 3, "logits mismatch"

def test_stable(expensive_model):
    pass
"""


def test_flaky_and_product_are_told_apart(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_suite=FLAKY_SUITE)
    result = pytester.runpytest("--gpu-fake-spec=h100x1", "--gpu-reruns=2", "--junitxml=j.xml", "--gpu-no-diagnostics")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        [
            "*gpu-testkit: attempt 2 failed*",
            "FLAKY  test_suite.py::test_flaky: passed on attempt 2 (failed 1 of 2 attempts)",
        ]
    )
    cases = junit_cases(pytester.path / "j.xml")
    assert (cases["test_flaky"]["gpu_testkit.verdict"], cases["test_flaky"]["gpu_testkit.attempts"]) == ("flaky", "2")
    assert (cases["test_product"]["gpu_testkit.verdict"], cases["test_product"]["gpu_testkit.attempts"]) == (
        "product",
        "3",
    )
    assert cases["test_product"]["_result"] == "failure"
    assert cases["test_stable"]["gpu_testkit.attempts"] == "1"
    # The module fixture survived both retries: built once for the whole module.
    assert (pytester.path / "setups.txt").read_text() == "x"


def test_no_reruns_by_default(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_suite=FLAKY_SUITE)
    result = pytester.runpytest("--gpu-fake-spec=h100x1", "--junitxml=j.xml", "--gpu-no-diagnostics")
    result.assert_outcomes(passed=1, failed=2)
    assert junit_cases(pytester.path / "j.xml")["test_flaky"]["gpu_testkit.verdict"] == "product"


def test_infra_from_exception_type_and_custom_ini(pytester: pytest.Pytester) -> None:
    pytester.makeini("[pytest]\ngpu_infra_exceptions =\n    gpu_testkit.classify.InfraError\n    NodeLost\n")
    pytester.makepyfile("""
        from gpu_testkit import InfraError

        class NodeLost(Exception):
            pass

        def test_lab_down():
            raise InfraError("BMC unreachable")

        def test_custom():
            raise NodeLost("node-17 lost")

        def test_timeout_is_not_infra_by_default():
            raise TimeoutError("server hung")
    """)
    result = pytester.runpytest("--gpu-fake-spec=h100x1", "--gpu-reruns=1", "--junitxml=j.xml", "--gpu-no-diagnostics")
    result.assert_outcomes(failed=3)
    cases = junit_cases(pytester.path / "j.xml")
    assert cases["test_lab_down"]["gpu_testkit.verdict"] == "infra"
    assert cases["test_custom"]["gpu_testkit.evidence"].endswith("NodeLost")
    assert cases["test_timeout_is_not_infra_by_default"]["gpu_testkit.verdict"] == "product"


def test_xid_during_test_makes_failure_infra(pytester: pytest.Pytester) -> None:
    """The fake kernel log is a file, so the test can 'trigger' an Xid like real hardware would."""
    log = pytester.path / "dmesg.log"
    log.write_text("[1.0] NVRM: Xid (PCI:0000:3b:00): 79, pid=0, GPU has fallen off the bus.\n")  # pre-existing
    pytester.makefile(
        ".json", spec=f'{{"gpus": [{{"name": "H100", "arch": "sm90", "mem_gb": 80}}], "kernel_log": "{log}"}}'
    )
    pytester.makepyfile(f"""
        LOG = {str(log)!r}

        def emit(code):
            with open(LOG, "a") as fh:
                fh.write(f"[9.0] NVRM: Xid (PCI:0000:5e:00): {{code}}, pid=1, name=python3, details\\n")

        def test_bus_drop():
            emit(79)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        def test_mmu_fault():
            emit(31)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        def test_unrelated_failure():
            assert False
    """)
    result = pytester.runpytest(
        f"--gpu-fake-spec={pytester.path / 'spec.json'}",
        "--junitxml=j.xml",
        "--gpu-no-diagnostics",
        "-p",
        "no:randomly",
    )
    result.assert_outcomes(failed=3)
    cases = junit_cases(pytester.path / "j.xml")
    assert cases["test_bus_drop"]["gpu_testkit.verdict"] == "infra"
    assert "Xid 79 on PCI 0000:5e:00" in cases["test_bus_drop"]["gpu_testkit.evidence"]
    assert cases["test_mmu_fault"]["gpu_testkit.verdict"] == "product"  # Xid 31 is the kernel's fault
    assert cases["test_unrelated_failure"]["gpu_testkit.verdict"] == "product"  # old Xid 79 not attributed


def test_pdb_mode_falls_back_to_pytest_protocol(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_ok(): pass")
    result = pytester.runpytest("--gpu-fake-spec=h100x1", "--pdb", "--gpu-reruns=3")
    result.assert_outcomes(passed=1)


def test_setup_failure_is_retried_and_reported_as_error(pytester: pytest.Pytester, tmp_path: Path) -> None:
    pytester.makepyfile("""
        import pytest

        @pytest.fixture
        def device():
            raise RuntimeError("cudaSetDevice failed")

        def test_x(device):
            pass
    """)
    result = pytester.runpytest("--gpu-fake-spec=h100x1", "--gpu-reruns=1", "--junitxml=j.xml", "--gpu-no-diagnostics")
    result.assert_outcomes(errors=1)
    assert junit_cases(pytester.path / "j.xml")["test_x"]["gpu_testkit.attempts"] == "2"
