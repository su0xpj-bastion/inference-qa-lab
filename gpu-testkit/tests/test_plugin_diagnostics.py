"""Diagnostics bundles produced by the plugin, attached to JUnit, extensible via a hook."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from tests.conftest import junit_cases

SUITE = """
import pytest

@pytest.mark.gpus(2)
def test_tp_forward(gpu_devices):
    print("loss=nan at step 3")
    assert False, "NaN in logits"

def test_ok():
    pass
"""


def bundle_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as tar:
        return sorted(tar.getnames())


def test_failure_writes_bundle_and_junit_points_at_it(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest(
        "--gpu-fake-spec=h100x4", "--gpu-diagnostics-dir=diag", "--gpu-run-id=run-7", "--junitxml=j.xml"
    )
    result.assert_outcomes(passed=1, failed=1)
    case = junit_cases(pytester.path / "j.xml")["test_tp_forward"]
    bundle = Path(case["gpu_testkit.diagnostics"])
    assert bundle == pytester.path / "diag" / "run-7" / "test_suite.py_test_tp_forward.tar.gz"
    assert "test/test.log" in bundle_members(bundle)
    with tarfile.open(bundle, "r:gz") as tar:
        log = tar.extractfile("test/test.log").read().decode()  # type: ignore[union-attr]
        host = json.loads(tar.extractfile("hardware/host.json").read())  # type: ignore[union-attr]
    assert "NaN in logits" in log
    assert "loss=nan at step 3" in log
    assert host["leased_devices"] == [0, 1]
    assert host["driver_version"] == "560.35.03"
    assert "gpu_testkit.diagnostics" not in junit_cases(pytester.path / "j.xml")["test_ok"]
    result.stdout.fnmatch_lines([f"diagnostics: {bundle}"])


def test_custom_collectors_via_hook_and_broken_ones_are_contained(pytester: pytest.Pytester) -> None:
    pytester.makeconftest("""
        import pytest

        class NcclDebug:
            name = "nccl"
            def collect(self, ctx):
                return {"nccl-debug.txt": f"NCCL INFO ring for {ctx.nodeid} on devices {ctx.devices}"}

        class Broken:
            name = "broken"
            def collect(self, ctx):
                raise OSError("rdma tool missing")

        @pytest.hookimpl
        def pytest_gpu_testkit_diagnostic_collectors(config):
            return [NcclDebug(), Broken()]
    """)
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest("--gpu-fake-spec=h100x4", "--gpu-diagnostics-dir=diag", "--junitxml=j.xml")
    result.assert_outcomes(passed=1, failed=1)  # a broken collector never becomes an internal error
    bundle = Path(junit_cases(pytester.path / "j.xml")["test_tp_forward"]["gpu_testkit.diagnostics"])
    with tarfile.open(bundle, "r:gz") as tar:
        nccl = tar.extractfile("nccl/nccl-debug.txt").read().decode()  # type: ignore[union-attr]
        manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]
    assert nccl.endswith("on devices (0, 1)")
    assert manifest["collectors"]["broken"] == {"error": "OSError: rdma tool missing"}
    assert "files" in manifest["collectors"]["env"]


def test_diagnostics_can_be_disabled(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_suite=SUITE)
    result = pytester.runpytest("--gpu-fake-spec=h100x4", "--gpu-diagnostics-dir=diag", "--gpu-no-diagnostics")
    result.assert_outcomes(passed=1, failed=1)
    assert not (pytester.path / "diag").exists()
