"""Diagnostics bundles: isolated collectors, redaction, and Xid events scoped to the test."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from gpu_testkit.backends.fake import FakeBackend, load_spec
from gpu_testkit.diagnostics import (
    BUILTIN_COLLECTORS,
    DiagnosticContext,
    EnvCollector,
    KernelLogCollector,
    Payload,
    bundle_name,
    write_bundle,
)


class BrokenCollector:
    name = "broken"

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, Payload]:
        raise RuntimeError("collector bug")


def context(fixtures_dir: Path, offset: int = 0) -> DiagnosticContext:
    spec = load_spec(
        f'{{"gpus": [{{"name": "H", "arch": "sm90", "mem_gb": 80}}], "kernel_log": "{fixtures_dir / "dmesg-xid.log"}"}}'
    )
    return DiagnosticContext(
        nodeid="tests/test_x.py::test_y[fp8]",
        host=spec.host,
        backend=FakeBackend(spec),
        verdict="product",
        attempts=2,
        devices=(0,),
        failure_text="AssertionError: boom",
        captured=[("Captured stdout call", "hello")],
        timings=[{"attempt": 1, "phase": "call", "seconds": 0.1}],
        kernel_log_offset=offset,
    )


def members(path: Path) -> dict[str, str]:
    with tarfile.open(path, "r:gz") as tar:
        return {m.name: tar.extractfile(m).read().decode() for m in tar.getmembers()}  # type: ignore[union-attr]


def test_bundle_contains_every_builtin_collector(tmp_path: Path, fixtures_dir: Path) -> None:
    path = write_bundle(tmp_path, context(fixtures_dir), BUILTIN_COLLECTORS)
    files = members(path)
    assert set(files) >= {
        "manifest.json",
        "env/env.json",
        "hardware/host.json",
        "nvidia-smi/nvidia-smi.txt",
        "kernel-log/dmesg-tail.txt",
        "kernel-log/xid-during-test.json",
        "test/test.log",
        "test/timings.json",
    }
    assert "AssertionError: boom" in files["test/test.log"]
    assert "hello" in files["test/test.log"]
    assert json.loads(files["hardware/host.json"])["leased_devices"] == [0]
    assert files["nvidia-smi/nvidia-smi.txt"].startswith("index, name, uuid, compute_cap")


def test_broken_collector_is_recorded_not_raised(tmp_path: Path, fixtures_dir: Path) -> None:
    path = write_bundle(tmp_path, context(fixtures_dir), [BrokenCollector(), *BUILTIN_COLLECTORS])
    manifest = json.loads(members(path)["manifest.json"])
    assert manifest["collectors"]["broken"] == {"error": "RuntimeError: collector bug"}
    assert "files" in manifest["collectors"]["env"]


def test_only_xids_logged_during_the_test_are_attributed(fixtures_dir: Path) -> None:
    files = KernelLogCollector().collect(context(fixtures_dir, offset=4))
    codes = [event["code"] for event in json.loads(str(files["xid-during-test.json"]))]
    assert codes == [74, 94]  # Xid 31 and 79 were already in the log when the test started


def test_env_collector_redacts_secrets(monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path) -> None:
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("GPU_TESTKIT_API_KEY", "sk-not-for-logs")
    env = json.loads(str(EnvCollector().collect(context(fixtures_dir))["env.json"]))["env"]
    assert env["NCCL_DEBUG"] == "INFO"
    assert env["GPU_TESTKIT_API_KEY"] == "<redacted>"


def test_bundle_names_are_filesystem_safe() -> None:
    assert bundle_name("tests/test_x.py::test_y[fp8-8 gpus]") == "tests_test_x.py_test_y_fp8-8_gpus.tar.gz"
