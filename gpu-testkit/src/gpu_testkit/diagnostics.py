"""Diagnostics bundles: everything needed to triage a GPU failure without the machine.

When a test finally fails, each registered collector contributes files to a
``.tar.gz``. A collector that raises is recorded in ``manifest.json`` and the rest
still run, because a broken collector must never hide the original failure.

Add collectors from any plugin or conftest::

    @pytest.hookimpl
    def pytest_gpu_testkit_diagnostic_collectors(config):
        return [MyNcclDebugCollector()]
"""

from __future__ import annotations

import io
import json
import os
import platform
import re
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from gpu_testkit.backends.base import Backend
from gpu_testkit.hardware import HostInfo
from gpu_testkit.xid import parse_xids

ENCODING = "utf-8"
KERNEL_LOG_TAIL = 200
ENV_PREFIXES = ("CUDA_", "NCCL_", "NVIDIA_", "GPU_TESTKIT_", "TORCH_", "PYTORCH_", "SLURM_", "UCX_")
ENV_NAMES = ("LD_LIBRARY_PATH", "PATH", "HOSTNAME", "VIRTUAL_ENV", "PYTHONPATH")
SECRET_PATTERN = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY|PRIVATE", re.IGNORECASE)
REDACTED = "<redacted>"
UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

Payload = str | bytes


@dataclass
class DiagnosticContext:
    """What a collector can see about the failed test."""

    nodeid: str
    host: HostInfo
    backend: Backend
    verdict: str
    attempts: int
    devices: tuple[int, ...] = ()
    failure_text: str = ""
    captured: list[tuple[str, str]] = field(default_factory=list)
    timings: list[dict[str, float | int | str]] = field(default_factory=list)
    kernel_log_offset: int = 0  # kernel log lines that existed before the test started


@runtime_checkable
class Collector(Protocol):
    name: str

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, Payload]: ...


class EnvCollector:
    """GPU-relevant environment variables, with anything secret-looking redacted."""

    name = "env"

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, Payload]:
        env = {
            key: (REDACTED if SECRET_PATTERN.search(key) else value)
            for key, value in sorted(os.environ.items())
            if key.startswith(ENV_PREFIXES) or key in ENV_NAMES
        }
        runtime = {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "xdist_worker": os.environ.get("PYTEST_XDIST_WORKER", ""),
        }
        return {"env.json": json.dumps({"env": env, "runtime": runtime}, indent=2)}


class HardwareCollector:
    """Driver and CUDA versions plus the device inventory, from the backend."""

    name = "hardware"

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, Payload]:
        data = asdict(ctx.host)
        data["leased_devices"] = list(ctx.devices)
        return {"host.json": json.dumps(data, indent=2)}


class SmiCollector:
    """``nvidia-smi -q`` on real backends; the CSV inventory on the fake backend."""

    name = "nvidia-smi"

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, Payload]:
        text = ctx.backend.smi_text()
        return {"nvidia-smi.txt": text if text is not None else "unavailable on this host\n"}


class KernelLogCollector:
    """Tail of the kernel log and every Xid logged while the test ran."""

    name = "kernel-log"

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, Payload]:
        lines = ctx.backend.kernel_log()
        if lines is None:
            return {"dmesg-tail.txt": "unavailable (no dmesg access on this host)\n"}
        during = lines[ctx.kernel_log_offset :]
        xids = [asdict(event) | {"hardware": event.hardware, "meaning": event.meaning} for event in parse_xids(during)]
        return {
            "dmesg-tail.txt": "\n".join(lines[-KERNEL_LOG_TAIL:]) + "\n",
            "xid-during-test.json": json.dumps(xids, indent=2),
        }


class FailureLogCollector:
    """The failure, captured output and per-phase timings."""

    name = "test"

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, Payload]:
        log = [f"# {ctx.nodeid}", f"verdict: {ctx.verdict} after {ctx.attempts} attempt(s)", ""]
        log.append(ctx.failure_text)
        for title, content in ctx.captured:
            log.extend(["", f"----- {title} -----", content])
        return {
            "test.log": "\n".join(log) + "\n",
            "timings.json": json.dumps(ctx.timings, indent=2),
        }


BUILTIN_COLLECTORS: tuple[Collector, ...] = (
    EnvCollector(),
    HardwareCollector(),
    SmiCollector(),
    KernelLogCollector(),
    FailureLogCollector(),
)


def bundle_name(nodeid: str) -> str:
    return UNSAFE_CHARS.sub("_", nodeid).strip("_")[:180] + ".tar.gz"


def _add(tar: tarfile.TarFile, name: str, payload: Payload) -> int:
    data = payload.encode(ENCODING) if isinstance(payload, str) else payload
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))
    return len(data)


def write_bundle(directory: Path, ctx: DiagnosticContext, collectors: Sequence[Collector]) -> Path:
    """Run every collector in isolation and write one tarball. Returns its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / bundle_name(ctx.nodeid)
    manifest: dict[str, object] = {"nodeid": ctx.nodeid, "verdict": ctx.verdict, "collectors": {}}
    results: dict[str, object] = {}
    with tarfile.open(path, "w:gz") as tar:
        for collector in collectors:
            started = time.monotonic()
            try:
                files = collector.collect(ctx)
            except Exception as exc:  # a broken collector must not hide the test failure
                results[collector.name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            sizes = {name: _add(tar, f"{collector.name}/{name}", payload) for name, payload in files.items()}
            results[collector.name] = {
                "files": sizes,
                "seconds": round(time.monotonic() - started, 4),
            }
        manifest["collectors"] = results
        _add(tar, "manifest.json", json.dumps(manifest, indent=2))
    return path
