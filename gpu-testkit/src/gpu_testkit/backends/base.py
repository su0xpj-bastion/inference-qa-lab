"""The backend protocol and helpers shared by every backend."""

from __future__ import annotations

import shutil
import socket
import subprocess
from typing import Protocol, runtime_checkable

from gpu_testkit.hardware import HostInfo

ENCODING = "utf-8"
COMMAND_TIMEOUT_S = 10.0
BYTES_PER_GB = 1_000_000_000  # decimal GB, so an "80GB" H100 (81559 MiB) qualifies for mem_gb=80
BYTES_PER_MIB = 1024 * 1024


@runtime_checkable
class Backend(Protocol):
    """Discovers hardware and exposes the raw evidence diagnostics need.

    ``discover`` must be cheap enough to call once per session. ``smi_text`` and
    ``kernel_log`` are only called when a test fails, and may return None when the
    information is unavailable (no binary, no permission).
    """

    name: str

    def discover(self) -> HostInfo: ...

    def smi_text(self) -> str | None: ...

    def kernel_log(self) -> list[str] | None: ...


def mib_to_gb(mib: float) -> float:
    return round(mib * BYTES_PER_MIB / BYTES_PER_GB, 1)


def bytes_to_gb(total: int) -> float:
    return round(total / BYTES_PER_GB, 1)


def run_command(argv: list[str]) -> str | None:
    """Run a diagnostic command; None if the binary is missing or the call fails."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding=ENCODING,
            errors="replace",
            timeout=COMMAND_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def read_dmesg() -> list[str] | None:
    """Kernel ring buffer, where NVRM Xid events land. Often needs privileges."""
    out = run_command(["dmesg"])
    return None if out is None else out.splitlines()


def hostname() -> str:
    return socket.gethostname()


class NullBackend:
    """A host with no GPUs. Every hardware-marked test skips with a reason."""

    name = "none"

    def discover(self) -> HostInfo:
        return HostInfo(hostname=hostname(), backend=self.name)

    def smi_text(self) -> str | None:
        return None

    def kernel_log(self) -> list[str] | None:
        return None
