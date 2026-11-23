"""gpu-testkit: a pytest plugin and diagnostics toolkit for GPU inference test suites."""

from __future__ import annotations

from gpu_testkit.backends import (
    Backend,
    FakeBackend,
    NullBackend,
    NvmlBackend,
    SmiBackend,
    select_backend,
)
from gpu_testkit.classify import InfraError, Verdict
from gpu_testkit.diagnostics import Collector, DiagnosticContext
from gpu_testkit.hardware import ArchSpec, GpuInfo, HostInfo, Requirement
from gpu_testkit.numerics import CloseReport, assert_close, compare, quantize, ulp_distance
from gpu_testkit.pool import DevicePool, Lease, PoolTimeout
from gpu_testkit.quarantine import QuarantineEntry, QuarantineList
from gpu_testkit.results import RunRecord, TestRecord, read_run

__version__ = "0.1.0"

__all__ = [
    "ArchSpec",
    "Backend",
    "CloseReport",
    "Collector",
    "DevicePool",
    "DiagnosticContext",
    "FakeBackend",
    "GpuInfo",
    "HostInfo",
    "InfraError",
    "Lease",
    "NullBackend",
    "NvmlBackend",
    "PoolTimeout",
    "QuarantineEntry",
    "QuarantineList",
    "Requirement",
    "RunRecord",
    "SmiBackend",
    "TestRecord",
    "Verdict",
    "__version__",
    "assert_close",
    "compare",
    "quantize",
    "read_run",
    "select_backend",
    "ulp_distance",
]
