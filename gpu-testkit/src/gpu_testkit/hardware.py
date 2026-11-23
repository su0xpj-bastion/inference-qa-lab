"""Hardware model: what a host offers and how a test states what it needs.

Everything here is plain data. Backends produce a :class:`HostInfo`; markers are
parsed into a :class:`Requirement`; :func:`Requirement.unmet_reason` compares the two.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

ARCH_PATTERN = re.compile(r"^\s*(>=|<=|==|!=|>|<)?\s*sm_?(\d{2,3})\s*$", re.IGNORECASE)

_OPS: dict[str, Callable[[int, int], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}

# Lowest SM architecture with native support for each dtype (tensor-core path).
# fp8 arrived with Ada (sm89) / Hopper (sm90); bf16 with Ampere (sm80).
DTYPE_MIN_ARCH: dict[str, int] = {
    "fp32": 0,
    "tf32": 80,
    "fp16": 0,
    "bf16": 80,
    "fp8": 89,
    "fp8_e4m3": 89,
    "fp8_e5m2": 89,
    "fp4": 100,
}


@dataclass(frozen=True)
class GpuInfo:
    """One visible device, as a backend reports it."""

    index: int
    name: str
    arch: int  # SM version as an integer: sm90 -> 90
    mem_gb: float
    uuid: str = ""

    @property
    def arch_label(self) -> str:
        return f"sm{self.arch}"


@dataclass(frozen=True)
class HostInfo:
    """A snapshot of the host (and job) the suite is running on."""

    hostname: str
    backend: str
    gpus: tuple[GpuInfo, ...] = ()
    driver_version: str | None = None
    cuda_version: str | None = None
    nodes: int = 1

    def summary(self) -> str:
        if not self.gpus:
            return f"{self.hostname}: no GPUs (backend={self.backend})"
        names: dict[str, int] = {}
        for gpu in self.gpus:
            key = f"{gpu.name} ({gpu.arch_label}, {gpu.mem_gb:g} GB)"
            names[key] = names.get(key, 0) + 1
        parts = ", ".join(f"{count}x {name}" for name, count in names.items())
        versions = f"driver {self.driver_version or '?'}, CUDA {self.cuda_version or '?'}"
        return f"{self.hostname}: {parts}; {versions}; nodes={self.nodes} (backend={self.backend})"


@dataclass(frozen=True)
class ArchSpec:
    """A comparison such as ``>=sm90``. A bare ``sm90`` means ``==sm90``."""

    op: str
    arch: int

    @classmethod
    def parse(cls, text: str) -> ArchSpec:
        match = ARCH_PATTERN.match(text)
        if match is None:
            raise ValueError(f"invalid arch spec {text!r}; expected e.g. 'sm90', '>=sm80', '<sm100'")
        return cls(op=match.group(1) or "==", arch=int(match.group(2)))

    def matches(self, arch: int) -> bool:
        return _OPS[self.op](arch, self.arch)

    def __str__(self) -> str:
        return f"{self.op}sm{self.arch}"


@dataclass(frozen=True)
class Requirement:
    """What a test needs from the host, merged from its gpu-testkit markers."""

    gpus: int = 0
    arch: tuple[ArchSpec, ...] = ()
    mem_gb: float = 0.0
    nodes: int = 1
    dtypes: tuple[str, ...] = field(default=())

    def device_qualifies(self, gpu: GpuInfo) -> bool:
        if gpu.mem_gb < self.mem_gb:
            return False
        if not all(spec.matches(gpu.arch) for spec in self.arch):
            return False
        return all(gpu.arch >= DTYPE_MIN_ARCH[dtype] for dtype in self.dtypes)

    def qualifying_devices(self, host: HostInfo) -> list[GpuInfo]:
        return [gpu for gpu in host.gpus if self.device_qualifies(gpu)]

    def unmet_reason(self, host: HostInfo) -> str | None:
        """Return a human-readable skip reason, or None when the host qualifies."""
        if self.nodes > host.nodes:
            return f"needs {self.nodes} nodes, job has {host.nodes} ({_host_tag(host)})"
        needed = max(self.gpus, 1 if (self.dtypes or self.arch or self.mem_gb) else 0)
        if needed == 0:
            return None
        qualifying = self.qualifying_devices(host)
        if len(qualifying) >= needed:
            return None
        return (
            f"needs {needed} GPU(s) matching [{self.describe()}], "
            f"host has {len(qualifying)} of {len(host.gpus)} qualifying ({_host_tag(host)})"
        )

    def describe(self) -> str:
        parts: list[str] = []
        if self.gpus:
            parts.append(f"gpus={self.gpus}")
        parts.extend(f"arch{spec}" for spec in self.arch)
        if self.mem_gb:
            parts.append(f"mem>={self.mem_gb:g}GB")
        parts.extend(f"dtype={dtype}(>=sm{DTYPE_MIN_ARCH[dtype]})" for dtype in self.dtypes)
        if self.nodes > 1:
            parts.append(f"nodes={self.nodes}")
        return ", ".join(parts) or "none"


def _host_tag(host: HostInfo) -> str:
    if not host.gpus:
        return f"{host.hostname}: no GPUs, backend={host.backend}"
    archs = sorted({gpu.arch_label for gpu in host.gpus})
    mems = sorted({gpu.mem_gb for gpu in host.gpus})
    return f"{host.hostname}: {len(host.gpus)} GPUs {'/'.join(archs)} {'/'.join(f'{m:g}' for m in mems)}GB"


def parse_arch_specs(value: str | Sequence[str] | None) -> tuple[ArchSpec, ...]:
    """Accept ``">=sm80,<sm100"`` or ``[">=sm80", "<sm100"]``."""
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else list(value)
    return tuple(ArchSpec.parse(item) for item in items if item.strip())


def arch_from_capability(major: int, minor: int) -> int:
    """NVML/CUDA compute capability (9, 0) -> 90."""
    return major * 10 + minor
