"""Discovery through NVML (``pynvml``, shipped by the ``nvidia-ml-py`` package)."""

from __future__ import annotations

import importlib
from types import ModuleType

from gpu_testkit.backends.base import bytes_to_gb, hostname, read_dmesg, run_command
from gpu_testkit.backends.smi import job_nodes
from gpu_testkit.hardware import GpuInfo, HostInfo, arch_from_capability


def load_pynvml() -> ModuleType | None:
    """Import and initialise NVML; None when the library or the driver is absent."""
    try:
        pynvml = importlib.import_module("pynvml")
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:  # pynvml raises NVMLError subclasses that are generated at runtime
        return None
    return pynvml


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def format_cuda_version(raw: int) -> str:
    """NVML encodes CUDA 12.6 as 12060."""
    return f"{raw // 1000}.{(raw % 1000) // 10}"


class NvmlBackend:
    """Real hardware. Note NVML enumerates every device and ignores CUDA_VISIBLE_DEVICES."""

    name = "nvml"

    def __init__(self, pynvml: ModuleType) -> None:
        self._nvml = pynvml

    def discover(self) -> HostInfo:
        nvml = self._nvml
        gpus: list[GpuInfo] = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            gpus.append(
                GpuInfo(
                    index=index,
                    name=_text(nvml.nvmlDeviceGetName(handle)),
                    arch=arch_from_capability(major, minor),
                    mem_gb=bytes_to_gb(nvml.nvmlDeviceGetMemoryInfo(handle).total),
                    uuid=_text(nvml.nvmlDeviceGetUUID(handle)),
                )
            )
        return HostInfo(
            hostname=hostname(),
            backend=self.name,
            gpus=tuple(gpus),
            driver_version=_text(nvml.nvmlSystemGetDriverVersion()),
            cuda_version=format_cuda_version(nvml.nvmlSystemGetCudaDriverVersion()),
            nodes=job_nodes(),
        )

    def smi_text(self) -> str | None:
        return run_command(["nvidia-smi", "-q"])

    def kernel_log(self) -> list[str] | None:
        return read_dmesg()
