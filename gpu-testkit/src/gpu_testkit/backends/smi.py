"""Discovery by parsing ``nvidia-smi`` output, for hosts without pynvml installed."""

from __future__ import annotations

import os
import re

from gpu_testkit.backends.base import hostname, mib_to_gb, read_dmesg, run_command
from gpu_testkit.hardware import GpuInfo, HostInfo

QUERY_FIELDS = "index,name,uuid,compute_cap,memory.total,driver_version"
QUERY_ARGV = ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"]
CUDA_VERSION_PATTERN = re.compile(r"CUDA Version:\s*([0-9.]+)")


def parse_query_csv(text: str) -> tuple[list[GpuInfo], str | None]:
    """Parse ``--query-gpu=index,name,uuid,compute_cap,memory.total,driver_version``.

    Accepts both ``noheader,nounits`` output and the default form with a header row
    and ``MiB`` suffixes.
    """
    gpus: list[GpuInfo] = []
    driver: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("index"):
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"unexpected nvidia-smi row ({len(fields)} fields): {line!r}")
        index, name, uuid, cap, mem, driver_version = fields
        major, _, minor = cap.partition(".")
        gpus.append(
            GpuInfo(
                index=int(index),
                name=name,
                arch=int(major) * 10 + int(minor or 0),
                mem_gb=mib_to_gb(float(mem.removesuffix("MiB").strip())),
                uuid=uuid,
            )
        )
        driver = driver_version
    return gpus, driver


def parse_cuda_version(banner: str) -> str | None:
    """Pull ``CUDA Version: 12.6`` out of the plain ``nvidia-smi`` banner."""
    match = CUDA_VERSION_PATTERN.search(banner)
    return match.group(1) if match else None


class SmiBackend:
    name = "smi"

    def discover(self) -> HostInfo:
        text = run_command(QUERY_ARGV)
        if text is None:
            return HostInfo(hostname=hostname(), backend=self.name)
        gpus, driver = parse_query_csv(text)
        banner = run_command(["nvidia-smi"]) or ""
        return HostInfo(
            hostname=hostname(),
            backend=self.name,
            gpus=tuple(gpus),
            driver_version=driver,
            cuda_version=parse_cuda_version(banner),
            nodes=job_nodes(),
        )

    def smi_text(self) -> str | None:
        return run_command(["nvidia-smi", "-q"])

    def kernel_log(self) -> list[str] | None:
        return read_dmesg()


def job_nodes() -> int:
    """Node count of the current job: explicit override, then Slurm, else 1."""
    for var in ("GPU_TESTKIT_NODES", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        value = os.environ.get(var)
        if value and value.isdigit():
            return int(value)
    return 1
