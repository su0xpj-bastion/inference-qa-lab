from __future__ import annotations

# All-reduce step: a stub interface for nccl-tests / rccl-tests.
#
# The pipeline only needs "run all_reduce_perf across these hosts and give me
# the text output". FixtureRunner answers from captured output (what CI and the
# simulated fleet use). NcclTestsRunner builds the real mpirun command line and
# runs it when the binary exists; on this repo's CI it never does.

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from bringup.parsers.common import ENCODING

MAX_ATTEMPTS = 2  # one retry: a first collective can lose to a slow SM/IB bring-up


class AllReduceRunner(Protocol):
    def run(self, pool: str, hosts: list[str], gpus_per_node: int) -> str: ...


class FixtureRunner:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, pool: str, hosts: list[str], gpus_per_node: int) -> str:
        path = self.root / "_cluster" / f"all_reduce_perf.{pool}.txt"
        if len(hosts) == 1:
            path = self.root / hosts[0] / "all_reduce_perf.txt"
        with path.open("r", encoding=ENCODING) as fh:
            return fh.read()


class NcclTestsRunner:
    def __init__(self, binary: str = "all_reduce_perf", mpirun: str = "mpirun",
                 extra_env: dict[str, str] | None = None) -> None:
        self.binary = binary
        self.mpirun = mpirun
        self.extra_env = extra_env or {}

    def command(self, hosts: list[str], gpus_per_node: int) -> list[str]:
        nranks = len(hosts) * gpus_per_node
        cmd = [self.mpirun, "-np", str(nranks), "-H",
               ",".join(f"{h}:{gpus_per_node}" for h in hosts),
               "--bind-to", "numa", "-x", "NCCL_DEBUG=WARN"]
        for k, v in sorted(self.extra_env.items()):
            cmd += ["-x", f"{k}={v}"]
        # 8 MiB .. 8 GiB, x4 steps, 1 GPU per rank, data validation on.
        cmd += [self.binary, "-b", "8M", "-e", "8G", "-f", "4", "-g", "1", "-c", "1"]
        return cmd

    def run(self, pool: str, hosts: list[str], gpus_per_node: int) -> str:
        if shutil.which(self.mpirun) is None or shutil.which(self.binary) is None:
            raise RuntimeError(f"{self.mpirun}/{self.binary} not on PATH: use FixtureRunner off-cluster")
        last = ""
        for attempt in range(MAX_ATTEMPTS):
            proc = subprocess.run(self.command(hosts, gpus_per_node), capture_output=True,
                                  text=True, timeout=1800, check=False)
            if proc.returncode == 0:
                return proc.stdout
            last = proc.stderr[-2000:]
        raise RuntimeError(f"all_reduce_perf failed after {MAX_ATTEMPTS} attempts: {last}")
