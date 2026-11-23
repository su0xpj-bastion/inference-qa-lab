"""A backend driven by a JSON spec, so hardware-dependent logic runs in CI without GPUs.

The spec can be a preset name (``h100x8``), inline JSON, or a path to a JSON file::

    {
      "hostname": "fake-h100-node0",
      "driver_version": "560.35.03",
      "cuda_version": "12.6",
      "nodes": 2,
      "gpus": [{"name": "NVIDIA H100 80GB HBM3", "arch": "sm90", "mem_mib": 81559}],
      "kernel_log": "dmesg.log"
    }

``gpus[*].count`` repeats an entry. ``kernel_log`` is a list of lines or a path
(relative to the spec file); tests can append Xid lines to that file mid-run to
simulate a hardware event.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_testkit.backends.base import ENCODING, mib_to_gb
from gpu_testkit.hardware import GpuInfo, HostInfo, parse_arch_specs

PRESET_PATTERN = re.compile(r"^([a-z0-9]+)x(\d+)$")

# Memory in MiB as nvidia-smi reports it for each part.
PRESET_MODELS: dict[str, dict[str, Any]] = {
    "h100": {"name": "NVIDIA H100 80GB HBM3", "arch": "sm90", "mem_mib": 81559},
    "h200": {"name": "NVIDIA H200", "arch": "sm90", "mem_mib": 143771},
    "b200": {"name": "NVIDIA B200", "arch": "sm100", "mem_mib": 183359},
    "a100": {"name": "NVIDIA A100-SXM4-80GB", "arch": "sm80", "mem_mib": 81920},
    "a100_40": {"name": "NVIDIA A100-SXM4-40GB", "arch": "sm80", "mem_mib": 40960},
    "l4": {"name": "NVIDIA L4", "arch": "sm89", "mem_mib": 23034},
    "t4": {"name": "Tesla T4", "arch": "sm75", "mem_mib": 15360},
}
DEFAULT_DRIVER = "560.35.03"
DEFAULT_CUDA = "12.6"


@dataclass(frozen=True)
class FakeSpec:
    host: HostInfo
    kernel_log_path: Path | None
    kernel_log_lines: tuple[str, ...] | None


def preset(name: str) -> dict[str, Any]:
    if name == "none":
        return {"hostname": "fake-cpu-only", "gpus": []}
    match = PRESET_PATTERN.match(name)
    if match is None or match.group(1) not in PRESET_MODELS:
        known = ", ".join(f"{model}xN" for model in PRESET_MODELS)
        raise ValueError(f"unknown fake preset {name!r}; known: none, {known}")
    model, count = match.group(1), int(match.group(2))
    return {"hostname": f"fake-{model}-node0", "gpus": [{**PRESET_MODELS[model], "count": count}]}


def load_spec(source: str) -> FakeSpec:
    """Resolve a preset name, inline JSON, or JSON file path into a FakeSpec."""
    base_dir = Path.cwd()
    text = source.strip()
    if text.startswith("{"):
        raw = json.loads(text)
    elif PRESET_PATTERN.match(text) or text == "none":
        raw = preset(text)
    else:
        path = Path(text).expanduser()
        with path.open("r", encoding=ENCODING) as handle:
            raw = json.load(handle)
        base_dir = path.resolve().parent
    if not isinstance(raw, dict):
        raise ValueError("fake spec must be a JSON object")
    return parse_spec(raw, base_dir)


def parse_spec(raw: dict[str, Any], base_dir: Path) -> FakeSpec:
    gpus: list[GpuInfo] = []
    for entry in raw.get("gpus", []):
        arch_specs = parse_arch_specs(str(entry["arch"]))
        if len(arch_specs) != 1 or arch_specs[0].op != "==":
            raise ValueError(f"gpu arch must be a plain value like 'sm90', got {entry['arch']!r}")
        mem_gb = mib_to_gb(float(entry["mem_mib"])) if "mem_mib" in entry else float(entry["mem_gb"])
        for _ in range(int(entry.get("count", 1))):
            index = len(gpus)
            gpus.append(
                GpuInfo(
                    index=index,
                    name=str(entry["name"]),
                    arch=arch_specs[0].arch,
                    mem_gb=mem_gb,
                    uuid=str(entry.get("uuid", f"GPU-fake-{index:04d}")),
                )
            )
    log = raw.get("kernel_log")
    log_path = (base_dir / log) if isinstance(log, str) else None
    log_lines = tuple(str(line) for line in log) if isinstance(log, list) else None
    host = HostInfo(
        hostname=str(raw.get("hostname", "fake-host")),
        backend="fake",
        gpus=tuple(gpus),
        driver_version=str(raw.get("driver_version", DEFAULT_DRIVER)) if gpus else None,
        cuda_version=str(raw.get("cuda_version", DEFAULT_CUDA)) if gpus else None,
        nodes=int(raw.get("nodes", 1)),
    )
    return FakeSpec(host=host, kernel_log_path=log_path, kernel_log_lines=log_lines)


def render_smi_csv(host: HostInfo) -> str:
    """Render the host the way ``nvidia-smi --query-gpu=... --format=csv`` prints it."""
    rows = ["index, name, uuid, compute_cap, memory.total [MiB], driver_version"]
    for gpu in host.gpus:
        mib = round(gpu.mem_gb * 1_000_000_000 / (1024 * 1024))
        cap = f"{gpu.arch // 10}.{gpu.arch % 10}"
        rows.append(f"{gpu.index}, {gpu.name}, {gpu.uuid}, {cap}, {mib} MiB, {host.driver_version}")
    return "\n".join(rows) + "\n"


class FakeBackend:
    name = "fake"

    def __init__(self, spec: FakeSpec) -> None:
        self._spec = spec

    @classmethod
    def from_source(cls, source: str) -> FakeBackend:
        return cls(load_spec(source))

    def discover(self) -> HostInfo:
        return self._spec.host

    def smi_text(self) -> str | None:
        return render_smi_csv(self._spec.host) if self._spec.host.gpus else None

    def kernel_log(self) -> list[str] | None:
        if self._spec.kernel_log_lines is not None:
            return list(self._spec.kernel_log_lines)
        path = self._spec.kernel_log_path
        if path is None or not path.exists():
            return None
        with path.open("r", encoding=ENCODING) as handle:
            return handle.read().splitlines()
