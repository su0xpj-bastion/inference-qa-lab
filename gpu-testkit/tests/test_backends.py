"""Backends: fake specs, the nvidia-smi parser against captured output, NVML mapping, selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_testkit.backends import BackendUnavailable, FakeBackend, NullBackend, NvmlBackend, select_backend
from gpu_testkit.backends import nvml as nvml_mod
from gpu_testkit.backends.fake import load_spec, render_smi_csv
from gpu_testkit.backends.smi import parse_cuda_version, parse_query_csv


def test_preset_matches_real_part_numbers() -> None:
    host = load_spec("h100x8").host
    assert len(host.gpus) == 8
    assert {g.arch for g in host.gpus} == {90}
    assert host.gpus[3].index == 3
    assert host.driver_version
    assert host.cuda_version


def test_unknown_preset_lists_known_ones() -> None:
    with pytest.raises(ValueError, match=r"known: none, h100xN"):
        load_spec("v100x8")


def test_file_spec_resolves_kernel_log_relative_to_spec(tmp_path: Path) -> None:
    (tmp_path / "dmesg.log").write_text("line one\nline two\n", encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "hostname": "n0",
                "nodes": 2,
                "kernel_log": "dmesg.log",
                "gpus": [{"name": "X", "arch": "sm90", "mem_gb": 80, "count": 2}],
            }
        ),
        encoding="utf-8",
    )
    backend = FakeBackend.from_source(str(spec_path))
    assert backend.discover().nodes == 2
    assert backend.kernel_log() == ["line one", "line two"]


def test_fake_gpu_arch_must_be_concrete() -> None:
    with pytest.raises(ValueError, match="plain value"):
        load_spec('{"gpus": [{"name": "X", "arch": ">=sm90", "mem_gb": 80}]}')


def test_smi_parser_reads_noheader_nounits(fixtures_dir: Path) -> None:
    gpus, driver = parse_query_csv((fixtures_dir / "nvidia-smi-query-noheader.csv").read_text(encoding="utf-8"))
    assert driver == "560.35.03"
    assert [g.arch for g in gpus] == [90, 90, 90, 90]
    assert gpus[2].uuid.startswith("GPU-6f1a2b3c-0002")
    assert gpus[0].mem_gb == pytest.approx(85.5)


def test_smi_parser_reads_default_csv_with_units(fixtures_dir: Path) -> None:
    gpus, driver = parse_query_csv((fixtures_dir / "nvidia-smi-query-header.csv").read_text(encoding="utf-8"))
    assert driver == "535.183.01"
    assert [(g.name, g.arch) for g in gpus] == [("NVIDIA A100-SXM4-40GB", 80), ("NVIDIA L4", 89)]


def test_smi_parser_rejects_unexpected_columns() -> None:
    with pytest.raises(ValueError, match="unexpected nvidia-smi row"):
        parse_query_csv("0, NVIDIA H100, 9.0\n")


def test_cuda_version_from_banner(fixtures_dir: Path) -> None:
    assert parse_cuda_version((fixtures_dir / "nvidia-smi-banner.txt").read_text(encoding="utf-8")) == "12.6"
    assert parse_cuda_version("no banner here") is None


def test_fake_renders_what_the_real_parser_reads() -> None:
    """The fake backend's nvidia-smi output must round-trip through the real-format parser."""
    host = load_spec("l4x2").host
    gpus, driver = parse_query_csv(render_smi_csv(host))
    assert gpus == list(host.gpus)
    assert driver == host.driver_version


def _stub_pynvml(init_error: bool = False) -> SimpleNamespace:
    def init() -> None:
        if init_error:
            raise RuntimeError("NVML Shared Library Not Found")

    return SimpleNamespace(
        nvmlInit=init,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda h: b"NVIDIA H100 80GB HBM3",  # older pynvml returns bytes
        nvmlDeviceGetCudaComputeCapability=lambda h: (9, 0),
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=85_520_809_984),
        nvmlDeviceGetUUID=lambda h: f"GPU-{h:04d}",
        nvmlSystemGetDriverVersion=lambda: "560.35.03",
        nvmlSystemGetCudaDriverVersion=lambda: 12060,
    )


def test_nvml_backend_maps_nvml_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _stub_pynvml())
    pynvml = nvml_mod.load_pynvml()
    assert pynvml is not None
    host = NvmlBackend(pynvml).discover()
    assert host.backend == "nvml"
    assert host.cuda_version == "12.6"
    assert [(g.name, g.arch, g.mem_gb, g.uuid) for g in host.gpus] == [
        ("NVIDIA H100 80GB HBM3", 90, 85.5, "GPU-0000"),
        ("NVIDIA H100 80GB HBM3", 90, 85.5, "GPU-0001"),
    ]


def test_nvml_init_failure_means_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _stub_pynvml(init_error=True))
    assert nvml_mod.load_pynvml() is None


@pytest.fixture
def no_real_gpu_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError
    monkeypatch.setattr("gpu_testkit.backends.shutil.which", lambda name: None)


@pytest.mark.usefixtures("no_real_gpu_stack")
def test_auto_prefers_fake_spec_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_TESTKIT_FAKE_SPEC", "h100x2")
    assert select_backend("auto").name == "fake"


@pytest.mark.usefixtures("no_real_gpu_stack")
def test_auto_degrades_to_null_without_gpu_stack() -> None:
    assert isinstance(select_backend("auto"), NullBackend)


@pytest.mark.usefixtures("no_real_gpu_stack")
@pytest.mark.parametrize("choice", ["nvml", "smi", "fake"])
def test_explicit_backend_never_degrades_silently(choice: str) -> None:
    with pytest.raises(BackendUnavailable):
        select_backend(choice)
