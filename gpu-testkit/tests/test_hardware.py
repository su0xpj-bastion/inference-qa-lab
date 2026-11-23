"""Marker semantics: what qualifies a host, and the skip reason when it doesn't."""

from __future__ import annotations

import pytest

from gpu_testkit.backends.fake import load_spec
from gpu_testkit.hardware import ArchSpec, Requirement, parse_arch_specs


@pytest.mark.parametrize(
    ("text", "op", "arch"),
    [("sm90", "==", 90), (">=sm90", ">=", 90), ("<sm100", "<", 100), (" != SM_80 ", "!=", 80), ("sm_89", "==", 89)],
)
def test_arch_spec_parses_operators(text: str, op: str, arch: int) -> None:
    assert ArchSpec.parse(text) == ArchSpec(op=op, arch=arch)


@pytest.mark.parametrize("text", ["hopper", ">=90", "sm9", "=>sm90", ""])
def test_arch_spec_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError, match="invalid arch spec"):
        ArchSpec.parse(text)


def test_arch_range_is_a_conjunction() -> None:
    specs = parse_arch_specs(">=sm80,<sm100")
    req = Requirement(gpus=1, arch=specs)
    h100, b200, t4 = (load_spec(p).host for p in ("h100x1", "b200x1", "t4x1"))
    assert req.unmet_reason(h100) is None
    assert req.unmet_reason(b200) is not None
    assert req.unmet_reason(t4) is not None


def test_mem_gb_is_decimal_so_80gb_parts_qualify_for_80() -> None:
    """An H100 80GB reports 81559 MiB = 79.6 GiB. Comparing in GiB would skip mem_gb=80
    tests on the exact card they were written for."""
    host = load_spec("h100x1").host
    assert host.gpus[0].mem_gb == pytest.approx(85.5)
    assert Requirement(gpus=1, mem_gb=80).unmet_reason(host) is None


def test_skip_reason_names_requirement_and_host() -> None:
    host = load_spec("a100x4").host
    reason = Requirement(gpus=2, arch=parse_arch_specs(">=sm90")).unmet_reason(host)
    assert reason is not None
    assert "needs 2 GPU(s) matching [gpus=2, arch>=sm90]" in reason
    assert "host has 0 of 4 qualifying" in reason
    assert "4 GPUs sm80" in reason


def test_dtype_support_gates_on_arch() -> None:
    fp8 = Requirement(dtypes=("fp8_e4m3",))
    assert fp8.unmet_reason(load_spec("l4x1").host) is None  # Ada sm89 has fp8
    reason = fp8.unmet_reason(load_spec("a100x8").host)
    assert reason is not None
    assert "dtype=fp8_e4m3(>=sm89)" in reason


def test_multinode_checks_job_size_first() -> None:
    host = load_spec("h100x8").host
    reason = Requirement(gpus=8, nodes=2).unmet_reason(host)
    assert reason is not None
    assert reason.startswith("needs 2 nodes, job has 1")


def test_cpu_only_host_reason_says_no_gpus() -> None:
    reason = Requirement(gpus=1).unmet_reason(load_spec("none").host)
    assert reason is not None
    assert "no GPUs" in reason


def test_mixed_host_counts_only_qualifying_devices() -> None:
    spec = load_spec(
        '{"gpus": [{"name": "NVIDIA H100 80GB HBM3", "arch": "sm90", "mem_mib": 81559, "count": 2},'
        ' {"name": "NVIDIA L4", "arch": "sm89", "mem_mib": 23034, "count": 2}]}'
    )
    req = Requirement(gpus=2, mem_gb=40)
    assert [g.index for g in req.qualifying_devices(spec.host)] == [0, 1]
    assert Requirement(gpus=3, mem_gb=40).unmet_reason(spec.host) is not None
