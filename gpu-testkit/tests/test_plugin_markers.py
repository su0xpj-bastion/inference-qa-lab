"""The plugin end to end (pytester): markers skip with reasons, leases set CUDA_VISIBLE_DEVICES."""

from __future__ import annotations

import pytest

SUITE = """
import os
import pytest

@pytest.mark.gpus(2, arch=">=sm90", mem_gb=80)
def test_tp2(gpu_devices):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, gpu_devices))
    assert len(gpu_devices) == 2

@pytest.mark.gpus(16)
def test_needs_16():
    pass

@pytest.mark.gpus(1, arch=">=sm100")
def test_blackwell_only():
    pass

@pytest.mark.multinode(nodes=2)
@pytest.mark.gpus(8)
def test_two_nodes():
    pass

@pytest.mark.numerics("fp8")
def test_fp8_kernel():
    pass

def test_plain_cpu_test():
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
"""


def test_markers_skip_with_clear_reasons_on_h100(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SUITE)
    result = pytester.runpytest("--gpu-fake-spec=h100x8", "-rs")
    result.assert_outcomes(passed=3, skipped=3)
    result.stdout.fnmatch_lines(
        [
            "gpu-testkit: fake-h100-node0: 8x NVIDIA H100 80GB HBM3 (sm90, 85.5 GB)*",
            "SKIPPED*gpu-testkit: needs 16 GPU(s) matching ?gpus=16?, host has 8 of 8 qualifying*",
            "SKIPPED*needs 1 GPU(s) matching ?gpus=1, arch>=sm100?, host has 0 of 8 qualifying*",
            "SKIPPED*gpu-testkit: needs 2 nodes, job has 1*",
        ]
    )


def test_fp8_skips_on_ampere(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SUITE)
    result = pytester.runpytest("--gpu-fake-spec=a100x8", "-rs", "-k", "test_fp8_kernel or test_tp2")
    result.assert_outcomes(skipped=2)
    result.stdout.fnmatch_lines(["*arch>=sm90*8 GPUs sm80*", "*dtype=fp8_e4m3(>=sm89)*8 GPUs sm80*"])


def test_everything_gpu_skips_on_a_cpu_host(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SUITE)
    result = pytester.runpytest("--gpu-backend=fake", "--gpu-fake-spec=none", "-rs")
    result.assert_outcomes(passed=1, skipped=5)
    result.stdout.fnmatch_lines(["*no GPUs, backend=fake*"])


def test_multinode_reads_job_size(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    pytester.makepyfile(SUITE)
    pytester.makefile(
        ".json", spec='{"nodes": 2, "gpus": [{"name": "NVIDIA H100", "arch": "sm90", "mem_gb": 80, "count": 8}]}'
    )
    result = pytester.runpytest(f"--gpu-fake-spec={pytester.path / 'spec.json'}", "-k", "two_nodes")
    result.assert_outcomes(passed=1)


def test_invalid_marker_fails_only_that_test(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("""
        import pytest

        @pytest.mark.gpus(0)
        def test_bad():
            pass

        @pytest.mark.gpus(1, arch="hopper")
        def test_bad_arch():
            pass

        def test_fine():
            pass
    """)
    result = pytester.runpytest("--gpu-fake-spec=h100x1")
    result.assert_outcomes(passed=1, errors=2)
    result.stdout.fnmatch_lines(
        [
            "*invalid gpu-testkit marker: gpus(n=0): n must be >= 1*",
            "*invalid gpu-testkit marker: invalid arch spec 'hopper'*",
        ]
    )


def test_gpu_devices_without_marker_is_a_clear_error(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_x(gpu_devices): pass")
    result = pytester.runpytest("--gpu-fake-spec=h100x1")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*gpu_devices fixture needs @pytest.mark.gpus(n)*"])


def test_numerics_fixture_uses_marker_tolerances(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("""
        import pytest

        @pytest.mark.numerics("bf16")
        def test_default_bf16(numerics):
            assert (numerics.dtype, numerics.rtol) == ("bf16", None)
            numerics.assert_close([1.0, 2.0], [1.0, 2.02])

        @pytest.mark.numerics("bf16", rtol=1e-3)
        def test_tightened(numerics):
            numerics.assert_close([1.0, 2.0], [1.0, 2.02])
    """)
    result = pytester.runpytest("--gpu-fake-spec=h100x1")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*assert_close failed (bf16, rtol=0.001*", "*worst at index (1,)*"])


def test_bad_backend_config_is_a_usage_error(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_x(): pass")
    result = pytester.runpytest("--gpu-fake-spec=v100x8")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*gpu-testkit: unknown fake preset 'v100x8'*"])


def test_env_spec_and_ini_defaults(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_TESTKIT_FAKE_SPEC", "l4x2")
    pytester.makeini("[pytest]\ngpu_reruns = 2\n")
    pytester.makepyfile("def test_x(gpu_host): assert gpu_host.gpus[0].name == 'NVIDIA L4'")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["gpu-testkit: reruns=2, quarantine=0 entries*"])
