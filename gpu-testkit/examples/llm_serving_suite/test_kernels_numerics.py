"""Kernel numerics against an fp64 reference, with tolerances set by dtype."""

from __future__ import annotations

import pytest
from simkernels import amax_scale, fp8_dequant, fp8_gemm, matmul, random_matrix, rmsnorm, softmax

from gpu_testkit import assert_close, ulp_distance


@pytest.mark.gpus(1, arch=">=sm90")
@pytest.mark.numerics("fp8", rtol=1e-5, atol=1e-5)
@pytest.mark.parametrize("shape", [(16, 32, 16), (8, 64, 8)], ids=lambda s: "x".join(map(str, s)))
def test_fp8_gemm_accumulation_matches_fp64(shape: tuple[int, int, int], numerics) -> None:
    """Isolate the kernel from input quantization: the reference is an fp64 GEMM of the *same*
    fp8-rounded inputs, so only fp32 accumulation error remains and the tolerance can be tight."""
    m, k, n = shape
    a, b = random_matrix(m, k, seed=1), random_matrix(k, n, seed=2)
    sa, sb = amax_scale(a), amax_scale(b)
    numerics.assert_close(fp8_gemm(a, b, sa, sb), matmul(fp8_dequant(a, sa), fp8_dequant(b, sb)))


@pytest.mark.gpus(1, arch=">=sm90")
@pytest.mark.numerics("fp8")
def test_fp8_gemm_end_to_end_error_budget() -> None:
    """End to end, per-tensor e4m3 scaling costs ~6% per element, growing ~sqrt(K) over the
    reduction. Budget: max error <= 12% of the output's max magnitude."""
    a, b = random_matrix(16, 32, seed=1), random_matrix(32, 16, seed=2)
    out, ref = fp8_gemm(a, b, amax_scale(a), amax_scale(b)), matmul(a, b)
    peak = max(abs(x) for row in ref for x in row)
    assert_close(out, ref, dtype="fp8", rtol=0.0, atol=0.12 * peak)


@pytest.mark.gpus(1)
@pytest.mark.numerics("bf16")
def test_bf16_softmax_rows_sum_to_one(numerics) -> None:
    row = [x / 4 for x in random_matrix(1, 128, seed=3)[0]]
    numerics.assert_close(softmax(row, "bf16"), softmax(row))
    assert abs(sum(softmax(row, "bf16")) - 1.0) < 1e-2


@pytest.mark.gpus(1)
@pytest.mark.numerics("fp16")
def test_fp16_rmsnorm_within_half_ulp_budget() -> None:
    row = random_matrix(1, 256, seed=4)[0]
    got, want = rmsnorm(row, dtype="fp16"), rmsnorm(row)
    assert_close(got, want, dtype="fp16")
    assert max(ulp_distance(g, w, "fp16") for g, w in zip(got, want, strict=True)) <= 1


@pytest.mark.gpus(1, arch=">=sm100")
def test_nvfp4_gemm_blackwell_only() -> None:
    pytest.fail("never runs on Hopper: the marker skips it with a reason")
