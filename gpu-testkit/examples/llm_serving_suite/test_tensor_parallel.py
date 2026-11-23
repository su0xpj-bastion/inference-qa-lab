"""Tensor-parallel layers must match the single-device result, and get exclusive devices."""

from __future__ import annotations

import os

import pytest
from simkernels import matmul, random_matrix, tensor_parallel_matmul

from gpu_testkit import assert_close


@pytest.mark.parametrize(
    "degree",
    [pytest.param(d, marks=pytest.mark.gpus(d, arch=">=sm80", mem_gb=40), id=f"tp{d}") for d in (2, 4, 8)],
)
def test_row_parallel_linear_matches_single_device(degree: int, gpu_devices: tuple[int, ...]) -> None:
    assert len(gpu_devices) == degree
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, gpu_devices))
    a, b = random_matrix(4, 64, seed=10), random_matrix(64, 8, seed=11)
    # Each rank rounds its partial sum (magnitude up to ~8) to bf16 before the allreduce, so the
    # absolute error scales with the partials, not the (possibly near-zero) final value.
    assert_close(tensor_parallel_matmul(a, b, degree, "bf16"), matmul(a, b), dtype="bf16", atol=6e-2)


@pytest.mark.gpus(2)
def test_tp_rejects_uneven_shards() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        tensor_parallel_matmul(random_matrix(2, 6, 1), random_matrix(6, 2, 2), degree=4, dtype="bf16")


@pytest.mark.multinode(nodes=2)
@pytest.mark.gpus(8)
def test_cross_node_allreduce_bandwidth() -> None:
    pytest.fail("needs a 2-node job: skipped with a reason on a single node")
