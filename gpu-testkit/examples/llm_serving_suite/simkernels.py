"""CPU stand-ins for the kernels a serving team validates.

These are simulations: they reproduce the *numerics* of the real kernels (inputs rounded
to fp8/bf16, fp32 accumulation, sharded matmul + allreduce) so the suite's assertions
are meaningful, but nothing here touches a GPU.
"""

from __future__ import annotations

import math
import random

from gpu_testkit import quantize

Matrix = list[list[float]]


class CommunicatorLost(RuntimeError):
    """NCCL communicator died (registered as infra in pytest.ini)."""


def random_matrix(rows: int, cols: int, seed: int, scale: float = 1.0) -> Matrix:
    rng = random.Random(seed)
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def cast(m: Matrix, dtype: str) -> Matrix:
    return [[quantize(x, dtype) for x in row] for row in m]


def matmul(a: Matrix, b: Matrix, *, accumulate: str = "fp64") -> Matrix:
    cols = list(zip(*b, strict=True))
    out: Matrix = []
    for row in a:
        values = []
        for col in cols:
            acc = 0.0
            for x, y in zip(row, col, strict=True):
                acc = acc + x * y if accumulate == "fp64" else quantize(acc + x * y, accumulate)
            values.append(acc)
        out.append(values)
    return out


def fp8_gemm(a: Matrix, b: Matrix, scale_a: float, scale_b: float) -> Matrix:
    """Per-tensor scaled fp8 (e4m3) GEMM with fp32 accumulation, like a Hopper fp8 kernel."""
    qa = [[quantize(x / scale_a, "fp8") for x in row] for row in a]
    qb = [[quantize(x / scale_b, "fp8") for x in row] for row in b]
    return [[v * scale_a * scale_b for v in row] for row in matmul(qa, qb, accumulate="fp32")]


def fp8_dequant(m: Matrix, scale: float) -> Matrix:
    """What the kernel actually sees: inputs rounded to e4m3 at this scale, back in float."""
    return [[quantize(x / scale, "fp8") * scale for x in row] for row in m]


def amax_scale(m: Matrix, fmt_max: float = 448.0) -> float:
    return max(abs(x) for row in m for x in row) / fmt_max


def softmax(row: list[float], dtype: str = "fp64") -> list[float]:
    peak = max(row)
    exps = [math.exp(x - peak) for x in row]
    total = sum(exps)
    return [quantize(e / total, dtype) if dtype != "fp64" else e / total for e in exps]


def rmsnorm(row: list[float], eps: float = 1e-6, dtype: str = "fp64") -> list[float]:
    rms = math.sqrt(sum(x * x for x in row) / len(row) + eps)
    return [quantize(x / rms, dtype) if dtype != "fp64" else x / rms for x in row]


def tensor_parallel_matmul(a: Matrix, b: Matrix, degree: int, dtype: str) -> Matrix:
    """Row-parallel linear: shard the K dimension across ranks, each rank computes a partial
    product in ``dtype``, then an allreduce sums the partials."""
    k = len(b)
    if k % degree:
        raise ValueError(f"K={k} not divisible by TP degree {degree}")
    shard = k // degree
    partials = []
    for rank in range(degree):
        a_r = [row[rank * shard : (rank + 1) * shard] for row in a]
        b_r = b[rank * shard : (rank + 1) * shard]
        partials.append(cast(matmul(a_r, b_r), dtype))
    return [[quantize(sum(p[i][j] for p in partials), dtype) for j in range(len(b[0]))] for i in range(len(a))]
