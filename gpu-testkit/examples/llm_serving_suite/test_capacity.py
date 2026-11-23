"""Capacity checks that only make sense on the memory class they were written for."""

from __future__ import annotations

import pytest

LLAMA_70B = {"layers": 80, "kv_heads": 8, "head_dim": 128, "weights_gb_fp8": 70.6}


def kv_cache_gb(tokens: int, bytes_per_elem: int) -> float:
    m = LLAMA_70B
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * bytes_per_elem * tokens / 1e9


@pytest.mark.gpus(8, mem_gb=80)
def test_70b_fp8_serves_64_x_8k_context_across_tp8(gpu_host, gpu_devices, engine_config) -> None:
    per_gpu = {g.index: g.mem_gb for g in gpu_host.gpus}
    budget = sum(per_gpu[i] for i in gpu_devices) * 0.9  # 10% for activations and CUDA graphs
    need = LLAMA_70B["weights_gb_fp8"] + kv_cache_gb(64 * 8192, bytes_per_elem=1)
    assert engine_config["kv_cache_dtype"] == "fp8_e4m3"
    assert need < budget, f"needs {need:.1f} GB, budget {budget:.1f} GB"


@pytest.mark.gpus(1, mem_gb=140)
def test_70b_fits_on_one_h200() -> None:
    assert LLAMA_70B["weights_gb_fp8"] + kv_cache_gb(8192, 1) < 140
