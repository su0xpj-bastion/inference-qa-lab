from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from gpu_testkit import DiagnosticContext

sys.path.insert(0, str(Path(__file__).parent))

ENGINE_CONFIG = {"engine": "sim-serve", "max_batch": 64, "kv_cache_dtype": "fp8_e4m3", "tp": 8}


class EngineConfigCollector:
    """Team-specific diagnostics: the serving engine config travels with every failure bundle."""

    name = "engine"

    def collect(self, ctx: DiagnosticContext) -> Mapping[str, str]:
        return {"engine-config.json": json.dumps(ENGINE_CONFIG | {"devices": list(ctx.devices)}, indent=2)}


@pytest.hookimpl
def pytest_gpu_testkit_diagnostic_collectors(config: pytest.Config) -> list[EngineConfigCollector]:
    return [EngineConfigCollector()]


@pytest.fixture(scope="session")
def engine_config() -> dict[str, object]:
    return dict(ENGINE_CONFIG)
