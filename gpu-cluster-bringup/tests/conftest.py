from __future__ import annotations

from pathlib import Path

import pytest

from bringup.allreduce import FixtureRunner
from bringup.collect import collect_fleet
from bringup.fleetgen import write_fleet
from bringup.pipeline import load_manifest, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "golden.json"


@pytest.fixture(scope="session")
def manifest() -> dict:
    return load_manifest(MANIFEST)


def _run(tmp: Path, scenario: str, manifest: dict) -> tuple[Path, dict]:
    write_fleet(tmp, scenario)
    return tmp, run_pipeline(collect_fleet(tmp), manifest, FixtureRunner(tmp))


@pytest.fixture(scope="session")
def defects_run(tmp_path_factory, manifest):
    return _run(tmp_path_factory.mktemp("defects"), "defects", manifest)


@pytest.fixture(scope="session")
def healthy_run(tmp_path_factory, manifest):
    return _run(tmp_path_factory.mktemp("healthy"), "healthy", manifest)


@pytest.fixture(scope="session")
def leaf_down_run(tmp_path_factory, manifest):
    return _run(tmp_path_factory.mktemp("leafdown"), "leaf-down", manifest)
