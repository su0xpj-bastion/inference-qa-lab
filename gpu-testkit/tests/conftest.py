from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
ISOLATED_ENV = (
    "GPU_TESTKIT_FAKE_SPEC",
    "GPU_TESTKIT_TODAY",
    "GPU_TESTKIT_NODES",
    "GPU_TESTKIT_RUN_ID",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NNODES",
    "CUDA_VISIBLE_DEVICES",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Each framework test gets its own lock dir and no inherited GPU/job settings."""
    for name in ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPU_TESTKIT_LOCK_DIR", str(tmp_path_factory.mktemp("locks")))
    return


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def junit_cases(path: Path) -> dict[str, dict[str, str]]:
    """{test name: {property: value}} plus the case's result element under '_result'."""
    cases: dict[str, dict[str, str]] = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        props = {p.get("name", ""): p.get("value", "") for p in case.iter("property")}
        results = [child.tag for child in case if child.tag in ("failure", "error", "skipped")]
        props["_result"] = results[0] if results else "passed"
        cases[case.get("name", "")] = props
    return cases
