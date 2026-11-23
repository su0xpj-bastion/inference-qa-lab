from __future__ import annotations

from pathlib import Path

import pytest

from faultsim.scenarios import run
from rca.engine import analyze

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def bundles(tmp_path_factory):
    """Session cache: (scenario, seed, load) -> (bundle dir, report, truth)."""
    root = tmp_path_factory.mktemp("bundles")
    cache: dict = {}

    def get(name: str, seed: int = 1, load: str | None = None, duration: int | None = None):
        key = (name, seed, load, duration)
        if key not in cache:
            bdir, _sim, truth = run(name, root / f"{name}-{seed}-{load}-{duration}", seed=seed, load=load,
                                    duration=duration)
            cache[key] = (bdir, analyze(bdir), truth)
        return cache[key]
    return get
