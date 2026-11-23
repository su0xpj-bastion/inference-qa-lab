from __future__ import annotations

import pytest

from ies.cli import load_slo
from ies.client import Budget, make_client
from ies.core import Ctx
from mockserve.server import API_KEY, serve


@pytest.fixture
def mock():
    srv, state = serve(0)
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()


@pytest.fixture
def make_ctx(mock, tmp_path):
    base_url, state = mock

    def _make(faults: set[str] = frozenset(), baseline_dir: str = "baselines") -> Ctx:
        state.faults = set(faults)
        return Ctx(client=make_client(API_KEY, base_url), model="qwen-3.8-27b", target="mock",
                   slo=load_slo("mock"), budget=Budget(500),
                   bad_key_client=lambda: make_client("wrong", base_url),
                   base_params={"reasoning_effort": "none"}, baseline_dir=baseline_dir,
                   api_key=API_KEY, base_url=base_url)
    return _make
