"""Tests that exercise flake handling and quarantine in the example run."""

from __future__ import annotations

import pytest
from simkernels import CommunicatorLost


@pytest.fixture
def first_attempt_marker(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest):
    # Reruns happen on the same worker, so the per-worker basetemp survives between attempts.
    return tmp_path_factory.getbasetemp() / f"attempted-{request.node.name}"


@pytest.mark.gpus(2)
def test_nccl_warmup_flake(first_attempt_marker) -> None:
    """Simulates a communicator dropping on first use: fails once, passes on rerun -> FLAKY."""
    if not first_attempt_marker.exists():
        first_attempt_marker.touch()
        raise CommunicatorLost("ncclCommInitRank: unhandled system error (simulated)")


@pytest.mark.gpus(1)
def test_speculative_decoding_acceptance_rate() -> None:
    """Known product bug (quarantined in quarantine.toml): runs, fails, doesn't gate."""
    acceptance = 0.41
    assert acceptance >= 0.6, f"acceptance rate {acceptance} below 0.6"
