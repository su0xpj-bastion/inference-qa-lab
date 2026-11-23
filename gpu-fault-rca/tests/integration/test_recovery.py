"""Recovery-path validation: the recommended action, replayed, must meet the recovery SLOs."""
from __future__ import annotations

import pytest

from faultsim.recovery import recovery_scenarios, validate
from faultsim.sim import Action

EXPECTED_FAIL = {"nvlink_degraded"}     # no N-1 headroom at the ramp peak; see README


@pytest.mark.parametrize("name", [n for n in recovery_scenarios() if n not in EXPECTED_FAIL])
def test_recommended_action_meets_recovery_slo(tmp_path, name):
    """TTD, TTM and requests lost are within slo/recovery.json when the RCA's action is applied."""
    r = validate(name, tmp_path, seed=1)
    assert r["verdict"] == "pass", r["checks"]


def test_nvlink_reshard_at_ramp_peak_has_no_headroom(tmp_path):
    """Detection is in time, but removing half a replica while load is still climbing never gets clean."""
    r = validate("nvlink_degraded", tmp_path, seed=1)
    assert r["checks"]["detected"] and r["checks"]["ttd"]
    assert not r["checks"]["mitigated"] and r["verdict"] == "fail"


def test_acting_on_the_red_herring_does_not_recover(tmp_path):
    """Restarting the worker that REPORTED the NCCL timeout (gpu-node-0) leaves the outage in place."""
    wrong = [Action("RESTART_WORKER", 210, "gpu-node-0", 0)]
    r = validate("gpu_fallen_off_bus", tmp_path, seed=1, actions=wrong)
    assert r["verdict"] == "fail" and not r["checks"]["mitigated"]


def test_no_action_loses_more_requests(tmp_path):
    """Doing nothing is measurably worse than the recommended drain."""
    good = validate("gpu_fallen_off_bus", tmp_path / "g", seed=1)
    none = validate("gpu_fallen_off_bus", tmp_path / "n", seed=1, actions=[])
    assert none["verdict"] == "fail" and none["requests_lost"] > 3 * good["requests_lost"]
