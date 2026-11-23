"""End-to-end: inject each fault, analyze the bundle, compare with ground truth."""
from __future__ import annotations

import filecmp
import shutil

import pytest

from faultsim.matrix import matches
from faultsim.scenarios import SCENARIOS, run
from rca.engine import THRESHOLD, analyze

FAULT_SCENARIOS = [n for n, s in SCENARIOS.items() if n != "no_fault" and "slow" not in s.tags]


@pytest.mark.parametrize("name", FAULT_SCENARIOS)
def test_rca_names_injected_root_cause(bundles, name):
    """Every injected fault is named (cause + node + GPU) at >= threshold, with no extra root cause."""
    _b, report, truth = bundles(name)
    preds = report["root_causes"]
    assert len(preds) == len(truth["root_causes"]), [(p["cause"], p["node"]) for p in preds]
    for t in truth["root_causes"]:
        hit = [p for p in preds if matches(p, t)]
        assert hit, f"missed {t['cause']} on {t['node']}; got {[(p['cause'], p['node'], p['gpu']) for p in preds]}"
        assert hit[0]["confidence"] >= THRESHOLD
        assert hit[0]["recommended_action"]["action"] == t["action"]
        assert hit[0]["evidence"], "a root cause must cite the lines it used"


def test_no_fault_raises_no_root_cause(bundles):
    """Noise (correctable SBEs, a stray profiler Xid 31, a latency blip) must not produce an incident."""
    _b, report, _t = bundles("no_fault")
    assert report["verdict"] == "healthy"
    assert report["root_causes"] == [] and report["low_confidence"] == []
    assert [p["cause"] for p in report["precursors"]] == ["app_fault"]


@pytest.mark.parametrize("name,node,red_herring_node", [
    ("gpu_fallen_off_bus", "gpu-node-1", "gpu-node-0"),
    ("node_power_loss", "gpu-node-3", "gpu-node-2"),
    ("driver_partial_upgrade", "gpu-node-2", "gpu-node-3"),
    ("ib_link_degraded", "gpu-node-1", "gpu-node-0"),
])
def test_timeouts_on_healthy_ranks_blame_the_silent_node(bundles, name, node, red_herring_node):
    """Healthy ranks that REPORT the NCCL timeout are explained, never blamed; the silent node is."""
    _b, report, _t = bundles(name)
    (rc,) = report["root_causes"]
    assert rc["node"] == node
    if name != "ib_link_degraded":
        assert red_herring_node in rc["explains"]["effects"]["nccl_timeout"]["nodes"]
        assert "nccl" in rc["sources"]
    else:
        assert any(red_herring_node in e["line"] and e["source"] == "nccl" for e in rc["evidence"])
    assert all(e["kind"] != "nccl_timeout" for e in report["unexplained_effects"])


def test_xid79_rank_and_blast_radius(bundles):
    """The report names the GPU's rank and the replica-wide blast radius."""
    _b, report, _t = bundles("gpu_fallen_off_bus")
    (rc,) = report["root_causes"]
    assert (rc["gpu"], rc["rank"], rc["replica"]) == (3, 11, "replica-a")
    br = rc["blast_radius"]
    assert br["ranks"] == 16 and br["nodes"] == ["gpu-node-0", "gpu-node-1"] and br["requests_failed"] > 0


def test_bad_build_is_software_not_six_rmas(bundles):
    """Xid 13/31 across six GPUs right after an image roll is one software root cause, not hardware."""
    _b, report, _t = bundles("bad_engine_build")
    (rc,) = report["root_causes"]
    assert (rc["cause"], rc["layer"], rc["node"]) == ("bad_build", "serving", None)
    assert "Do NOT RMA" in rc["recommended_action"]["detail"]


def test_memory_leak_detected_before_oom(bundles):
    """The FB-used trend names the leaking GPU before the CUDA OOM line appears."""
    _b, report, _t = bundles("worker_memory_leak")
    (rc,) = report["root_causes"]
    oom = [e for e in rc["evidence"] if e["source"] == "engine"]
    assert oom and rc["detected_at_s"] < oom[0]["t_s"]


@pytest.mark.parametrize("name", ["nvlink_degraded", "thermal_throttle"])
def test_stress_only_faults_are_latent_at_idle(bundles, name):
    """At idle the fault exists but has no user impact: no root cause. Under load it is found (above)."""
    _b, report, _t = bundles(name, load="idle")
    assert report["root_causes"] == []


def test_short_run_only_sees_slow_leak_as_precursor(bundles):
    """A 10-minute steady run shows the 0.6 MiB/request leak as a precursor trend only."""
    _b, report, _t = bundles("soak_slow_leak", load="steady", duration=600)
    assert report["root_causes"] == []
    assert [(p["cause"], p["node"], p["gpu"]) for p in report["precursors"]] == [("memory_leak", "gpu-node-2", 4)]


@pytest.mark.slow
def test_soak_exposes_slow_leak(bundles):
    """The 50-minute soak drives the same leak to OOM and the RCA names it."""
    _b, report, _t = bundles("soak_slow_leak")
    assert [(p["cause"], p["node"], p["gpu"]) for p in report["root_causes"]] == [("memory_leak", "gpu-node-2", 4)]


def test_ground_truth_is_never_read(bundles, tmp_path):
    """Deleting ground_truth.json does not change the analysis."""
    bdir, report, _t = bundles("gpu_fallen_off_bus")
    copy = tmp_path / "b"
    shutil.copytree(bdir, copy)
    (copy / "ground_truth.json").unlink()
    assert analyze(copy) == report


def test_bundles_are_deterministic(tmp_path):
    """Same scenario + seed -> byte-identical bundle (so CI failures replay locally)."""
    a, _s, _t = run("ecc_uncorrectable", tmp_path / "a", seed=7)
    b, _s, _t = run("ecc_uncorrectable", tmp_path / "b", seed=7)
    cmp = filecmp.dircmp(a, b)
    files = ["manifest.json", "router/access.log", "k8s/events.json", "nodes/gpu-node-2/kern.log",
             "nodes/gpu-node-2/dcgm.prom"]
    for f in files:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    assert not cmp.diff_files
