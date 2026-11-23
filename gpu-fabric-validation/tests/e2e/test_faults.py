from __future__ import annotations

import pytest

from fabval.analyze import analyze
from fabval.model import NodeType
from fabval.sim import Fault, SimCluster

ROCE = "h100-sxm-8x-400g-roce"
IB = "h100-sxm-8x-ndr400-ib"


def run(tmp_path, node_type, faults, nodes=4, seed=0):
    SimCluster(NodeType.load(node_type), nodes, [Fault.parse(f) for f in faults], seed).write(tmp_path)
    return analyze(tmp_path)


@pytest.mark.parametrize("node_type", [ROCE, IB])
@pytest.mark.parametrize("seed", range(5))
def test_healthy_cluster_passes_with_no_findings(tmp_path, node_type, seed):
    """No false positives: a clean cluster yields PASS and zero findings."""
    rep = run(tmp_path, node_type, [], seed=seed)
    assert rep.verdict == "PASS", rep.findings
    assert rep.findings == [] and rep.errors == []


@pytest.mark.parametrize("node_type,fault,location,category", [
    (IB, "downtrain:node-03:2", "node-03/mlx5_2", "link"),
    (ROCE, "downtrain:node-03:2", "node-03/mlx5_2", "link"),
    (IB, "numa:node-02:6", "node-02/mlx5_6", "affinity"),
    (ROCE, "pfc:node-04:4", "node-04/mlx5_4", "lossless"),
    (ROCE, "gid:node-01", "node-01", "gid"),
    (IB, "corrupt:node-04", "node-04", "corruption"),
])
def test_each_fault_detected_and_localized(tmp_path, node_type, fault, location, category):
    """Each injected fault fails the run and is localized to exactly the right node/port."""
    rep = run(tmp_path, node_type, [fault])
    assert rep.exit_code == 1
    assert [(c.location, c.category) for c in rep.root_causes] == [(location, category)]


def test_all_faults_at_once_on_eight_nodes(tmp_path):
    """Five simultaneous faults on different nodes are each localized; nothing extra is blamed."""
    faults = ["downtrain:node-03:2", "pfc:node-05:4", "numa:node-02:6", "gid:node-07", "corrupt:node-04"]
    rep = run(tmp_path, ROCE, faults, nodes=8)
    assert sorted(c.location for c in rep.root_causes) == [
        "node-02/mlx5_6", "node-03/mlx5_2", "node-04", "node-05/mlx5_4", "node-07"]


def test_pfc_passes_bandwidth_gate_but_fails_latency(tmp_path):
    """Lossless misconfig hides from the bandwidth gate; only the small-message latency gate sees it."""
    rep = run(tmp_path, ROCE, ["pfc:node-04:4"])
    pair = [f for f in rep.findings if f.stage == "pairs"]
    assert {f.category for f in pair} == {"latency"}


def test_one_bad_rail_caps_full_collective(tmp_path):
    """A single down-trained HCA drags the whole-cluster all_reduce to about half of expected."""
    rep = run(tmp_path, IB, ["downtrain:node-03:2"])
    assert 40 < rep.stages["collective"]["pct_expected"] < 60


def test_missing_pair_file_is_inconclusive(tmp_path):
    """Incomplete coverage on a clean run is INCONCLUSIVE (exit 2), never PASS."""
    SimCluster(NodeType.load(IB), 3, [], 0).write(tmp_path)
    next((tmp_path / "pairs").glob("*.txt")).unlink()
    assert analyze(tmp_path).exit_code == 2
