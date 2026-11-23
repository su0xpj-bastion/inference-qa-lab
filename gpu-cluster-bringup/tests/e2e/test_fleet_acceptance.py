from __future__ import annotations

import json

import pytest

from bringup import fleetgen
from bringup.allreduce import FixtureRunner
from bringup.cli import main
from bringup.collect import collect_fleet
from bringup.pipeline import DEGRADED, REJECTED, run_pipeline

from tests.conftest import MANIFEST

INJECTED = sorted(fleetgen.DEFAULT_INJECTIONS.items())


def _verdict(report: dict, host: str) -> dict:
    return next(n for n in report["nodes"] if n["hostname"] == host)


@pytest.mark.parametrize("host,defect", INJECTED, ids=[f"{d}@{h}" for h, d in INJECTED])
def test_each_defect_caught_at_the_right_stage_for_the_right_reason(defects_run, host, defect):
    """Each injected defect cordons its node at the stage that owns it, by that check alone.

    'Alone' matters: a defect that trips three unrelated checks produces three
    tickets and a technician who fixes the wrong thing first. Declared
    downstream consequences are allowed, and must come after the root cause.
    """
    _, report = defects_run
    stage, check = fleetgen.DEFECTS[defect]
    v = _verdict(report, host)
    assert v["status"] == "cordoned", v
    assert v["cordon_stage"] == stage
    assert v["failed_checks"] == [check, *fleetgen.CONSEQUENCES.get(defect, [])]
    assert v["reasons"] and v["reasons"][0].startswith(f"{check}: ")


def test_cordoned_nodes_do_not_enter_later_stages(defects_run):
    """Stage order is real: a node cordoned at 'node' never ran burn-in."""
    _, report = defects_run
    order = ["node", "burnin", "domain", "cluster"]
    for n in report["nodes"]:
        if n["status"] == "cordoned":
            assert n["stages_run"] == order[: order.index(n["cordon_stage"]) + 1], n["hostname"]
            assert {c["stage"] for c in n["checks"]} <= set(n["stages_run"])


def test_only_injected_nodes_are_cordoned(defects_run):
    """No collateral: every other node is accepted, and none is silently missing."""
    root, report = defects_run
    cordoned = {c["hostname"] for c in report["cordon"]}
    assert cordoned == set(fleetgen.DEFAULT_INJECTIONS)
    on_disk = {p.name for p in root.iterdir() if (p / "meta.json").exists()}
    assert {n["hostname"] for n in report["nodes"]} == on_disk


def test_defects_fleet_is_degraded_not_rejected(defects_run):
    """Losing 11 of 32 H100 nodes still leaves the 20 the first job needs, so the cluster is usable."""
    _, report = defects_run
    assert report["decision"] == DEGRADED and report["exit_code"] == 3
    assert report["pools"]["hgx-h100-ndr"] == {"total": 32, "accepted": 21, "required": 20, "gpus_accepted": 168}
    assert report["pools"]["mi300x-ndr"]["accepted"] == 3
    assert report["summary"]["cordoned_by_stage"] == {"node": 7, "burnin": 3, "domain": 1, "cluster": 1}
    assert all(c["status"] == "pass" for c in report["cluster_checks"])
    assert report["rack_findings"] == []  # scattered single-node defects are not a rack problem


def test_healthy_fleet_accepted(healthy_run):
    _, report = healthy_run
    assert report["decision"] == "ACCEPTED" and report["exit_code"] == 0
    assert report["summary"]["cordoned"] == 0
    for n in report["nodes"]:
        assert n["stages_run"] == ["node", "burnin", "domain", "cluster"]
        assert not [c for c in n["checks"] if c["status"] == "fail"]


def test_amd_nodes_run_the_same_gates(healthy_run):
    """Vendor-neutral: AMD nodes go through the same checks; DCGM is an explicit SKIP, not a silent pass."""
    _, report = healthy_run
    nv = {c["name"] for c in _verdict(report, "gpu-r01-n01")["checks"]}
    amd = _verdict(report, "amd-r05-n01")
    assert {c["name"] for c in amd["checks"]} == nv
    dcgm = next(c for c in amd["checks"] if c["name"] == "dcgm_diag")
    assert dcgm["status"] == "skip" and "RVS" in dcgm["reason"]


def test_leaf_switch_failure_reported_as_one_rack_finding(leaf_down_run):
    """Five nodes losing the same port in one rack is one leaf-switch ticket, not five RMAs."""
    _, report = leaf_down_run
    assert report["decision"] == DEGRADED
    [finding] = report["rack_findings"]
    assert finding["rack"] == "r03" and finding["check"] == "fabric_ports"
    assert len(finding["nodes"]) == 5 and "leaf switch" in finding["reason"]


def test_capacity_shortfall_rejects_the_cluster(tmp_path, manifest):
    """Cordoning is not free: below the pool minimum, the cluster is REJECTED (exit 1)."""
    nodes = fleetgen.build_fleet("healthy")
    for n in nodes[:13]:  # 32 - 13 = 19 < 20
        fleetgen.inject(n, "driver_drift")
    for n in nodes:
        fleetgen.write_node(n, tmp_path)
    fleetgen.write_cluster_results(tmp_path)
    report = run_pipeline(collect_fleet(tmp_path), manifest, FixtureRunner(tmp_path))
    assert report["decision"] == REJECTED and report["exit_code"] == 1
    cap = next(c for c in report["cluster_checks"] if c["name"] == "capacity[hgx-h100-ndr]")
    assert cap["status"] == "fail" and "19 accepted nodes < required 20" in cap["reason"]


def test_slow_multinode_allreduce_rejects(tmp_path, manifest):
    """Every node can pass and the fabric still be wrong (e.g. adaptive routing off): the multi-node gate catches it."""
    fleetgen.write_fleet(tmp_path, "healthy")
    with (tmp_path / "_cluster" / "all_reduce_perf.hgx-h100-ndr.txt").open("w", encoding="utf-8") as fh:
        fh.write(fleetgen.render_all_reduce(210.0, 256, "pool"))
    report = run_pipeline(collect_fleet(tmp_path), manifest, FixtureRunner(tmp_path))
    ar = next(c for c in report["cluster_checks"] if c["name"] == "allreduce_multi_node[hgx-h100-ndr]")
    assert ar["status"] == "fail" and "< 340" in ar["reason"]
    assert report["decision"] == REJECTED


def test_missing_evidence_fails_closed(tmp_path, manifest):
    """A node whose dcgmi capture is missing is cordoned for missing evidence, never passed."""
    nodes = fleetgen.build_fleet("healthy")
    node_dir = fleetgen.write_node(nodes[0], tmp_path)
    (node_dir / "dcgmi-diag-r3.txt").unlink()
    with (node_dir / "ibstat.txt").open("w", encoding="utf-8") as fh:
        fh.write("ibstat: command not found\n")
    report = run_pipeline(collect_fleet(tmp_path), manifest, FixtureRunner(tmp_path))
    v = report["nodes"][0]
    assert v["cordon_stage"] == "node" and "evidence_complete" in v["failed_checks"]
    reason = " ".join(v["reasons"])
    assert "dcgmi-diag-r3.txt" in reason and "ibstat.txt (unparseable" in reason


def test_unknown_node_type_is_cordoned_not_skipped(tmp_path, manifest):
    node = fleetgen.build_fleet("healthy")[0]
    node.node_type = "hgx-h200-ndr"
    fleetgen.write_node(node, tmp_path)
    report = run_pipeline(collect_fleet(tmp_path), manifest, FixtureRunner(tmp_path))
    assert report["nodes"][0]["failed_checks"] == ["node_type_known"]


def test_generator_is_deterministic(tmp_path):
    """Same seed, byte-identical captures: reports diff cleanly between runs."""
    a, b = tmp_path / "a", tmp_path / "b"
    fleetgen.write_fleet(a)
    fleetgen.write_fleet(b)
    for f in sorted(a.rglob("*")):
        if f.is_file():
            assert f.read_bytes() == (b / f.relative_to(a)).read_bytes(), f


# -------------------------------------------------------------------- CLI


def test_cli_run_writes_reports_and_exit_code(tmp_path, capsys):
    fleet, out = tmp_path / "fleet", tmp_path / "out"
    assert main(["gen-fleet", "--out", str(fleet)]) == 0
    assert main(["run", "--fleet", str(fleet), "--manifest", str(MANIFEST), "--out", str(out)]) == 3
    with (out / "bringup.json").open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["schema"] == "bringup-report/v1" and doc["decision"] == DEGRADED
    md = (out / "bringup.md").read_text(encoding="utf-8")
    assert "ACCEPTED_DEGRADED" in md and "`gpu-r01-n02`" in md
    assert 'scontrol update NodeName=gpu-r02-n03 State=DRAIN Reason="bringup:node:fabric_ports"' in md
    assert "CORDON gpu-r04-n05 [burnin]" in capsys.readouterr().out


def test_cli_node_hook_exit_codes(tmp_path):
    """The provisioning hook: exit 0 on a good node, 1 on a bad one, with a JSON verdict."""
    good, bad = fleetgen.build_fleet("healthy")[:2]
    fleetgen.inject(bad, "pcie_downtrain")
    gdir, bdir = fleetgen.write_node(good, tmp_path), fleetgen.write_node(bad, tmp_path)
    assert main(["node", "--captures", str(gdir), "--manifest", str(MANIFEST)]) == 0
    out = tmp_path / "v.json"
    assert main(["node", "--captures", str(bdir), "--manifest", str(MANIFEST), "--json", str(out)]) == 1
    with out.open("r", encoding="utf-8") as fh:
        v = json.load(fh)
    assert v["passed"] is False and v["failed"][0]["name"] == "pcie_link"

