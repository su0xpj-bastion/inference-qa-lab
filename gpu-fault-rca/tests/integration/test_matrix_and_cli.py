"""Precision/recall gate over the scenario library, and the two CLIs."""
from __future__ import annotations

import json

from faultsim import cli as chaos_cli
from faultsim.matrix import evaluate
from rca import cli as rca_cli


def test_precision_recall_over_library(tmp_path):
    """Two seeds of every scenario (plus idle runs of stress-only faults): precision and recall >= 0.95."""
    res = evaluate(tmp_path, seeds=(11, 12))
    assert res["precision"] >= 0.95 and res["recall"] >= 0.95, res["rows"]


def test_chaos_run_then_rca_analyze(tmp_path, capsys):
    """`chaos run` writes a bundle; `rca analyze` writes incident.md + incident.json and exits 1 (incident)."""
    bdir = tmp_path / "bundle"
    assert chaos_cli.main(["run", "gpu_fallen_off_bus", "--out", str(bdir), "--seed", "3"]) == 0
    out = tmp_path / "report"
    assert rca_cli.main(["analyze", str(bdir), "--out", str(out)]) == 1
    md = (out / "incident.md").read_text(encoding="utf-8")
    with (out / "incident.json").open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert "`gpu_lost` on gpu-node-1 GPU 3 (rank 11 of replica-a)" in md
    assert data["root_causes"][0]["recommended_action"]["action"] == "DRAIN_NODE"
    assert rca_cli.main(["analyze", str(_healthy(tmp_path))]) == 0
    capsys.readouterr()


def _healthy(tmp_path):
    bdir = tmp_path / "healthy"
    chaos_cli.main(["run", "no_fault", "--out", str(bdir)])
    return bdir
