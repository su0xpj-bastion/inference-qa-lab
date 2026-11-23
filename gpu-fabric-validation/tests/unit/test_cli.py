from __future__ import annotations

import json

from fabval.cli import main
from tests.conftest import ENCODING, FIXTURES


def test_expected(capsys):
    assert main(["expected", "--node-type", "h100-sxm-8x-ndr400-ib", "--collective", "sendrecv",
                 "--nodes", "2", "--ranks-per-node", "1"]) == 0
    assert capsys.readouterr().out.strip() == "47.0"


def test_parse_nccl_tests(capsys):
    assert main(["parse", "nccl-tests", str(FIXTURES / "nccl_tests/sendrecv_perf.pair.healthy.txt")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["collective"] == "sendrecv" and out["nranks"] == 2 and out["consistency_problems"] == []


def test_simulate_analyze_exit_codes(tmp_path, capsys):
    """analyze exits 0 on a clean run, 1 on a fault, 2 with no run directory; writes md + json."""
    clean, bad = tmp_path / "clean", tmp_path / "bad"
    assert main(["simulate", "--nodes", "3", "--out", str(clean)]) == 0
    assert main(["analyze", str(clean), "--quiet"]) == 0
    assert main(["simulate", "--nodes", "3", "--fault", "numa:node-02:5", "--out", str(bad)]) == 0
    md, js = tmp_path / "r.md", tmp_path / "r.json"
    assert main(["analyze", str(bad), "--quiet", "--md", str(md), "--json", str(js)]) == 1
    with js.open("r", encoding=ENCODING) as fh:
        rep = json.load(fh)
    assert rep["verdict"] == "FAIL" and rep["root_causes"][0]["location"] == "node-02/mlx5_5"
    with md.open("r", encoding=ENCODING) as fh:
        assert "### Root causes" in fh.read()
    assert main(["analyze", str(tmp_path / "nope"), "--quiet"]) == 2
