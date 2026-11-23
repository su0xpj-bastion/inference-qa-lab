"""Signal math and catalog invariants."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from rca import catalog
from rca.detectors import _fit
from rca.symptoms import detect, p99
from rca.parsers import Request

ROOT = Path(__file__).resolve().parents[2]


def test_fit_slope_and_r2():
    """Linear trend is recovered; flat noise has low R2."""
    slope, r2 = _fit([0, 15, 30, 45], [100, 145, 190, 235])
    assert abs(slope - 3.0) < 1e-9 and r2 > 0.999
    _s, r2_flat = _fit([0, 15, 30, 45], [100, 102, 99, 101])
    assert r2_flat < 0.5


def test_p99_and_symptom_windows():
    """One slow window is a blip; two consecutive are a latency symptom; 499s never count."""
    reqs = [Request(1000 + i * 0.1, 200, 0.5, "replica-a") for i in range(300)]
    reqs += [Request(1030 + i * 0.1, 200, 9.0, "replica-a") for i in range(50)]
    reqs += [Request(1000 + i, 499, 99.0, "replica-a") for i in range(30)]
    assert detect(reqs, 1000, 3.0, 0.01) == []
    reqs += [Request(1040 + i * 0.1, 200, 9.0, "replica-a") for i in range(50)]
    syms = detect(reqs, 1000, 3.0, 0.01)
    assert [(s.kind, s.start - 1000, s.end - 1000) for s in syms] == [("latency_slo", 30, 50)]
    assert p99([1.0] * 99 + [50.0]) == 1.0


def test_every_cause_has_layer_base_edges_and_action():
    """A cause kind without an action or causal edges would silently never be reported."""
    for kind in catalog.CAUSE_LAYER:
        assert kind in catalog.BASE_CONFIDENCE and kind in catalog.CAN_CAUSE and kind in catalog.ACTIONS


def test_signal_cheatsheet_covers_catalog():
    """docs/SIGNALS.md lists every Xid and throttle bit the engine knows."""
    text = (ROOT / "docs" / "SIGNALS.md").read_text(encoding="utf-8")
    for code in catalog.XID:
        assert re.search(rf"^\| {code} \|", text, re.M), f"Xid {code} missing from docs/SIGNALS.md"
    for bit in catalog.THROTTLE_BITS:
        assert f"0x{bit:x}" in text


def test_rca_never_imports_the_simulator():
    """The analyzer must not share code with the thing that generated the telemetry."""
    for py in (ROOT / "rca").glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(n.startswith("faultsim") for n in names), py.name


def test_future_annotations_first_in_every_module():
    """House rule: `from __future__ import annotations` in every module."""
    for py in list((ROOT / "rca").glob("*.py")) + list((ROOT / "faultsim").glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        assert any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body), py.name
