"""Scenario matrix: inject every fault over several seeds, score the RCA against ground truth.

A prediction is a root cause at confidence >= threshold. It matches a truth when
cause and node agree, and the GPU agrees whenever the truth names one.
Stress-only faults are also run at idle load, where the correct answer is
"no root cause" (any prediction there is a false positive).
"""
from __future__ import annotations

import json
from pathlib import Path

from faultsim.scenarios import SCENARIOS, run
from rca.engine import THRESHOLD, analyze

ENCODING = "utf-8"
DEFAULT_SEEDS = (1, 2, 3)


def matches(pred: dict, truth: dict) -> bool:
    if pred["cause"] != truth["cause"] or pred["node"] != truth["node"]:
        return False
    if truth.get("replica") and pred.get("replica") != truth["replica"]:
        return False
    return truth["gpu"] is None or pred["gpu"] == truth["gpu"]


def score(report: dict, truths: list[dict]) -> dict:
    preds = report["root_causes"]
    used: set[int] = set()
    tp = 0
    for t in truths:
        for i, p in enumerate(preds):
            if i not in used and matches(p, t):
                used.add(i)
                tp += 1
                break
    return {"tp": tp, "fp": len(preds) - len(used), "fn": len(truths) - tp,
            "predicted": [(p["cause"], p["node"], p["gpu"], p["confidence"]) for p in preds]}


def cases(seeds: tuple[int, ...]) -> list[tuple[str, str | None, int, bool]]:
    """(scenario, load override, seed, expect_root_cause)"""
    out = []
    for name, sc in SCENARIOS.items():
        for seed in seeds:
            out.append((name, None, seed, bool(sc.make())))
            if "stress-only" in sc.tags:
                out.append((name, "idle", seed, False))
    return out


def evaluate(workdir: Path, seeds: tuple[int, ...] = DEFAULT_SEEDS, threshold: float = THRESHOLD) -> dict:
    rows = []
    for name, load, seed, expect in cases(seeds):
        bdir, _sim, truth = run(name, Path(workdir) / f"{name}-{load or 'default'}-s{seed}", seed=seed, load=load)
        report = analyze(bdir, threshold)
        truths = truth["root_causes"] if expect else []
        s = score(report, truths)
        rows.append({"scenario": name, "load": truth["load"], "seed": seed, "truth": len(truths), **s,
                     "min_confidence": min((p["confidence"] for p in report["root_causes"]), default=None)})
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return {"threshold": threshold, "seeds": list(seeds), "rows": rows, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 1.0}


def markdown(res: dict) -> str:
    agg: dict[tuple[str, str], dict] = {}
    for r in res["rows"]:
        a = agg.setdefault((r["scenario"], r["load"]), {"runs": 0, "tp": 0, "fp": 0, "fn": 0, "truth": 0,
                                                         "conf": []})
        a["runs"] += 1
        for k in ("tp", "fp", "fn", "truth"):
            a[k] += r[k]
        if r["min_confidence"] is not None:
            a["conf"].append(r["min_confidence"])
    lines = [f"## RCA scenario matrix (threshold {res['threshold']}, seeds {res['seeds']})", "",
             f"**Precision {res['precision']:.3f} · Recall {res['recall']:.3f}** "
             f"(TP {res['tp']}, FP {res['fp']}, FN {res['fn']})", "",
             "| Scenario | Load | Runs | Truth | TP | FP | FN | Min confidence |",
             "|---|---|---|---|---|---|---|---|"]
    for (name, load), a in agg.items():
        conf = f"{min(a['conf']):.2f}" if a["conf"] else "-"
        lines.append(f"| `{name}` | {load} | {a['runs']} | {a['truth']} | {a['tp']} | {a['fp']} | {a['fn']} | {conf} |")
    return "\n".join(lines) + "\n"


def write(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "matrix.json").open("w", encoding=ENCODING) as fh:
        json.dump(res, fh, indent=2)
    with (out / "matrix.md").open("w", encoding=ENCODING) as fh:
        fh.write(markdown(res))
