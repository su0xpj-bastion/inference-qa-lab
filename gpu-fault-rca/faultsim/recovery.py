"""Recovery-path validation: replay the incident with the RCA's recommended action applied.

1. Run the scenario with no remediation -> incident bundle -> RCA.
2. For each root cause, schedule its recommended action at detected_at + AUTOMATION_LAG_S.
3. Re-run the same seed with those actions (request/noise RNG streams are
   independent of faults, so traffic is identical up to the first action).
4. Measure from the replay's router log: time to detect, time to mitigate,
   requests lost; compare with slo/recovery.json.

The harness side may import `rca`; `rca` never imports `faultsim`.
"""
from __future__ import annotations

import json
from pathlib import Path

from faultsim.scenarios import SCENARIOS, run
from faultsim.sim import ACTION_DURATION_S, Action
from rca import bundle as rbundle
from rca.engine import analyze
from rca.symptoms import detect

ENCODING = "utf-8"
AUTOMATION_LAG_S = 10
CLEAN_PERIOD_S = 60
SLO_FILE = Path(__file__).resolve().parent.parent / "slo" / "recovery.json"


def load_slo(path: Path = SLO_FILE) -> dict:
    with path.open("r", encoding=ENCODING) as fh:
        return json.load(fh)


def plan(report: dict) -> list[Action]:
    acts = []
    for rc in report["root_causes"]:
        ra = rc["recommended_action"]
        if ra["action"] not in ACTION_DURATION_S:
            continue
        tgt = ra["target"]
        acts.append(Action(ra["action"], int(rc["detected_at_s"]) + AUTOMATION_LAG_S, tgt["node"], tgt["gpu"],
                           tgt["replica"] if ra["action"] == "ROLLBACK_DEPLOY" else None))
    return acts


def validate(name: str, workdir: Path, seed: int = 0, actions: list[Action] | None = None,
             load: str | None = None) -> dict:
    workdir = Path(workdir)
    inc_dir, _sim, truth = run(name, workdir / f"{name}-s{seed}-incident", seed=seed, load=load)
    report = analyze(inc_dir)
    impact = [s for s in report["symptoms"]]
    if not impact:
        return {"scenario": name, "seed": seed, "impact": False, "verdict": "no-impact"}
    impact_start = min(s["start_s"] for s in impact)
    acts = plan(report) if actions is None else actions
    rep_dir, _sim2, _ = run(name, workdir / f"{name}-s{seed}-replay", seed=seed, load=load, actions=tuple(acts))
    b = rbundle.load(rep_dir)
    syms = detect(b.requests, b.start, b.slo["p99_latency_s"], b.slo["error_rate"])
    fails = sorted(q.end - b.start for q in b.requests if q.status >= 500)
    done = max((a.at + ACTION_DURATION_S[a.kind] for a in acts), default=None)
    ttm = None
    breaches_after: list[dict] = []
    if done is not None:
        t = float(done)
        horizon = b.manifest["duration_s"]
        while t + CLEAN_PERIOD_S <= horizon:
            clash = [s for s in syms if s.start - b.start < t + CLEAN_PERIOD_S and s.end - b.start > t]
            # a broken replica left in rotation is masked by passive ejection to ~1 probe failure / 10 s:
            # under a symptom threshold, but not mitigated. Clean means zero 5xx.
            stray = [x for x in fails if t < x < t + CLEAN_PERIOD_S]
            if stray and not clash:
                t = stray[-1]
                continue
            if not clash:
                ttm = max(t, impact_start) - impact_start
                breaches_after = [s.to_dict(b.start) for s in syms if s.start - b.start >= t + CLEAN_PERIOD_S]
                break
            t = max(s.end - b.start for s in clash)
    onset = min(rc["onset_s"] for rc in truth["root_causes"]) if truth["root_causes"] else 0
    window_end = (ttm + impact_start + CLEAN_PERIOD_S) if ttm is not None else b.manifest["duration_s"]
    lost = sum(1 for q in b.requests if q.status >= 500 and onset <= q.end - b.start <= window_end)
    detected = min((rc["detected_at_s"] for rc in report["root_causes"]), default=None)
    ttd = None if detected is None else round(detected - impact_start, 1)
    slo_all = load_slo()
    causes = [rc["cause"] for rc in report["root_causes"]]
    slo = dict(slo_all["default"])
    for c in causes:
        for k, v in slo_all.get(c, {}).items():
            slo[k] = max(slo.get(k, v), v) if len(causes) > 1 else v
    checks = {
        "detected": ttd is not None,
        "ttd": ttd is not None and ttd <= slo["ttd_s"],
        "mitigated": ttm is not None,
        "ttm": ttm is not None and ttm <= slo["ttm_s"],
        "requests_lost": lost <= slo["requests_lost"],
    }
    return {"scenario": name, "seed": seed, "impact": True, "causes": causes,
            "actions": [{"action": a.kind, "at_s": a.at, "node": a.node, "gpu": a.gpu, "replica": a.replica}
                        for a in acts],
            "impact_start_s": impact_start, "ttd_s": ttd, "ttm_s": None if ttm is None else round(ttm, 1),
            "requests_lost": lost, "slo": slo, "checks": checks,
            "post_recovery_breaches": breaches_after,
            "verdict": "pass" if all(checks.values()) else "fail"}


def recovery_scenarios() -> list[str]:
    return [n for n, s in SCENARIOS.items() if n != "no_fault" and "soak-only" not in s.tags]
