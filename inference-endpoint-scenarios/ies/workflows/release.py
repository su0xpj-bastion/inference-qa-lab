"""Release: qualify a model / backend build against a golden suite and a baseline.

Every case has a machine-checkable answer. Each case runs `reps` times at T=0 so
a flaky case is told apart from a regression:
  regression   passed in the baseline, fails every rep   -> NO-GO
  flaky        mixed results across reps                 -> INCONCLUSIVE (re-run / investigate)
  pass rate    below the floor                           -> NO-GO
A system_fingerprint change (a new backend build behind the same model id) is
reported, because it's the moment a silent regression ships.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..client import parse_args
from ..core import Check, Ctx, guarded
from .inference import PERSON_SCHEMA, WEATHER_TOOL

W = "release"
GOLDENS = Path(__file__).resolve().parent.parent / "goldens.json"


def load_goldens(path: Path = GOLDENS) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _norm(s: str) -> str:
    return re.sub(r"[\s.]+$", "", s.strip().lower())


def grade(case: dict, ctx: Ctx) -> tuple[bool, str, str]:
    kind, want = case["check"], case["expect"]
    params: dict = {"temperature": 0, "seed": 11, "max_completion_tokens": case.get("max_tokens", 32)}
    if kind == "json":
        params["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "person", "strict": True, "schema": PERSON_SCHEMA}}
    if kind == "tool":
        params.update(tools=[WEATHER_TOOL], tool_choice="required")
    c = ctx.chat(case["prompt"], **params)
    if kind == "tool":
        got = {"name": c.tool_calls[0]["name"] if c.tool_calls else None, **parse_args(c)}
        return got == want, json.dumps(got), c.fingerprint
    if kind == "json":
        got = json.loads(c.content)
        return got == want, json.dumps(got), c.fingerprint
    if kind == "exact":
        return _norm(c.content) == _norm(str(want)), c.content, c.fingerprint
    if kind == "list":
        nums = re.findall(r"-?\d+", c.content)
        return nums == [str(x) for x in want], c.content, c.fingerprint
    if kind == "contains":
        return str(want).lower() in c.content.lower(), c.content, c.fingerprint
    raise ValueError(f"unknown check kind {kind}")


def baseline_path(ctx: Ctx) -> Path:
    return Path(ctx.baseline_dir) / f"{ctx.target}-{ctx.model}.json"


def _load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def qualify(ctx: Ctx) -> list[Check]:
    cases = load_goldens()
    reps = ctx.slo["release_reps"]
    results: dict[str, dict] = {}
    fps: set[str] = set()
    for case in cases:
        outcomes = []
        sample = ""
        for _ in range(reps):
            ok, got, fp = grade(case, ctx)
            outcomes.append(ok)
            fps.add(fp)
            if not ok and not sample:
                sample = got
        results[case["id"]] = {"passes": sum(outcomes), "reps": reps, "sample_fail": sample[:80]}

    passed = [k for k, r in results.items() if r["passes"] == r["reps"]]
    flaky = [k for k, r in results.items() if 0 < r["passes"] < r["reps"]]
    failed = [k for k, r in results.items() if r["passes"] == 0]
    stable = len(results) - len(flaky)
    rate = len(passed) / stable if stable else 0.0
    fingerprint = ",".join(sorted(fps))

    bpath = baseline_path(ctx)
    base = _load_baseline(bpath)
    regressions = [k for k in failed if base and base["results"].get(k) is True]
    fp_changed = bool(base) and base["fingerprint"] != fingerprint

    rate_ok = rate >= ctx.slo["min_golden_pass_rate"]
    checks = [Check(W, "golden_pass_rate", ("warn" if flaky else "pass") if rate_ok else "fail",
                    f"{len(passed)}/{stable} stable cases pass all {reps} reps ({rate:.0%}; floor "
                    f"{ctx.slo['min_golden_pass_rate']:.0%}); failed {failed or '-'}; flaky {flaky or '-'}",
                    metrics={"results": results})]
    if base is None:
        checks.append(Check(W, "baseline_diff", "warn",
                            f"no baseline at {bpath}; run with --update-baseline to record one", gated=False))
    else:
        detail = f"regressions vs baseline: {regressions or 'none'}"
        if fp_changed:
            detail += f"; system_fingerprint {base['fingerprint']} -> {fingerprint} (new backend build)"
        checks.append(Check(W, "baseline_diff", "fail" if regressions else ("warn" if fp_changed else "pass"),
                            detail, metrics={"fingerprint": fingerprint}))
        for k in regressions:
            checks[-1].detail += f"; {k}: got {results[k]['sample_fail']!r}"

    if regressions or not rate_ok:
        gate, why = "NO-GO", "regression or pass rate below floor"
    elif flaky:
        gate, why = "INCONCLUSIVE", f"flaky cases {flaky}: non-deterministic at T=0; re-run before deciding"
    else:
        gate, why = "GO", "all cases stable; no regressions vs baseline"
    checks.append(Check(W, "release_gate", "pass" if gate == "GO" else "fail", f"{gate}: {why}",
                        metrics={"gate": gate, "flaky": flaky, "regressions": regressions}))

    if ctx.update_baseline:
        bpath.parent.mkdir(parents=True, exist_ok=True)
        bpath.write_text(json.dumps({"model": ctx.model, "target": ctx.target, "fingerprint": fingerprint,
                                     "results": {k: k in passed for k in results}}, indent=2) + "\n")
    return checks


def models_diff(ctx: Ctx) -> Check:
    """/v1/models vs the recorded list: a removed model is a deprecation users hit as 404s."""
    ctx.budget.take()
    ids = sorted(m.id for m in ctx.client.models.list().data)
    path = Path(ctx.baseline_dir) / f"{ctx.target}-models.json"
    base = None
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            base = json.load(fh)["models"]
    if ctx.update_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"models": ids}, indent=1) + "\n")
    if base is None:
        return Check(W, "models_diff", "warn", f"no model-list baseline at {path}; {len(ids)} models now",
                     gated=False)
    removed, added = sorted(set(base) - set(ids)), sorted(set(ids) - set(base))
    if ctx.model in removed:
        return Check(W, "models_diff", "fail", f"model under test {ctx.model} REMOVED from /v1/models")
    status = "warn" if removed else "pass"
    return Check(W, "models_diff", status, f"removed {removed or 'none'}; added {added or 'none'}",
                 gated=False, metrics={"removed": removed, "added": added})


def run(ctx: Ctx) -> list[Check]:
    out: list[Check] = []

    def wrapped() -> Check:
        out.extend(qualify(ctx))
        return out[-1]

    gate = guarded(W, "release_gate", wrapped)
    checks = out if out else [gate]
    return checks + [guarded(W, "models_diff", lambda: models_diff(ctx))]
