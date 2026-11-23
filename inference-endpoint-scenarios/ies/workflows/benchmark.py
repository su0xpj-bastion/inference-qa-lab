"""Benchmark: a fixed GSM8K slice as a statistical release gate.

The 8 golden cases in `release` say whether specific answers broke. This says whether
overall quality moved, and whether the move is bigger than noise: 50 grade-school
math problems (openai/grade-school-math, MIT, vendored in ies/data/) with exact
numeric answers, scored against the recorded baseline with a one-sided
two-proportion z-test.

  benchmark_gate   NO-GO         drop is significant (p < nogo_alpha)
                   INCONCLUSIVE  a drop that could be noise or real (nogo_alpha <= p < go_alpha):
                                 re-run with more items before deciding
                   GO            no significant drop
                   Also reports the Wilson 95% interval and which items flipped.

It is a release gate, not a model leaderboard: the question is "did this build get
worse than the last one we qualified", not "how good is the model".
"""
from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..core import Check, Ctx, guarded

W = "benchmark"
DATA = Path(__file__).resolve().parent.parent / "data" / "gsm8k_slice.jsonl"
SUFFIX = "\n\nSolve it briefly, then give the final answer on the last line as: Answer: <number>"
ANSWER = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")


def load_items(path: Path = DATA) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract(text: str) -> str | None:
    found = ANSWER.findall(text)
    if not found:
        return None
    v = found[-1].replace(",", "")
    return v[:-2] if v.endswith(".0") else v.rstrip(".")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def drop_p_value(k0: int, n0: int, k1: int, n1: int) -> float:
    """One-sided p-value that the current rate (k1/n1) is lower than the baseline (k0/n0)."""
    p0, p1 = k0 / n0, k1 / n1
    pooled = (k0 + k1) / (n0 + n1)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    if se == 0:
        return 1.0 if p1 >= p0 else 0.0
    z = (p0 - p1) / se
    return 0.5 * math.erfc(z / math.sqrt(2))


def _score(ctx: Ctx, item: dict) -> tuple[str, bool, str]:
    c = ctx.chat(item["question"] + SUFFIX, temperature=0, seed=5, max_completion_tokens=ctx.slo["bench_max_tokens"])
    got = extract(c.content)
    return item["id"], got == item["answer"], got or "-"


def benchmark_gate(ctx: Ctx) -> Check:
    items = load_items()[: ctx.slo["bench_items"]]
    with ThreadPoolExecutor(ctx.slo["bench_concurrency"]) as pool:
        results = list(pool.map(lambda it: _score(ctx, it), items))
    passed = {i for i, ok, _ in results if ok}
    k, n = len(passed), len(results)
    lo, hi = wilson(k, n)
    path = Path(ctx.baseline_dir) / f"{ctx.target}-{ctx.model}-gsm8k.json"
    base = None
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            base = json.load(fh)
    if ctx.update_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"passed": sorted(passed), "n": n}, indent=1) + "\n")
    head = f"{k}/{n} = {k / n:.0%} (95% CI {lo:.0%}-{hi:.0%})"
    if base is None:
        return Check(W, "benchmark_gate", "warn", f"{head}; no baseline at {path}, run --update-baseline",
                     gated=False, metrics={"k": k, "n": n})
    k0, n0 = len(base["passed"]), base["n"]
    p = drop_p_value(k0, n0, k, n)
    flipped = sorted(set(base["passed"]) - passed)
    fixed = sorted(passed - set(base["passed"]))
    if p < ctx.slo["bench_nogo_alpha"]:
        gate = "NO-GO"
    elif p < ctx.slo["bench_go_alpha"]:
        gate = "INCONCLUSIVE"
    else:
        gate = "GO"
    detail = (f"{gate}: {head} vs baseline {k0}/{n0}; one-sided p={p:.3f} "
              f"(NO-GO < {ctx.slo['bench_nogo_alpha']}, GO >= {ctx.slo['bench_go_alpha']}); "
              f"newly failing {flipped[:4] or 'none'}{' ...' if len(flipped) > 4 else ''}; newly passing {len(fixed)}")
    return Check(W, "benchmark_gate", "pass" if gate == "GO" else "fail", detail,
                 metrics={"gate": gate, "p": p, "k": k, "n": n, "flipped": flipped})


CHECKS = (benchmark_gate,)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
