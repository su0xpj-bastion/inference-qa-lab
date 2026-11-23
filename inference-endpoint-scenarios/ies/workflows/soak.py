"""Soak: steady low-rate traffic for minutes, watching for slow degradation.

Not part of `all` (it runs for minutes live): `python -m ies run -w soak`, and the
nightly live CI job runs it. One small request per tick at a fixed rate, far below
free-tier limits; the run is split into windows and compared first vs last.

  soak_errors        0 requests failed after retries for the whole run
  soak_latency_creep p95 latency in the last window <= cap x the first window
                     (memory growth, leaking connections, a degrading node)
  soak_queue         queue_time p95 per window: reported; spikes are capacity events
"""
from __future__ import annotations

import time

import cerebras.cloud.sdk as cerebras

from ..core import Check, Ctx, guarded, pct

W = "soak"


def _run(ctx: Ctx) -> list[dict]:
    if "_soak" in ctx.__dict__:
        return ctx.__dict__["_soak"]
    period = 1.0 / ctx.slo["soak_rps"]
    end = time.monotonic() + ctx.slo["soak_seconds"]
    rows = []
    i = 0
    while time.monotonic() < end:
        t0 = time.monotonic()
        row = {"i": i, "t": t0}
        try:
            c = ctx.chat(f"Reply with only the number {i % 97}.", max_completion_tokens=8, temperature=0)
            row.update(ok=True, wall=c.wall, queue=c.queue_time or 0.0, retries=c.attempts - 1)
        except cerebras.RateLimitError:
            row.update(ok=False, err="429")
        except cerebras.APIStatusError as exc:
            row.update(ok=False, err=str(exc.status_code))
        except cerebras.APIError as exc:
            row.update(ok=False, err=type(exc).__name__)
        rows.append(row)
        i += 1
        time.sleep(max(0.0, period - (time.monotonic() - t0)))
    ctx.__dict__["_soak"] = rows
    return rows


def _windows(rows: list[dict], n: int) -> list[list[dict]]:
    size = max(1, len(rows) // n)
    return [rows[k:k + size] for k in range(0, size * n, size)]


def soak_errors(ctx: Ctx) -> Check:
    rows = _run(ctx)
    bad = [(r["i"], r["err"]) for r in rows if not r["ok"]]
    return Check(W, "soak_errors", "fail" if bad else "pass",
                 f"{len(bad)}/{len(rows)} failed over {ctx.slo['soak_seconds']}s at {ctx.slo['soak_rps']} rps"
                 + (f"; first {bad[:5]}" if bad else ""))


def soak_latency_creep(ctx: Ctx) -> Check:
    wins = _windows([r for r in _run(ctx) if r["ok"]], ctx.slo["soak_windows"])
    p95 = [pct([r["wall"] for r in w], 95) for w in wins if w]
    if len(p95) < 2:
        return Check(W, "soak_latency_creep", "fail", "not enough successful requests")
    ratio = p95[-1] / max(p95[0], ctx.slo["latency_floor_s"])
    cap = ctx.slo["soak_creep_cap"]
    return Check(W, "soak_latency_creep", "pass" if ratio <= cap else "fail",
                 f"p95 per window (ms) {[round(x * 1e3) for x in p95]}; last/first {ratio:.1f}x (cap {cap}x)",
                 metrics={"p95_windows": p95})


def soak_queue(ctx: Ctx) -> Check:
    wins = _windows([r for r in _run(ctx) if r["ok"]], ctx.slo["soak_windows"])
    q = [pct([r["queue"] for r in w], 95) for w in wins if w]
    return Check(W, "soak_queue", "info", f"queue_time p95 per window (ms) {[round(x * 1e3, 1) for x in q]}",
                 gated=False, metrics={"queue_p95_windows": q})


CHECKS = (soak_errors, soak_latency_creep, soak_queue)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
