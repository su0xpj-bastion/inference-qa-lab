"""Cluster: behaviour under concurrent load.

From the client side a serving cluster is: how latency degrades as concurrency
rises, whether 429s are absorbed by backoff or leak out as errors, how much rate
limit headroom remains, and whether concurrent requests stay isolated (each
gets its own answer, never a neighbour's: the batch-isolation property).

Each level starts with an unmeasured warm-up round (connection setup is not
steady-state latency). Load is deliberately small (default 8 concurrent, ~60 requests) so a live run
stays far inside free-tier limits; this measures the curve's shape, it isn't a
stress test.
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor

import cerebras.cloud.sdk as cerebras

from ..core import Check, Ctx, guarded, pct

W = "cluster"


def _one(ctx: Ctx, i: int) -> dict:
    try:
        c = ctx.chat(f"In one sentence, name a use for fast inference (#{i}).", stream=True,
                     max_completion_tokens=48)
        return {"ok": True, "wall": c.wall, "ttft": c.ttft, "queue": c.queue_time or 0.0,
                "retries": c.attempts - 1, "tokens": c.completion_tokens or 0}
    except cerebras.RateLimitError:
        return {"ok": False, "err": "429", "retries": 2}
    except cerebras.APIStatusError as exc:
        return {"ok": False, "err": str(exc.status_code), "retries": 0}
    except cerebras.APIError as exc:
        return {"ok": False, "err": type(exc).__name__, "retries": 0}


def _sweep(ctx: Ctx) -> list[dict]:
    if "_sweep" in ctx.__dict__:
        return ctx.__dict__["_sweep"]
    levels = []
    for level in ctx.slo["concurrency_levels"]:
        n = level * ctx.slo["requests_per_slot"]
        with ThreadPoolExecutor(level) as pool:
            # warm-up round, not measured: opens the connections this level needs, so
            # connection setup isn't counted as steady-state latency
            list(pool.map(lambda i: _one(ctx, -1 - i), range(level)))
            t0 = time.perf_counter()
            res = list(pool.map(lambda i: _one(ctx, i), range(n)))
        elapsed = time.perf_counter() - t0
        ok = [r for r in res if r["ok"]]
        levels.append({
            "concurrency": level, "requests": n, "errors": n - len(ok),
            "error_kinds": sorted({r["err"] for r in res if not r["ok"]}),
            "retries": sum(r["retries"] for r in res),
            "p50": pct([r["wall"] for r in ok], 50), "p95": pct([r["wall"] for r in ok], 95),
            "ttft_p95": pct([r["ttft"] for r in ok if r["ttft"] is not None], 95),
            "queue_p95": pct([r["queue"] for r in ok], 95),
            "rps": len(ok) / elapsed, "tok_s": sum(r["tokens"] for r in ok) / elapsed,
        })
    ctx.__dict__["_sweep"] = levels
    return levels


def _table(levels: list[dict]) -> str:
    return "; ".join(f"c={x['concurrency']}: p50 {x['p50'] * 1e3:.0f} ms, p95 {x['p95'] * 1e3:.0f} ms, "
                     f"{x['rps']:.1f} req/s, err {x['errors']}, retries {x['retries']}" for x in levels)


def error_rate(ctx: Ctx) -> Check:
    levels = _sweep(ctx)
    total = sum(x["requests"] for x in levels)
    errs = sum(x["errors"] for x in levels)
    kinds = sorted({k for x in levels for k in x["error_kinds"]})
    return Check(W, "error_rate", "pass" if errs == 0 else "fail",
                 f"{errs}/{total} requests failed after retries {kinds or ''}".strip(),
                 metrics={"levels": levels})


def retry_absorption(ctx: Ctx) -> Check:
    levels = _sweep(ctx)
    total = sum(x["requests"] for x in levels)
    retries = sum(x["retries"] for x in levels)
    ratio = retries / total
    cap = ctx.slo["max_retry_ratio"]
    status = "pass" if ratio <= cap else "fail"
    return Check(W, "retry_absorption", status,
                 f"{retries} retries over {total} requests ({ratio:.0%}; cap {cap:.0%}); "
                 "429s are backed off using retry-after, then jittered exponential",
                 metrics={"retry_ratio": ratio})


def concurrency_latency(ctx: Ctx) -> Check:
    levels = _sweep(ctx)
    base = max(levels[0]["p50"], ctx.slo["latency_floor_s"])
    top = levels[-1]
    infl = top["p95"] / base
    cap = ctx.slo["max_latency_inflation"]
    return Check(W, "concurrency_latency", "pass" if infl <= cap else "fail",
                 f"p95 at c={top['concurrency']} is {infl:.1f}x c=1 p50 (cap {cap}x). {_table(levels)}",
                 metrics={"inflation": infl, "levels": levels})


def request_isolation(ctx: Ctx) -> Check:
    nums = random.Random(42).sample(range(100, 999), ctx.slo["isolation_requests"])

    def ask(n: int) -> tuple[int, str]:
        c = ctx.chat(f"Reply with only the number that comes after {n}.", temperature=0,
                     max_completion_tokens=8)
        return n, c.content.strip()

    with ThreadPoolExecutor(len(nums)) as pool:
        got = list(pool.map(ask, nums))
    wrong = [(n, a) for n, a in got if a.rstrip(".") != str(n + 1)]
    foreign = [(n, a) for n, a in wrong if a.rstrip(".").isdigit() and int(a.rstrip(".")) - 1 in nums]
    if foreign:
        status, detail = "fail", f"{len(foreign)} responses answered ANOTHER in-flight request: {foreign[:3]}"
    elif wrong:
        status, detail = "warn", f"{len(wrong)} wrong answers (model quality, not crosstalk): {wrong[:3]}"
    else:
        status, detail = "pass", f"{len(nums)} concurrent requests each got their own answer"
    return Check(W, "request_isolation", status, detail)


def rate_limit_headroom(ctx: Ctx) -> Check:
    c = ctx.chat("Reply with only: ok", max_completion_tokens=4)
    h = c.headers
    if not h:
        return Check(W, "rate_limit_headroom", "info", "no x-ratelimit-* headers returned", gated=False)
    lim = h.get("x-ratelimit-limit-requests-minute")
    rem = h.get("x-ratelimit-remaining-requests-minute")
    detail = f"requests/min remaining {rem}/{lim}; " + ", ".join(f"{k}={v}" for k, v in sorted(h.items()))
    return Check(W, "rate_limit_headroom", "info", detail, gated=False, metrics=h)


def _remaining(headers: dict[str, str]) -> tuple[str, int] | None:
    for k, v in sorted(headers.items()):
        if "remaining" in k and "request" in k:
            try:
                return k, int(float(v))
            except ValueError:
                return None
    return None


def rate_limit_refill(ctx: Ctx) -> Check:
    """Cerebras documents token-bucket limits that refill continuously, not fixed windows.
    After an idle gap the remaining-requests count must recover (>= the previous reading),
    where a fixed window would keep counting down until the window resets."""
    a = ctx.chat("Reply with only: ok", max_completion_tokens=4)
    first = _remaining(a.headers)
    if first is None:
        return Check(W, "rate_limit_refill", "info",
                     f"no remaining-requests header; headers seen {sorted(a.headers) or 'none'}", gated=False)
    time.sleep(ctx.slo["refill_wait_s"])
    b = ctx.chat("Reply with only: ok", max_completion_tokens=4)
    second = _remaining(b.headers)
    if second is None:
        return Check(W, "rate_limit_refill", "fail", f"{first[0]} missing on the second response")
    ok = second[1] >= first[1]
    return Check(W, "rate_limit_refill", "pass" if ok else ("fail" if ctx.slo["refill_gated"] else "warn"),
                 f"{first[0]}: {first[1]} -> {second[1]} after {ctx.slo['refill_wait_s']}s idle "
                 f"({'refilled: token bucket' if ok else 'still falling: fixed window?'})",
                 gated=ctx.slo["refill_gated"])


CHECKS = (error_rate, retry_absorption, concurrency_latency, request_isolation, rate_limit_headroom,
          rate_limit_refill)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
