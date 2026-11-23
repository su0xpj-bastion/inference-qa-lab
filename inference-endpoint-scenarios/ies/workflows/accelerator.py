"""Accelerator (GPU / wafer-scale): performance and numerics, observed from the API.

A hosted endpoint hides the hardware, but the hardware still shows up in what the
API returns: decode throughput (server time_info), prefill rate vs prompt length,
TTFT and where it goes (queue / prefill / network), prompt-cache effect, and
numerical reproducibility (identical output and logprob drift at T=0 + seed).
A kernel, compiler, or firmware regression moves these before it moves anything else.
"""
from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..core import Check, Ctx, client_overhead, guarded, mean, pct, server_decode_tps, tbt

W = "accelerator"
LONG_PROMPT = "Write a detailed essay on how inference hardware affects application design."
FILLER = ("Wafer-scale engines keep model weights in on-chip SRAM, so decode is limited by compute "
          "and fabric bandwidth rather than by off-chip memory traffic. ")


def _filler(tokens: int) -> str:
    per = 24
    return FILLER * max(1, tokens // per)


def _decode_runs(ctx: Ctx) -> list:
    if "_decode" not in ctx.__dict__:
        ctx.__dict__["_decode"] = [ctx.chat(LONG_PROMPT, stream=True, max_completion_tokens=256, temperature=0.7)
                                   for _ in range(ctx.slo["decode_runs"])]
    return ctx.__dict__["_decode"]


def decode_throughput(ctx: Ctx) -> Check:
    runs = _decode_runs(ctx)
    tps = [t for t in (server_decode_tps(c) for c in runs) if t]
    short = [c.completion_tokens for c in runs if (c.completion_tokens or 0) < 64]
    if not tps:
        return Check(W, "decode_throughput", "fail", "no time_info.completion_time in any run")
    p50 = pct(tps, 50)
    floor = ctx.slo["min_decode_tps"]
    detail = f"server decode p50 {p50:,.0f} tok/s (floor {floor:,}); min {min(tps):,.0f}; n={len(tps)}"
    if short:
        detail += f"; {len(short)} short runs ({short}) weaken the sample"
    return Check(W, "decode_throughput", "pass" if p50 >= floor else "fail", detail,
                 metrics={"decode_tps_p50": p50, "decode_tps": tps,
                          "tbt_ms": [tbt(c.content_times) * 1e3 for c in runs]})


def ttft(ctx: Ctx) -> Check:
    runs = _decode_runs(ctx)
    t = [c.ttft for c in runs if c.ttft is not None]
    if not t:
        return Check(W, "ttft", "fail", "no content tokens streamed")
    p50, budget = pct(t, 50), ctx.slo["ttft_p50_max_s"]
    q = mean([c.queue_time or 0 for c in runs])
    p = mean([c.prompt_time or 0 for c in runs])
    net = mean([o for o in (client_overhead(c) for c in runs) if o is not None])
    return Check(W, "ttft", "pass" if p50 <= budget else "fail",
                 f"TTFT p50 {p50 * 1e3:.0f} ms (budget {budget * 1e3:.0f}); server queue {q * 1e3:.1f} ms, "
                 f"prefill {p * 1e3:.1f} ms, unaccounted wall (network/TLS/client) {net * 1e3:.0f} ms",
                 metrics={"ttft_p50": p50, "ttft_p95": pct(t, 95), "queue": q, "prefill": p, "network": net})


def prefill_scaling(ctx: Ctx) -> Check:
    rows = []
    for size in ctx.slo["prefill_sizes"]:
        nonce = uuid.uuid4().hex   # defeat the prompt cache: measure real prefill
        c = ctx.chat([{"role": "system", "content": f"[{nonce}] " + _filler(size)},
                      {"role": "user", "content": "Reply with only: ok"}], max_completion_tokens=4)
        uncached = (c.prompt_tokens or 0) - c.cached_tokens
        rows.append((c.prompt_tokens, uncached, c.prompt_time))
    big = rows[-1]
    if not big[2]:
        return Check(W, "prefill_scaling", "fail", "time_info.prompt_time missing")
    rate = big[1] / big[2]
    floor = ctx.slo["min_prefill_tps"]
    detail = "; ".join(f"{p} tok -> {t * 1e3:.1f} ms" for p, _, t in rows if t is not None)
    return Check(W, "prefill_scaling", "pass" if rate >= floor else "fail",
                 f"prefill {rate:,.0f} tok/s at {big[0]} tokens (floor {floor:,}); {detail}",
                 metrics={"rows": rows, "prefill_tps": rate})


def prompt_cache(ctx: Ctx) -> Check:
    key = f"ies-{uuid.uuid4().hex[:12]}"
    system = f"[{key}] " + _filler(ctx.slo["cache_prefix_tokens"])
    calls = [ctx.chat([{"role": "system", "content": system},
                       {"role": "user", "content": q}], max_completion_tokens=4,
                       extra_body={"prompt_cache_key": key})   # typed param only in newer SDKs
             for q in ("Reply with only: one", "Reply with only: two")]
    first, second = calls
    hit = second.cached_tokens > 0
    ratio = (second.prompt_time / first.prompt_time) if first.prompt_time and second.prompt_time else None
    detail = (f"2nd call cached_tokens={second.cached_tokens}/{second.prompt_tokens}; prompt_time "
              f"{(first.prompt_time or 0) * 1e3:.1f} -> {(second.prompt_time or 0) * 1e3:.1f} ms")
    return Check(W, "prompt_cache", "pass" if hit else "warn", detail, gated=ctx.slo["cache_gated"],
                 metrics={"cached": second.cached_tokens, "prompt_time_ratio": ratio})


def determinism(ctx: Ctx) -> Check:
    outs = [ctx.chat("Explain in detail what a KV cache stores.", temperature=0, seed=1234,
                     max_completion_tokens=64).content for _ in range(ctx.slo["determinism_reps"])]
    modal = max(set(outs), key=outs.count)
    rate = outs.count(modal) / len(outs)
    floor = ctx.slo["min_identical_rate"]
    return Check(W, "determinism", "pass" if rate >= floor else "fail",
                 f"{outs.count(modal)}/{len(outs)} identical at T=0 seed=1234 (floor {floor:.0%}); "
                 f"{len(set(outs))} distinct outputs", gated=ctx.slo["determinism_gated"],
                 metrics={"identical_rate": rate})


def logprob_stability(ctx: Ctx) -> Check:
    runs = [ctx.chat("Explain in detail what a KV cache stores.", temperature=0, seed=1234,
                     max_completion_tokens=32, logprobs=True, top_logprobs=3).logprobs
            for _ in range(3)]
    if not all(runs):
        return Check(W, "logprob_stability", "fail", "logprobs requested but not returned")
    drift = 0.0
    compared = 0
    for other in runs[1:]:
        for (ta, la), (tb, lb) in zip(runs[0], other):
            if ta != tb:
                break                      # paths diverged: later positions aren't comparable
            drift = max(drift, abs(la - lb))
            compared += 1
    tol = ctx.slo["max_logprob_drift"]
    return Check(W, "logprob_stability", "pass" if drift <= tol else "fail",
                 f"max |dlogprob| {drift:.2e} over {compared} shared-prefix tokens (tol {tol:.0e})",
                 gated=ctx.slo["determinism_gated"], metrics={"max_drift": drift})


INVARIANCE_PROMPT = "Explain in detail how a wafer-scale processor schedules decode."


def load_invariance(ctx: Ctx) -> Check:
    """Same T=0 request alone vs inside a busy batch. Batch-size-dependent kernels (reduction
    order changes with batch shape) make outputs depend on load, a known GPU serving bug."""
    params = {"temperature": 0, "seed": 9, "max_completion_tokens": 48}
    alone = [ctx.chat(INVARIANCE_PROMPT, **params).content for _ in range(2)]
    loaded = []
    for r in range(ctx.slo["invariance_rounds"]):
        others = [f"Write a detailed paragraph about topic {r}-{i}." for i in range(ctx.slo["invariance_load"])]
        with ThreadPoolExecutor(len(others) + 1) as pool:
            futs = [pool.submit(ctx.chat, o, max_completion_tokens=64) for o in others]
            target = pool.submit(ctx.chat, INVARIANCE_PROMPT, **params)
            [f.result() for f in futs]
            loaded.append(target.result().content)
    base = alone[0]
    alone_same = all(a == base for a in alone)
    loaded_same = sum(x == base for x in loaded)
    gated = ctx.slo["invariance_gated"]
    if not alone_same:
        return Check(W, "load_invariance", "fail" if gated else "warn",
                     "not even reproducible alone at T=0 + seed; load effect can't be isolated", gated=gated)
    ok = loaded_same == len(loaded)
    return Check(W, "load_invariance", "pass" if ok else ("fail" if gated else "warn"),
                 f"T=0 output identical alone (2/2) and under load with {ctx.slo['invariance_load']} concurrent "
                 f"requests in {loaded_same}/{len(loaded)} rounds", gated=gated,
                 metrics={"loaded_identical": loaded_same, "rounds": len(loaded)})


NUMERICS_PROMPTS = ("Explain in detail what a KV cache stores.", "Write a detailed story about a lighthouse.",
                    "Explain in detail how gradient checkpointing saves memory.")


def _numerics_path(ctx: Ctx) -> Path:
    return Path(ctx.baseline_dir) / f"{ctx.target}-{ctx.model}-numerics.json"


def logprob_baseline(ctx: Ctx) -> Check:
    """Top-1 logprobs on fixed prompts vs the recorded baseline. Same backend build: drift must stay
    at the noise floor. New build (system_fingerprint changed): drift is reported, since numerics
    are expected to move and the size of the move is the information."""
    now: dict[str, list] = {}
    fps = set()
    for prompt in NUMERICS_PROMPTS:
        c = ctx.chat(prompt, temperature=0, seed=1234, max_completion_tokens=24, logprobs=True, top_logprobs=1)
        now[prompt] = [[t, lp] for t, lp in c.logprobs]
        fps.add(c.fingerprint)
    fp = ",".join(sorted(fps))
    path = _numerics_path(ctx)
    base = None
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            base = json.load(fh)
    if ctx.update_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fingerprint": fp, "logprobs": now}, indent=1) + "\n")
    if base is None:
        return Check(W, "logprob_baseline", "warn", f"no numerics baseline at {path}; run --update-baseline",
                     gated=False)
    drift, compared, diverged = 0.0, 0, []
    for prompt, cur in now.items():
        old = base["logprobs"].get(prompt, [])
        for i, ((ta, la), (tb, lb)) in enumerate(zip(old, cur)):
            if ta != tb:
                diverged.append(i)
                break
            drift = max(drift, abs(la - lb))
            compared += 1
    tol = ctx.slo["max_logprob_drift"]
    detail = (f"max |dlogprob| vs baseline {drift:.2e} over {compared} tokens (tol {tol:.0e}); "
              f"token paths diverged at {diverged or 'none'}")
    if base["fingerprint"] != fp:
        return Check(W, "logprob_baseline", "warn",
                     f"new backend build {base['fingerprint']} -> {fp}: {detail}", gated=False,
                     metrics={"drift": drift})
    bad = drift > tol or bool(diverged)
    return Check(W, "logprob_baseline", "fail" if bad else "pass", f"same build {fp}: {detail}",
                 gated=ctx.slo["determinism_gated"], metrics={"drift": drift})


CHECKS = (decode_throughput, ttft, prefill_scaling, prompt_cache, determinism, logprob_stability,
          load_invariance, logprob_baseline)


def run(ctx: Ctx) -> list[Check]:
    out = []
    for fn in CHECKS:
        c = guarded(W, fn.__name__, lambda fn=fn: fn(ctx))
        out.append(c)
    return out
