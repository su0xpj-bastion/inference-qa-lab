"""Conversation: multi-turn memory and prompt-cache reuse as history grows.

A chat app (cerebras/inference-examples/conversational-memory-langchain) resends the
whole history every turn. Two things must hold:

  conversation_recall        a fact given in turn 1 is recalled correctly in the last turn
                             (history really reached the model, nothing silently dropped)
  conversation_cache_growth  with one prompt_cache_key per conversation, cached_tokens grows
                             turn over turn: each turn re-uses the previous turns' prefix
  conversation_prefill_flat  turns 2..n pay prefill only for new messages: the uncached share
                             of the prompt stays small (~5% here) instead of ~100%
"""
from __future__ import annotations

import uuid

from ..core import Check, Ctx, guarded

W = "conversation"
CODENAME = "BLUE-HERON-47"
TOPICS = ("rate limits", "streaming", "prompt caching", "tool calling", "structured outputs", "retries",
          "timeouts", "batching", "observability", "cost tracking")
GUIDE = ("You are a concise assistant for an inference platform team. Answer in one sentence. "
         "Team handbook follows. ") + " ".join(
    f"Section {i}: requests carry a budget, a deadline, and a trace id; services log latency per hop and "
    f"retry only on 429 with jittered backoff." for i in range(1, 61))


def _run(ctx: Ctx) -> dict:
    if "_conv" in ctx.__dict__:
        return ctx.__dict__["_conv"]
    key = f"conv-{uuid.uuid4().hex[:12]}"
    msgs: list[dict] = [{"role": "system", "content": f"[{key}] " + GUIDE},
                        {"role": "user", "content": f"My project codename is {CODENAME}. "
                                                    "Remember it. Reply with only: noted"}]
    turns = []
    n = ctx.slo["conv_turns"]
    for t in range(n):
        if t > 0:
            q = (f"What is my project codename? Reply with only the codename." if t == n - 1
                 else f"Give one short tip about {TOPICS[t % len(TOPICS)]}.")
            msgs.append({"role": "user", "content": q})
        c = ctx.chat(msgs, temperature=0, max_completion_tokens=48, extra_body={"prompt_cache_key": key})
        msgs.append({"role": "assistant", "content": c.content})
        turns.append({"turn": t + 1, "prompt": c.prompt_tokens or 0, "cached": c.cached_tokens,
                      "uncached": (c.prompt_tokens or 0) - c.cached_tokens, "prompt_time": c.prompt_time,
                      "content": c.content})
    ctx.__dict__["_conv"] = {"turns": turns, "final": turns[-1]["content"]}
    return ctx.__dict__["_conv"]


def conversation_recall(ctx: Ctx) -> Check:
    r = _run(ctx)
    ok = CODENAME.lower() in r["final"].lower()
    prompts = [t["prompt"] for t in r["turns"]]
    detail = f"turn {len(prompts)} answer {r['final']!r} (want {CODENAME}); prompt tokens per turn {prompts}"
    if any(b < a for a, b in zip(prompts, prompts[1:])):
        detail += "; prompt_tokens SHRANK between turns (history dropped)"
        ok = False
    return Check(W, "conversation_recall", "pass" if ok else "fail", detail)


def conversation_cache_growth(ctx: Ctx) -> Check:
    turns = _run(ctx)["turns"]
    cached = [t["cached"] for t in turns]
    later = cached[1:]
    grows = all(c > 0 for c in later) and later[-1] >= later[0]
    gated = ctx.slo["conv_cache_gated"]
    status = "pass" if grows else ("fail" if gated else "warn")
    return Check(W, "conversation_cache_growth", status,
                 f"cached_tokens per turn {cached} (prompt {[t['prompt'] for t in turns]})", gated=gated,
                 metrics={"cached": cached})


def conversation_prefill_flat(ctx: Ctx) -> Check:
    """Turns 2..n should pay prefill only for the new messages, not the whole history."""
    turns = _run(ctx)["turns"][1:]
    uncached = [t["uncached"] for t in turns]
    share = sum(uncached) / max(1, sum(t["prompt"] for t in turns))
    cap = ctx.slo["conv_uncached_share_max"]
    times = [f"{(t['prompt_time'] or 0) * 1e3:.1f}" for t in turns]
    gated = ctx.slo["conv_cache_gated"]
    status = "pass" if share <= cap else ("fail" if gated else "warn")
    return Check(W, "conversation_prefill_flat", status,
                 f"uncached share of prompt over turns 2-{len(turns) + 1}: {share:.0%} (cap {cap:.0%}); "
                 f"uncached tokens {uncached}; prompt_time ms {times}",
                 gated=gated, metrics={"uncached": uncached, "uncached_share": share})


CHECKS = (conversation_recall, conversation_cache_growth, conversation_prefill_flat)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
