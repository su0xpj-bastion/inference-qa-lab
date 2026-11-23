"""Reasoning: effort levels, where reasoning text goes, and how it is billed.

Reasoning models (qwen-3.8-27b, gpt-oss-120b on Cerebras; gpt-oss uses OpenAI's
harmony format, see openai/harmony) generate hidden tokens before the answer. The
API contract around them is easy to break without any error:

  reasoning_effort_scaling  none <= low <= medium <= high in reasoning_tokens (none = 0)
  reasoning_separation      reasoning arrives in message.reasoning / delta.reasoning, never in
                            content: no <think> blocks, no harmony control tokens
                            (<|channel|>, <|start|>, <|message|>, <|end|>)
  reasoning_billing         reasoning_tokens are included in completion_tokens (and usage adds up)
  reasoning_budget          max_completion_tokens covers reasoning: a tight budget ends with
                            finish_reason "length", not a "stop" with an empty answer
"""
from __future__ import annotations

from ..core import Check, Ctx, guarded

W = "reasoning"
PROMPT = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
          "How much does the ball cost? Reply with only the amount.")
MARKERS = ("<think>", "</think>", "<|channel|>", "<|start|>", "<|message|>", "<|end|>", "<|return|>")
LEVELS = ("none", "low", "medium", "high")
NONE_OK = {"qwen-3.8-27b", "gemma-4-31b"}   # models that accept reasoning_effort="none" (API reference)


def _levels(ctx: Ctx) -> dict:
    if "_levels" not in ctx.__dict__:
        out = {}
        for level in ctx.slo["reasoning_levels"]:
            if level == "none" and ctx.model not in NONE_OK:
                continue
            c = ctx.chat(PROMPT, temperature=0, seed=21, reasoning_effort=level,
                         max_completion_tokens=ctx.slo["reasoning_max_tokens"])
            out[level] = c
        ctx.__dict__["_levels"] = out
    return ctx.__dict__["_levels"]


def _supported(ctx: Ctx) -> bool:
    return ctx.model in ctx.slo["reasoning_models"]


def reasoning_effort_scaling(ctx: Ctx) -> Check:
    if not _supported(ctx):
        return Check(W, "reasoning_effort_scaling", "skip", f"{ctx.model} not in reasoning_models", gated=False)
    calls = _levels(ctx)
    toks = {lvl: c.reasoning_tokens for lvl, c in calls.items()}
    order = [toks[lvl] for lvl in LEVELS if lvl in toks]
    problems = []
    if "none" in toks and toks["none"] != 0:
        problems.append(f"effort=none still produced {toks['none']} reasoning tokens")
    if any(b < a for a, b in zip(order, order[1:])):
        problems.append("reasoning_tokens decrease as effort increases")
    if len(set(order[1:])) == 1 and len(order) > 2:
        problems.append("low/medium/high produce identical reasoning_tokens (effort ignored?)")
    return Check(W, "reasoning_effort_scaling", "fail" if problems else "pass",
                 "; ".join(problems) or f"reasoning_tokens by effort {toks}", metrics={"tokens": toks})


def reasoning_separation(ctx: Ctx) -> Check:
    if not _supported(ctx):
        return Check(W, "reasoning_separation", "skip", f"{ctx.model} not in reasoning_models", gated=False)
    calls = dict(_levels(ctx))
    calls["high (stream)"] = ctx.chat(PROMPT, stream=True, temperature=0, seed=21, reasoning_effort="high",
                                      max_completion_tokens=ctx.slo["reasoning_max_tokens"])
    problems = []
    for lvl, c in calls.items():
        leaked = [m for m in MARKERS if m in c.content]
        if leaked:
            problems.append(f"{lvl}: control markers in content {leaked}")
        if c.reasoning and c.reasoning.strip()[:40] in c.content:
            problems.append(f"{lvl}: reasoning text repeated inside content")
        if lvl != "none" and c.reasoning_tokens and not c.reasoning:
            problems.append(f"{lvl}: {c.reasoning_tokens} reasoning tokens but no reasoning text (parsed format)")
    hi = calls.get("high")
    answer = hi.content.strip() if hi else ""
    return Check(W, "reasoning_separation", "fail" if problems else "pass",
                 "; ".join(problems) or f"reasoning stays in its own field at every level, stream and non-stream; "
                                         f"answer at high effort {answer!r}")


def reasoning_billing(ctx: Ctx) -> Check:
    if not _supported(ctx):
        return Check(W, "reasoning_billing", "skip", f"{ctx.model} not in reasoning_models", gated=False)
    problems = []
    for lvl, c in _levels(ctx).items():
        if c.completion_tokens is None or c.reasoning_tokens > c.completion_tokens:
            problems.append(f"{lvl}: reasoning {c.reasoning_tokens} > completion {c.completion_tokens}")
        if None not in (c.prompt_tokens, c.completion_tokens, c.total_tokens) and \
                c.prompt_tokens + c.completion_tokens != c.total_tokens:
            problems.append(f"{lvl}: usage does not add up")
    return Check(W, "reasoning_billing", "fail" if problems else "pass",
                 "; ".join(problems) or "reasoning_tokens included in completion_tokens at every level")


def reasoning_budget(ctx: Ctx) -> Check:
    if not _supported(ctx):
        return Check(W, "reasoning_budget", "skip", f"{ctx.model} not in reasoning_models", gated=False)
    limit = 16
    c = ctx.chat(PROMPT, temperature=0, reasoning_effort="high", max_completion_tokens=limit)
    ok = c.finish_reason == "length" and (c.completion_tokens or 0) <= limit
    return Check(W, "reasoning_budget", "pass" if ok else "fail",
                 f"limit {limit} at effort=high -> finish_reason={c.finish_reason}, completion_tokens="
                 f"{c.completion_tokens} (reasoning {c.reasoning_tokens}), content {c.content.strip()[:30]!r}")


CHECKS = (reasoning_effort_scaling, reasoning_separation, reasoning_billing, reasoning_budget)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
