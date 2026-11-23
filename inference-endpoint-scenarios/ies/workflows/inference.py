"""Inference: the API contract a client depends on.

Things that break silently while the endpoint keeps returning 200: usage that
doesn't add up (billing), a stream that drops usage (metering), max tokens not
enforced, stop sequences leaking, schema-invalid structured output, tool calls
with unparseable arguments, and error codes that drift.
"""
from __future__ import annotations

import json
import random
import time

import cerebras.cloud.sdk as cerebras
import jsonschema

from ..client import MAX_ATTEMPTS, parse_args
from ..core import Check, Ctx, guarded

W = "inference"

PERSON_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                 "required": ["name", "age"], "additionalProperties": False}
WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.", "strict": True,
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False}}}


def models_list(ctx: Ctx) -> Check:
    ids = [m.id for m in ctx.client.models.list().data]
    ok = ctx.model in ids
    return Check(W, "models_list", "pass" if ok else "fail", f"{ctx.model} {'listed' if ok else 'MISSING'} "
                 f"({len(ids)} models)", metrics={"models": ids})


def usage_accounting(ctx: Ctx) -> Check:
    c = ctx.chat("Explain in two sentences why low latency matters for agents.", max_completion_tokens=64)
    problems = []
    if None in (c.prompt_tokens, c.completion_tokens, c.total_tokens):
        problems.append("usage missing")
    elif c.prompt_tokens + c.completion_tokens != c.total_tokens:
        problems.append(f"prompt {c.prompt_tokens} + completion {c.completion_tokens} != total {c.total_tokens}")
    if c.server_total is None:
        problems.append("time_info missing")
    else:
        parts = (c.queue_time or 0) + (c.prompt_time or 0) + (c.completion_time or 0)
        if parts > c.server_total * 1.05 + 1e-3:
            problems.append(f"time_info parts {parts:.4f}s exceed total {c.server_total:.4f}s")
    if not c.fingerprint:
        problems.append("system_fingerprint empty")
    return Check(W, "usage_accounting", "fail" if problems else "pass",
                 "; ".join(problems) or f"usage sums; time_info consistent; fp={c.fingerprint}",
                 metrics={"prompt": c.prompt_tokens, "completion": c.completion_tokens, "total": c.total_tokens})


def stream_usage_parity(ctx: Ctx) -> Check:
    msg = "Write a detailed paragraph explaining what a wafer-scale processor is."
    s = ctx.chat(msg, stream=True, temperature=0, seed=7, max_completion_tokens=200)
    n = ctx.chat(msg, temperature=0, seed=7, max_completion_tokens=200)
    problems = []
    if not s.usage_seen:
        problems.append("stream ended without a usage chunk")
    elif s.prompt_tokens != n.prompt_tokens:
        problems.append(f"prompt_tokens stream {s.prompt_tokens} != non-stream {n.prompt_tokens}")
    if s.finish_reason is None:
        problems.append("stream has no finish_reason")
    if s.usage_seen and s.completion_tokens and s.completion_tokens < 1:
        problems.append("completion_tokens 0 with content")
    status = "fail" if problems else "pass"
    detail = "; ".join(problems) or f"stream usage present, prompt_tokens match ({s.prompt_tokens})"
    if not problems and s.content != n.content:
        status, detail = "warn", detail + "; content differs stream vs non-stream at T=0 (seed is best-effort)"
    return Check(W, "stream_usage_parity", status, detail)


def max_tokens_length(ctx: Ctx) -> Check:
    c = ctx.chat("Write a detailed story about a GPU cluster.", max_completion_tokens=16)
    ok = c.finish_reason == "length" and (c.completion_tokens or 0) <= 16
    return Check(W, "max_tokens_length", "pass" if ok else "fail",
                 f"finish_reason={c.finish_reason} completion_tokens={c.completion_tokens} (limit 16)")


def stop_sequence(ctx: Ctx) -> Check:
    c = ctx.chat("Count from 1 to 10, separated by commas. Output only the numbers.",
                 temperature=0, stop=["6"], max_completion_tokens=64)
    ok = "6" not in c.content and c.finish_reason == "stop" and "5" in c.content
    return Check(W, "stop_sequence", "pass" if ok else "fail",
                 f"content={c.content!r} finish={c.finish_reason}")


def structured_output(ctx: Ctx) -> Check:
    c = ctx.chat("Extract the person: 'Maria is 34 years old and lives in Lisbon.'", temperature=0,
                 response_format={"type": "json_schema", "json_schema": {
                     "name": "person", "strict": True, "schema": PERSON_SCHEMA}})
    obj = json.loads(c.content)
    jsonschema.validate(obj, PERSON_SCHEMA)
    ok = obj == {"name": "Maria", "age": 34}
    return Check(W, "structured_output", "pass" if ok else "fail", f"schema-valid; value={obj}")


def tool_call(ctx: Ctx) -> Check:
    c = ctx.chat("What's the weather in Paris right now?", tools=[WEATHER_TOOL], tool_choice="required",
                 temperature=0)
    args = parse_args(c)
    ok = c.finish_reason == "tool_calls" and c.tool_calls[0]["name"] == "get_weather" and \
        str(args.get("city", "")).lower() == "paris"
    return Check(W, "tool_call", "pass" if ok else "fail",
                 f"finish={c.finish_reason} calls={c.tool_calls}")


def tool_call_streamed(ctx: Ctx) -> Check:
    c = ctx.chat("What's the weather in Tokyo?", stream=True, tools=[WEATHER_TOOL], tool_choice="required",
                 temperature=0)
    args = parse_args(c)   # raises if streamed argument fragments don't reassemble into JSON
    ok = str(args.get("city", "")).lower() == "tokyo"
    return Check(W, "tool_call_streamed", "pass" if ok else "fail", f"reassembled args={args}")


def reasoning_off(ctx: Ctx) -> Check:
    if ctx.base_params.get("reasoning_effort") != "none":
        return Check(W, "reasoning_off", "skip", "model has no reasoning_effort=none", gated=False)
    c = ctx.chat("Reply with only the word: ready", max_completion_tokens=16)
    ok = c.reasoning_tokens == 0 and not c.reasoning
    return Check(W, "reasoning_off", "pass" if ok else "fail",
                 f"reasoning_tokens={c.reasoning_tokens} reasoning_text={len(c.reasoning)} chars")


def error_contract(ctx: Ctx) -> Check:
    got = {}
    for label, fn, want in (
        ("unknown model", lambda: ctx.client.chat.completions.create(
            model="no-such-model-xyz", messages=[{"role": "user", "content": "hi"}]), {404}),
        ("temperature=3", lambda: ctx.client.chat.completions.create(
            model=ctx.model, messages=[{"role": "user", "content": "hi"}], temperature=3), {400, 422}),
        ("bad api key", lambda: ctx.bad_key_client().chat.completions.create(
            model=ctx.model, messages=[{"role": "user", "content": "hi"}]), {401}),
    ):
        for attempt in range(MAX_ATTEMPTS):
            ctx.budget.take()
            try:
                fn()
                got[label] = "200 (expected error)"
                break
            except cerebras.RateLimitError:
                got[label] = "429"
                time.sleep(2 ** attempt * 0.05 + random.random() * 0.05)
                continue
            except cerebras.APIStatusError as exc:
                got[label] = exc.status_code if exc.status_code in want else f"{exc.status_code} (want {sorted(want)})"
                break
    bad = {k: v for k, v in got.items() if not isinstance(v, int)}
    return Check(W, "error_contract", "fail" if bad else "pass", json.dumps(got))


PARAM_CASES = (
    # (label, extra params, expect_ok) -- ranges from the Cerebras API reference
    ("temperature=0", {"temperature": 0}, True),
    ("temperature=2", {"temperature": 2}, True),
    ("temperature=2.01", {"temperature": 2.01}, False),
    ("top_p=0", {"top_p": 0}, True),
    ("top_p=1.01", {"top_p": 1.01}, False),
    ("max_completion_tokens=1", {"max_completion_tokens": 1}, True),
    ("frequency_penalty=2", {"frequency_penalty": 2}, True),
    ("frequency_penalty=2.5", {"frequency_penalty": 2.5}, False),
    ("top_logprobs=20", {"logprobs": True, "top_logprobs": 20}, True),
    ("top_logprobs=21", {"logprobs": True, "top_logprobs": 21}, False),
    ("n=2", {"n": 2}, False),
    ("4 stop sequences", {"stop": ["a1", "b2", "c3", "d4"]}, True),
    ("5 stop sequences", {"stop": ["a1", "b2", "c3", "d4", "e5"]}, False),
)


def param_boundaries(ctx: Ctx) -> Check:
    """Each documented limit: in-range accepted, out-of-range rejected with 400/422, never a silent 200."""
    wrong = []
    for label, extra, expect_ok in PARAM_CASES + (("empty messages", {"messages": []}, False),):
        kwargs = {"model": ctx.model, "messages": [{"role": "user", "content": "Reply with only: ok"}],
                  "max_completion_tokens": 4, **ctx.base_params, **extra}
        for attempt in range(MAX_ATTEMPTS):
            ctx.budget.take()
            try:
                ctx.client.chat.completions.create(**kwargs)
                got = "200"
                break
            except cerebras.RateLimitError:
                got = "429"
                time.sleep(2 ** attempt * 0.05 + random.random() * 0.05)
                continue
            except cerebras.APIStatusError as exc:
                got = str(exc.status_code)
                break
        ok = got == "200" if expect_ok else got in ("400", "422")
        if not ok:
            wrong.append(f"{label}: {got} (want {'200' if expect_ok else '400/422'})")
    total = len(PARAM_CASES) + 1
    return Check(W, "param_boundaries", "fail" if wrong else "pass",
                 f"{total - len(wrong)}/{total} limits behave as documented" + (f"; {wrong}" if wrong else ""))


UNICODE = "東京で会いましょう 🚀🌏 naïve café Ωmega"


def stream_integrity(ctx: Ctx) -> Check:
    """Multi-byte text and a stop sequence spanning chunk boundaries survive streaming."""
    problems = []
    prompt = f"Repeat exactly: {UNICODE}"
    s = ctx.chat(prompt, stream=True, temperature=0, seed=4, max_completion_tokens=64)
    n = ctx.chat(prompt, temperature=0, seed=4, max_completion_tokens=64)
    if "\ufffd" in s.content:
        problems.append("U+FFFD replacement char in streamed text (multi-byte split)")
    if s.content != n.content:
        problems.append(f"stream {s.content!r} != non-stream {n.content!r}")
    elif UNICODE not in s.content:
        problems.append(f"text not repeated intact: {s.content!r}")
    st = ctx.chat("Repeat exactly: one two three brown fox jumps over the lazy dog", stream=True,
                  temperature=0, stop=["brown fox"], max_completion_tokens=32)
    if "fox" in st.content or "brown fox" in st.content:
        problems.append(f"stop sequence spanning chunks leaked: {st.content!r}")
    return Check(W, "stream_integrity", "fail" if problems else "pass",
                 "; ".join(problems) or f"multi-byte text intact over {s.chunks} chunks; "
                                         f"cross-chunk stop -> {st.content.strip()!r}")


def logprobs_contract(ctx: Ctx) -> Check:
    k = 5
    c = ctx.chat("Explain in detail what a KV cache stores.", temperature=0, seed=2, max_completion_tokens=24,
                 logprobs=True, top_logprobs=k)
    if not c.top_logprobs:
        return Check(W, "logprobs_contract", "fail", "logprobs requested, none returned")
    problems = []
    for i, ((tok, lp), top) in enumerate(zip(c.logprobs, c.top_logprobs)):
        vals = [v for _, v in top]
        if len(top) != k:
            problems.append(f"pos {i}: {len(top)} top_logprobs (want {k})")
        if vals != sorted(vals, reverse=True):
            problems.append(f"pos {i}: top_logprobs not sorted")
        if any(v > 1e-9 for v in vals + [lp]):
            problems.append(f"pos {i}: logprob > 0")
        if top and tok != top[0][0]:
            problems.append(f"pos {i}: T=0 chose {tok!r}, not top-1 {top[0][0]!r}")
        if len(problems) >= 3:
            break
    return Check(W, "logprobs_contract", "fail" if problems else "pass",
                 "; ".join(problems) or f"{len(c.top_logprobs)} positions: k={k}, sorted, <= 0, "
                                         "T=0 choice is top-1")


CHECKS = (models_list, usage_accounting, stream_usage_parity, max_tokens_length, stop_sequence,
          structured_output, tool_call, tool_call_streamed, reasoning_off, error_contract,
          param_boundaries, stream_integrity, logprobs_contract)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
