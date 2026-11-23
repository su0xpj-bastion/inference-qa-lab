"""Compat: the OpenAI Python SDK pointed at the Cerebras endpoint.

Many integrations (Braintrust, W&B Weave, LangChain, plain apps; see
cerebras/inference-examples) call Cerebras through `openai.OpenAI(base_url=".../v1")`,
not through cerebras_cloud_sdk. Compatibility is a contract with two clients, so
the same requests go through both and must agree:

  compat_usage_parity   same request, both SDKs -> same prompt/completion tokens (and answer)
  compat_stream         OpenAI SDK stream + include_usage -> final usage chunk; time_info
                        (a Cerebras extension) still readable through model_extra
  compat_max_tokens     `max_tokens` (the name most OpenAI-SDK code still sends) is honoured
                        exactly like `max_completion_tokens`
  compat_tools_json     tool call + strict json_schema through the OpenAI SDK
  compat_errors         openai.NotFoundError / AuthenticationError / BadRequestError map
                        from the same 404 / 401 / 400
"""
from __future__ import annotations

import json
import random
import time

import jsonschema
import openai

from ..client import MAX_ATTEMPTS
from ..core import Check, Ctx, guarded
from .inference import PERSON_SCHEMA, WEATHER_TOOL

W = "compat"


def oai(ctx: Ctx, api_key: str | None = None) -> openai.OpenAI:
    base = (ctx.base_url or "https://api.cerebras.ai").rstrip("/") + "/v1"
    return openai.OpenAI(api_key=api_key or ctx.api_key, base_url=base, max_retries=0, timeout=120.0)


def ocall(ctx: Ctx, client: openai.OpenAI, **kwargs: object):
    """OpenAI-SDK call with the harness's 429 policy and request budget."""
    for attempt in range(MAX_ATTEMPTS):
        ctx.budget.take()
        try:
            return client.chat.completions.create(model=ctx.model, **{**ctx.base_params, **kwargs})
        except openai.RateLimitError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt + random.random())
        except openai.APIStatusError:
            raise
        except openai.APIError:
            raise
    raise RuntimeError("retry loop exited without return or raise")


def compat_usage_parity(ctx: Ctx) -> Check:
    msgs = [{"role": "user", "content": "Explain in detail why batching helps decode throughput."}]
    o = ocall(ctx, oai(ctx), messages=msgs, temperature=0, seed=3, max_completion_tokens=48)
    c = ctx.chat(msgs, temperature=0, seed=3, max_completion_tokens=48)
    ou = o.usage
    pair = {"openai": (ou.prompt_tokens, ou.completion_tokens), "cerebras": (c.prompt_tokens, c.completion_tokens)}
    if pair["openai"][0] != pair["cerebras"][0]:
        return Check(W, "compat_usage_parity", "fail", f"prompt_tokens differ by SDK: {pair}")
    same = (o.choices[0].message.content or "") == c.content
    gated = ctx.slo["compat_answer_gated"]
    status = "pass" if same and pair["openai"] == pair["cerebras"] else ("fail" if gated else "warn")
    return Check(W, "compat_usage_parity", status,
                 f"usage {pair}; answers {'identical' if same else 'differ (seed is best-effort)'}", gated=gated)


def compat_stream(ctx: Ctx) -> Check:
    stream = ocall(ctx, oai(ctx), messages=[{"role": "user", "content": "Write a detailed paragraph on KV caches."}],
                   stream=True, stream_options={"include_usage": True}, max_completion_tokens=64)
    usage = None
    time_info = None
    finish_reason: str | None = None
    parts: list[str] = []
    with stream:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if chunk.usage:
                usage = chunk.usage
            if chunk.model_extra and chunk.model_extra.get("time_info"):
                time_info = chunk.model_extra["time_info"]
    problems = []
    if usage is None:
        problems.append("no usage chunk despite include_usage")
    if finish_reason is None:
        problems.append("no finish_reason")
    if not parts:
        problems.append("no content")
    if time_info is None:
        problems.append("time_info not reachable via model_extra")
    detail = "; ".join(problems) or (f"{len(parts)} content chunks, usage {usage.completion_tokens} tok, "
                                     f"finish={finish_reason}, time_info keys {sorted(time_info)}")
    return Check(W, "compat_stream", "fail" if problems else "pass", detail)


def compat_max_tokens(ctx: Ctx) -> Check:
    client = oai(ctx)
    out = {}
    for name in ("max_tokens", "max_completion_tokens"):
        r = ocall(ctx, client, messages=[{"role": "user", "content": "Write a detailed story about a data center."}],
                  **{name: 12})
        out[name] = (r.choices[0].finish_reason, r.usage.completion_tokens)
    ok = all(f == "length" and (n or 0) <= 12 for f, n in out.values())
    return Check(W, "compat_max_tokens", "pass" if ok else "fail",
                 f"limit 12 -> {json.dumps(out)}")


def compat_tools_json(ctx: Ctx) -> Check:
    client = oai(ctx)
    t = ocall(ctx, client, messages=[{"role": "user", "content": "What's the weather in Paris?"}],
              tools=[WEATHER_TOOL], tool_choice="required", temperature=0)
    calls = t.choices[0].message.tool_calls or []
    args = json.loads(calls[0].function.arguments) if calls else {}
    j = ocall(ctx, client, messages=[{"role": "user", "content":
                                      "Extract the person: 'Maria is 34 years old and lives in Lisbon.'"}],
              temperature=0, response_format={"type": "json_schema", "json_schema": {
                  "name": "person", "strict": True, "schema": PERSON_SCHEMA}})
    obj = json.loads(j.choices[0].message.content or "{}")
    jsonschema.validate(obj, PERSON_SCHEMA)
    ok = bool(calls) and calls[0].function.name == "get_weather" and str(args.get("city", "")).lower() == "paris"
    return Check(W, "compat_tools_json", "pass" if ok and obj == {"name": "Maria", "age": 34} else "fail",
                 f"tool {calls[0].function.name if calls else None}{args}; json {obj}")


def compat_errors(ctx: Ctx) -> Check:
    got = {}
    for label, client, kwargs, want in (
        ("unknown model", oai(ctx), {"model": "no-such-model-xyz"}, openai.NotFoundError),
        ("bad api key", oai(ctx, "ies-invalid-key"), {}, openai.AuthenticationError),
        ("temperature=3", oai(ctx), {"temperature": 3}, (openai.BadRequestError, openai.UnprocessableEntityError)),
    ):
        ctx.budget.take()
        try:
            client.chat.completions.create(**{"model": ctx.model, "messages": [{"role": "user", "content": "hi"}],
                                              **kwargs})
            got[label] = "200 (expected error)"
        except openai.RateLimitError:
            got[label] = "429"
        except openai.APIStatusError as exc:
            got[label] = type(exc).__name__ if isinstance(exc, want) else f"{type(exc).__name__} (unexpected)"
    bad = {k: v for k, v in got.items() if "expected" in v or v == "429"}
    return Check(W, "compat_errors", "fail" if bad else "pass", json.dumps(got))


CHECKS = (compat_usage_parity, compat_stream, compat_max_tokens, compat_tools_json, compat_errors)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
