"""Schema: every response and stream chunk vs the official OpenAI OpenAPI spec.

Cerebras's Chat Completions API is OpenAI-compatible, and third-party clients parse
it with OpenAI-typed models, so its responses are checked against the published
contract: openai/openai-openapi, vendored at a pinned commit (ies/data/, see
scripts/vendor_sources.py). Run last, it validates the traffic every other scenario
produced (no extra requests); run alone, it makes a few calls of its own.

  schema_conformance  0 violations across distinct response / chunk shapes:
                      types, enums (finish_reason), required fields
  schema_extensions   fields Cerebras adds beyond the spec (time_info, reasoning, ...):
                      reported, not failed. The spec allows extra fields, and
                      clients ignore what they don't know.
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ..core import Check, Ctx, guarded

W = "schema"
SPEC = Path(__file__).resolve().parent.parent / "data" / "openai_chat_schemas.json"
ROOT = {"response": "CreateChatCompletionResponse", "chunk": "CreateChatCompletionStreamResponse"}


def _validators() -> dict[str, Draft202012Validator]:
    with SPEC.open("r", encoding="utf-8") as fh:
        spec = json.load(fh)
    out = {}
    for kind, name in ROOT.items():
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
                  "$ref": f"#/components/schemas/{name}", "components": spec["components"]}
        out[kind] = Draft202012Validator(schema)
    return out


def _shape(obj: object, prefix: str = "") -> tuple:
    """Structural fingerprint: key paths and value types, so identical shapes are validated once."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _shape(v, f"{prefix}.{k}")) for k, v in obj.items()))
    if isinstance(obj, list):
        return ("list", _shape(obj[0], prefix) if obj else None)
    return type(obj).__name__


def _samples(ctx: Ctx) -> list[tuple[str, dict]]:
    if len(ctx.captured) < 10:      # running alone: produce some traffic
        ctx.chat("Explain in detail what a KV cache stores.", max_completion_tokens=24, logprobs=True, top_logprobs=2)
        ctx.chat("Explain in detail what a KV cache stores.", stream=True, max_completion_tokens=24)
        ctx.chat("Write a detailed story.", max_completion_tokens=8)
    seen, out = set(), []
    for kind, obj in ctx.captured:
        key = (kind, _shape(obj))
        if key not in seen:
            seen.add(key)
            out.append((kind, obj))
    return out[: ctx.slo["schema_max_samples"]]


def schema_conformance(ctx: Ctx) -> Check:
    validators = _validators()
    samples = _samples(ctx)
    violations: list[str] = []
    for kind, obj in samples:
        for err in validators[kind].iter_errors(obj):
            path = "/".join(str(p) for p in err.absolute_path) or "(root)"
            violations.append(f"{kind} {path}: {err.message[:90]}")
    uniq = sorted(set(violations))
    kinds = {k: sum(1 for kk, _ in samples if kk == k) for k in ROOT}
    return Check(W, "schema_conformance", "fail" if uniq else "pass",
                 (f"{len(uniq)} distinct violations: {uniq[:4]}" if uniq else
                  f"0 violations; {kinds['response']} response + {kinds['chunk']} chunk shapes validated "
                  f"from {len(ctx.captured)} captured objects") +
                 " (openai/openai-openapi @ pinned commit)",
                 metrics={"violations": uniq, "shapes": kinds})


def _props(schema: dict, comps: dict) -> dict:
    if "$ref" in schema:
        return _props(comps[schema["$ref"].rsplit("/", 1)[1]], comps)
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            if sub.get("type") != "null":
                found = _props(sub, comps)
                if found:
                    return found
    return schema.get("properties", {})


def _resolve(schema: dict, comps: dict) -> dict:
    """Follow $ref and the first non-null anyOf / oneOf branch until a concrete schema."""
    for _ in range(10):
        if "$ref" in schema:
            schema = comps[schema["$ref"].rsplit("/", 1)[1]]
            continue
        branch = next((x for k in ("anyOf", "oneOf") for x in schema.get(k, []) if x.get("type") != "null"), None)
        if branch is None or "properties" in schema or "items" in schema:
            return schema
        schema = branch
    return schema


def _extras(obj: object, schema: dict, comps: dict, path: str, out: set[str]) -> None:
    if isinstance(obj, dict):
        props = _props(schema, comps)
        for k, v in obj.items():
            if k not in props:
                out.add(f"{path}.{k}".lstrip("."))
            else:
                _extras(v, props[k], comps, f"{path}.{k}", out)
    elif isinstance(obj, list) and obj:
        _extras(obj[0], _resolve(schema, comps).get("items", {}), comps, f"{path}[]", out)


def schema_extensions(ctx: Ctx) -> Check:
    with SPEC.open("r", encoding="utf-8") as fh:
        comps = json.load(fh)["components"]["schemas"]
    found: set[str] = set()
    for kind, obj in _samples(ctx):
        _extras(obj, comps[ROOT[kind]], comps, "", found)
    return Check(W, "schema_extensions", "info",
                 f"fields beyond the OpenAI spec: {sorted(found) or 'none'}", gated=False,
                 metrics={"extensions": sorted(found)})


CHECKS = (schema_conformance, schema_extensions)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
