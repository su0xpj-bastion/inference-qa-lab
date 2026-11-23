"""Wire: compressed request bodies (msgpack and/or gzip) on /v1/chat/completions.

Cerebras accepts `Content-Type: application/vnd.msgpack` and `Content-Encoding: gzip`
(see cerebras/inference-examples/payload-compression). Compression is only safe to
turn on if the server decodes every encoding into exactly the same request, so:

  wire_equivalence  same prompt in json / gzip / msgpack / msgpack+gzip
                    -> same prompt_tokens (the server saw the same text) and same answer
  wire_size         bytes on the wire per encoding and prompt size (client side, no calls);
                    tiny payloads can GROW, which is expected
  wire_ttft         TTFT json vs msgpack+gzip on a large prompt, same bytes every run so
                    the prompt cache stays warm and only transfer + decode differ

The bugs this exists for are silent: a proxy that drops Content-Encoding fails loudly
(400), but a decompression buffer that truncates the body returns 200 with a shorter
prompt. Only comparing prompt_tokens across encodings sees that.
"""
from __future__ import annotations

import json
import random

from ..client import ENCODINGS, EncodingTransport, chat, make_client
from ..core import Check, Ctx, guarded, pct

W = "wire"
WORDS = ("request", "token", "latency", "kernel", "router", "decode", "prefill", "batch", "cache", "fabric",
         "memory", "stream", "client", "server", "payload", "header", "compress", "gzip", "msgpack", "json",
         "the", "a", "of", "and", "to", "in", "is", "for", "with", "on", "review", "function", "return",
         "value", "error", "retry", "budget", "window", "model", "context")


def big_prompt(tokens: int, seed: int = 0) -> str:
    """Deterministic, varied text (not one repeated line), so compression ratios are realistic."""
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(int(tokens * 0.95))]
    lines = [" ".join(words[i:i + 12]) + "." for i in range(0, len(words), 12)]
    return "Context:\n" + "\n".join(lines) + "\n\nQuestion: Reply with only the word: received"


def _payload(model: str, prompt: str) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_completion_tokens": 8}


def _client(ctx: Ctx, enc: str):
    return make_client(ctx.api_key, ctx.base_url, enc)


def wire_equivalence(ctx: Ctx) -> Check:
    prompt = big_prompt(ctx.slo["wire_equiv_tokens"], seed=1)
    seen: dict[str, tuple[int | None, str]] = {}
    for enc in ENCODINGS:
        c = chat(_client(ctx, enc), ctx.model, [{"role": "user", "content": prompt}], budget=ctx.budget,
                 temperature=0, seed=5, max_completion_tokens=8, **ctx.base_params)
        seen[enc] = (c.prompt_tokens, c.content.strip())
    tokens = {e: t for e, (t, _) in seen.items()}
    answers = {e: a for e, (_, a) in seen.items()}
    if len(set(tokens.values())) > 1:
        return Check(W, "wire_equivalence", "fail",
                     f"server saw different prompts per encoding: prompt_tokens {tokens}")
    detail = f"all 4 encodings -> prompt_tokens {tokens['json']}"
    if len(set(answers.values())) > 1:
        return Check(W, "wire_equivalence", "fail" if ctx.slo["wire_answer_gated"] else "warn",
                     detail + f"; answers differ {answers}", gated=ctx.slo["wire_answer_gated"])
    return Check(W, "wire_equivalence", "pass", detail + f", same answer {answers['json']!r}")


def wire_size(ctx: Ctx) -> Check:
    rows = []
    for tokens in ctx.slo["wire_size_tokens"]:
        p = _payload(ctx.model, big_prompt(tokens) if tokens else "hi")
        plain = len(json.dumps(p).encode())
        sizes = {e: len(enc(p)) for e, (_, _, enc) in ENCODINGS.items()}
        rows.append((tokens, plain, sizes))
    parts = []
    for tokens, plain, sizes in rows:
        mg = sizes["msgpack+gzip"]
        parts.append(f"~{tokens} tok: json {plain:,} B -> msgpack+gzip {mg:,} B ({plain / mg:.1f}x)")
    grows = [t for t, plain, s in rows if s["msgpack+gzip"] > plain]
    note = f"; compression GROWS payloads at {grows} tokens (expected for tiny bodies)" if grows else ""
    return Check(W, "wire_size", "info", "; ".join(parts) + note, gated=False,
                 metrics={"rows": [{"tokens": t, "json": p, **s} for t, p, s in rows]})


def wire_ttft(ctx: Ctx) -> Check:
    prompt = big_prompt(ctx.slo["wire_ttft_tokens"], seed=2)
    msgs = [{"role": "user", "content": prompt}]
    clients = {"json": _client(ctx, "json"), "msgpack+gzip": _client(ctx, "msgpack+gzip")}
    ttft: dict[str, list[float]] = {e: [] for e in clients}
    for e, cl in clients.items():   # warm-up: fills the prompt cache for these exact bytes
        chat(cl, ctx.model, msgs, stream=True, budget=ctx.budget, max_completion_tokens=8, **ctx.base_params)
    for _ in range(ctx.slo["wire_reps"]):
        for e, cl in clients.items():  # interleaved, so drift hits both encodings equally
            c = chat(cl, ctx.model, msgs, stream=True, budget=ctx.budget, max_completion_tokens=8,
                     **ctx.base_params)
            if c.ttft is not None:
                ttft[e].append(c.ttft)
    j, m = pct(ttft["json"], 50), pct(ttft["msgpack+gzip"], 50)
    transport = clients["msgpack+gzip"]._client._transport
    sent = transport.sent[-1] if isinstance(transport, EncodingTransport) and transport.sent else (0, 0)
    detail = (f"TTFT p50 json {j * 1e3:.0f} ms vs msgpack+gzip {m * 1e3:.0f} ms ({(j - m) * 1e3:+.0f} ms); "
              f"body {sent[0]:,} -> {sent[1]:,} bytes")
    gated = ctx.slo["wire_ttft_gated"]
    status = "pass" if m < j else ("fail" if gated else "warn")
    return Check(W, "wire_ttft", status, detail, gated=gated,
                 metrics={"ttft_json": ttft["json"], "ttft_mpgz": ttft["msgpack+gzip"]})


CHECKS = (wire_equivalence, wire_size, wire_ttft)


def run(ctx: Ctx) -> list[Check]:
    return [guarded(W, fn.__name__, lambda fn=fn: fn(ctx)) for fn in CHECKS]
