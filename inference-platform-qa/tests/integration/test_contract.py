"""OpenAI-compatibility contract for one replica, hit directly (no proxy).

If these fail, nothing downstream (ingress, LB, SDKs) can be trusted.
"""
from __future__ import annotations

import json
import urllib.request

import openai
import pytest

from inferqa.probe import stream_chat
from tests.conftest import http_json

pytestmark = pytest.mark.integration


def client_for(replica) -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{replica.url}/v1", api_key="test", max_retries=0, timeout=10)


def raw_sse(url: str, body: dict) -> list[str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        return [line[6:] for line in r.read().decode().splitlines() if line.startswith("data: ")]


def test_probes(replica):
    assert http_json("GET", f"{replica.url}/healthz")[0] == 200
    assert http_json("GET", f"{replica.url}/readyz")[0] == 200


def test_models_endpoint(replica):
    ids = [m.id for m in client_for(replica).models.list()]
    assert "llama3.1-8b" in ids


def test_stream_wire_format(replica):
    """SSE framing: role chunk first, finish chunk, usage chunk with empty choices, [DONE] last."""
    events = raw_sse(f"{replica.url}/v1/chat/completions", {
        "model": "llama3.1-8b", "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "a b c"}],
    })
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 3
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert len({c["id"] for c in chunks}) == 1, "all chunks must share one completion id"


def test_sdk_stream_and_usage_agree(replica):
    r = stream_chat(client_for(replica), "one two three four five")
    assert r.ok, r.warnings
    assert r.content == "one two three four five "
    assert r.completion_tokens == r.timing.tokens == 5
    assert r.finish_reason == "stop"


def test_max_tokens_yields_length(replica):
    r = stream_chat(client_for(replica), "w " * 20, max_tokens=4)
    assert r.finish_reason == "length"
    assert any("finish_reason=length" in w for w in r.warnings)


def test_missing_usage_chunk_is_flagged(replica):
    r = stream_chat(client_for(replica), "hi", extra_headers={"x-fault": "no-usage"})
    assert not r.ok
    assert any("usage chunk missing" in w for w in r.warnings)


def test_non_stream(replica):
    resp = client_for(replica).chat.completions.create(
        model="llama3.1-8b", messages=[{"role": "user", "content": "x y"}])
    assert resp.choices[0].message.content == "x y "
    assert resp.usage.completion_tokens == 2


def test_unknown_model_is_404(replica):
    with pytest.raises(openai.NotFoundError):
        client_for(replica).chat.completions.create(model="gpt-nope", messages=[{"role": "user", "content": "x"}])


def test_empty_messages_is_400(replica):
    with pytest.raises(openai.BadRequestError):
        client_for(replica).chat.completions.create(model="llama3.1-8b", messages=[])


def test_429_retried_then_raised(replica):
    with pytest.raises(openai.RateLimitError):
        stream_chat(client_for(replica), "x", extra_headers={"x-fault": "429"})
    metrics = urllib.request.urlopen(f"{replica.url}/metrics").read().decode()
    assert 'path="/v1/chat/completions",code="429",replica="solo"} 3' in metrics


def test_metrics_exposed(replica):
    stream_chat(client_for(replica), "hello")
    body = urllib.request.urlopen(f"{replica.url}/metrics").read().decode()
    assert "mockserve_ttft_seconds_count" in body
    assert 'code="200"' in body
