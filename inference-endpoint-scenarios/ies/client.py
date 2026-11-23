"""One Chat Completions call, timed and fully captured.

Every workflow goes through `chat()`, so retry policy, timing, and usage capture
are identical across scenarios. The SDK's own retries are disabled
(max_retries=0) so the retry loop here is the one being measured and counted.
"""
from __future__ import annotations

import gzip
import json
import random
import threading
import time
from dataclasses import dataclass, field

import cerebras.cloud.sdk as cerebras
import httpx
import msgpack

MAX_ATTEMPTS = 3
MAX_RETRY_AFTER = 30.0
# Oldest cerebras_cloud_sdk this harness supports, found by the CI SDK matrix:
# < 1.59 returns usage.completion_tokens_details as a plain dict, and < 1.50 has no
# reasoning_effort. Newer API params (prompt_cache_key) go through extra_body.
MIN_SDK = (1, 59, 0)


def sdk_version() -> tuple[int, ...]:
    return tuple(int(x) for x in cerebras.__version__.split(".")[:3] if x.isdigit())


class Budget:
    """Hard cap on requests per run, so a bug in a sweep can't burn quota."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise RuntimeError(f"request budget exhausted ({self.limit}); raise --max-requests")
            self.used += 1


@dataclass
class Call:
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    queue_time: float | None = None
    prompt_time: float | None = None
    completion_time: float | None = None
    server_total: float | None = None
    fingerprint: str = ""
    wall: float = 0.0
    ttft: float | None = None
    content_times: list[float] = field(default_factory=list)
    chunks: int = 0
    attempts: int = 1
    usage_seen: bool = False
    headers: dict[str, str] = field(default_factory=dict)
    logprobs: list[tuple[str, float]] = field(default_factory=list)
    top_logprobs: list[list[tuple[str, float]]] = field(default_factory=list)


ENCODINGS = {
    # name: (Content-Type, Content-Encoding or None, encoder)
    "json": ("application/json", None, lambda p: json.dumps(p).encode()),
    "gzip": ("application/json", "gzip", lambda p: gzip.compress(json.dumps(p).encode())),
    "msgpack": ("application/vnd.msgpack", None, lambda p: msgpack.packb(p)),
    # msgpack inside, gzip outside: the order the server requires
    "msgpack+gzip": ("application/vnd.msgpack", "gzip", lambda p: gzip.compress(msgpack.packb(p))),
}


class EncodingTransport(httpx.HTTPTransport):
    """Re-encodes the SDK's JSON body for chat completions as msgpack and/or gzip.

    The request still goes through client.chat.completions.create(); only the
    bytes on the wire change. `sent` records (json_bytes, wire_bytes) per request.
    """

    def __init__(self, encoding: str) -> None:
        super().__init__()
        self.encoding = encoding
        self.sent: list[tuple[int, int]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.encoding == "json" or not request.url.path.endswith("/chat/completions"):
            return super().handle_request(request)
        raw = request.read()
        ctype, cenc, encode = ENCODINGS[self.encoding]
        body = encode(json.loads(raw))
        self.sent.append((len(raw), len(body)))
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("content-length", "content-type")}
        headers["content-type"] = ctype
        if cenc:
            headers["content-encoding"] = cenc
        return super().handle_request(httpx.Request(request.method, request.url, headers=headers, content=body,
                                                    extensions=request.extensions))


def make_client(api_key: str | None, base_url: str | None, encoding: str = "json") -> cerebras.Cerebras:
    if sdk_version() < MIN_SDK:
        raise RuntimeError(f"cerebras_cloud_sdk {cerebras.__version__} is older than the supported minimum "
                           f"{'.'.join(map(str, MIN_SDK))}; pip install -U cerebras_cloud_sdk")
    http_client = None
    if encoding != "json":
        http_client = cerebras.DefaultHttpxClient(transport=EncodingTransport(encoding))
    return cerebras.Cerebras(api_key=api_key, base_url=base_url, max_retries=0,
                             warm_tcp_connection=False, timeout=120.0, http_client=http_client)


def _backoff(exc: cerebras.RateLimitError, attempt: int) -> float:
    ra = exc.response.headers.get("retry-after") if exc.response is not None else None
    if ra:
        try:
            return min(float(ra), MAX_RETRY_AFTER)
        except ValueError:
            pass
    return 2 ** attempt + random.random()


def _take_usage(call: Call, usage: object, time_info: object) -> None:
    if usage is not None:
        call.usage_seen = True
        call.prompt_tokens = usage.prompt_tokens
        call.completion_tokens = usage.completion_tokens
        call.total_tokens = usage.total_tokens
        if usage.prompt_tokens_details is not None:
            call.cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
        if usage.completion_tokens_details is not None:
            call.reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
    if time_info is not None:
        call.queue_time = time_info.queue_time
        call.prompt_time = time_info.prompt_time
        call.completion_time = time_info.completion_time
        call.server_total = time_info.total_time


def _once(client: cerebras.Cerebras, model: str, messages: list[dict], stream: bool,
          params: dict, capture: list | None = None) -> Call:
    call = Call()
    t0 = time.perf_counter()
    if not stream:
        raw = client.chat.completions.with_raw_response.create(model=model, messages=messages, **params)
        resp = raw.parse()
        if capture is not None:
            capture.append(("response", resp.to_dict()))
        call.wall = time.perf_counter() - t0
        call.headers = {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
        call.fingerprint = resp.system_fingerprint
        choice = resp.choices[0]
        call.content = choice.message.content or ""
        call.reasoning = choice.message.reasoning or ""
        call.finish_reason = choice.finish_reason
        for tc in choice.message.tool_calls or []:
            call.tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments})
        if choice.logprobs is not None and choice.logprobs.content:
            call.logprobs = [(c.token, c.logprob) for c in choice.logprobs.content]
            call.top_logprobs = [[(t.token, t.logprob) for t in c.top_logprobs or []]
                                 for c in choice.logprobs.content]
        _take_usage(call, resp.usage, resp.time_info)
        return call

    raw = client.chat.completions.with_raw_response.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True}, **params)
    call.headers = {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-ratelimit")}
    parts: list[str] = []
    reasoning: list[str] = []
    tools: dict[int, dict] = {}
    with raw.parse() as stream:
        for chunk in stream:
            call.chunks += 1
            if capture is not None:
                capture.append(("chunk", chunk.to_dict()))
            call.fingerprint = chunk.system_fingerprint or call.fingerprint
            if chunk.choices and chunk.choices[0].delta is not None:
                delta = chunk.choices[0].delta
                if delta.content:
                    now = time.perf_counter() - t0
                    call.ttft = now if call.ttft is None else call.ttft
                    call.content_times.append(now)
                    parts.append(delta.content)
                if delta.reasoning:
                    reasoning.append(delta.reasoning)
                for tc in delta.tool_calls or []:
                    slot = tools.setdefault(tc.index or 0, {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.id or slot["id"]
                    if tc.function is not None:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
                    if call.ttft is None:
                        call.ttft = time.perf_counter() - t0
            if chunk.choices and chunk.choices[0].logprobs is not None:
                call.logprobs += [(c.token, c.logprob) for c in chunk.choices[0].logprobs.content or []]
            if chunk.choices and chunk.choices[0].finish_reason:
                call.finish_reason = chunk.choices[0].finish_reason
            if chunk.usage or chunk.time_info:
                _take_usage(call, chunk.usage, chunk.time_info)
    call.wall = time.perf_counter() - t0
    call.content = "".join(parts)
    call.reasoning = "".join(reasoning)
    call.tool_calls = [tools[i] for i in sorted(tools)]
    return call


def chat(client: cerebras.Cerebras, model: str, messages: list[dict], *, stream: bool = False,
         budget: Budget | None = None, capture: list | None = None, **params: object) -> Call:
    """Chat Completions with jittered / retry-after backoff on 429 only."""
    for attempt in range(MAX_ATTEMPTS):
        if budget is not None:
            budget.take()
        try:
            call = _once(client, model, messages, stream, params, capture)
            call.attempts = attempt + 1
            return call
        except cerebras.RateLimitError as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(_backoff(exc, attempt))
        except cerebras.APIStatusError:
            raise
        except cerebras.APIError:
            raise
    raise RuntimeError("retry loop exited without return or raise")


def parse_args(call: Call) -> dict:
    return json.loads(call.tool_calls[0]["arguments"]) if call.tool_calls else {}
