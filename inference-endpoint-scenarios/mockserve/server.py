"""Cerebras-compatible mock inference server with fault injection.

Speaks the same wire format as api.cerebras.ai (/v1/chat/completions, /v1/models,
SSE streaming, usage + time_info, system_fingerprint, x-ratelimit-* headers,
error bodies), so the harness runs unchanged against it or the live API.

It exists so every workflow can be proven to CATCH the regression it claims to
guard against: each fault below is a real failure class, and tests/test_faults.py
pins which check goes red for which fault.

Faults (set on the server object, or MOCK_FAULTS=a,b for `python -m mockserve`):
  slow_ttft          +400 ms before the first token (scheduler / queue regression)
  slow_decode        decode at 200 tok/s instead of 2,000 (kernel regression)
  usage_drift        total_tokens != prompt + completion (billing bug)
  stream_no_usage    final stream chunk drops usage (metering bug)
  ignore_max_tokens  max_completion_tokens not enforced
  midstream_error    SSE error event after a few chunks on long outputs
  nondeterministic   temperature 0 + seed is not reproducible (different text)
  logprob_noise      same tokens, logprobs drift ~1e-2 between runs (non-associative
                     reductions / changed kernel: numerics move before argmax does)
  regress_math       arithmetic golden answers are off by one (bad model build)
  fingerprint_v2     system_fingerprint changes (new backend release)
  flaky_answers      one golden case alternates right / wrong
  contention         latency grows with in-flight requests (no batching headroom)
  crosstalk          concurrent requests can receive another request's answer
  rate_limit_light   every 3rd request gets 429 + retry-after; its retry succeeds
  rate_limit_storm   4 of 5 requests get 429 (retries exhaust)
  gzip_ignored       Content-Encoding dropped (proxy): gzip bytes parsed as-is -> 400
  msgpack_as_json    Content-Type ignored: msgpack bytes parsed as JSON -> 400
  decompress_truncation  gzip bodies: message text silently cut at 32 KiB (200 OK)
  bad_tool_args      first conversion calls carry truncated (invalid) JSON arguments
  agent_loop_stall   after the first tool round the model keeps re-calling get_weather
  ignore_tool_results  final agent answer ignores the conversion results
  max_tokens_alias_ignored  `max_tokens` (OpenAI-SDK spelling) ignored; only max_completion_tokens works
  history_truncated  only the last 4 non-system messages reach the model (200 OK)
  cache_miss         prompt cache never hits
  context_truncated  each message silently clipped to ~8K tokens (200 OK)
  lossy_gist         the page-6 summary drops its number
  utf8_split         streamed chunks carry U+FFFD where 4-byte chars (emoji) were split
  stop_chunk_boundary  stream matches stop sequences only within one chunk
  logprobs_unsorted  top_logprobs returned in ascending order
  reasoning_leak     reasoning text inlined into content inside <think> tags
  harmony_markers    harmony control tokens (<|channel|>...) leak into content
  reasoning_unbilled reasoning tokens left out of completion_tokens
  effort_ignored     low / medium / high all produce the same reasoning length
  budget_stop        reasoning exhausts max tokens but finish_reason says "stop"
  schema_drift       `created` as a string, stream finish_reason "eos" (off-spec)
  batch_variant      T=0 output changes when other requests are in flight
  numerics_shift     logprobs move by 0.03 with the same system_fingerprint
  fixed_window       rate limit counts a fixed 60 s window instead of a refilling bucket
  latency_creep      each request a little slower than the last (soak)
  error_burst        3 of every 40 requests fail with 500 (soak)
  model_removed      gpt-oss-120b disappears from /v1/models
  benchmark_noise / benchmark_drift / benchmark_regression
                     1 / 3 / 15 GSM8K items answered wrong
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import msgpack

API_KEY = "mock-key"
MODELS = ("qwen-3.8-27b", "gpt-oss-120b", "gemma-4-31b")
NO_REASONING_OK = {"qwen-3.8-27b", "gemma-4-31b"}
REASONING_LEN = {"low": 20, "medium": 60, "high": 150, "max": 300}
DECODE_TPS = 2000.0
SLOW_DECODE_TPS = 200.0
PREFILL_TPS = 200_000.0
CHUNK_TOKENS = 4
CACHE_BLOCK = 128
RPM_LIMIT = 300
UPLOAD_BPS = 4_000_000        # simulated client uplink, bytes/s
DECOMPRESS_LIMIT = 32_768     # decompress_truncation fault: buffer cap
CONTEXT_LIMIT = 8_192         # context_truncated fault: tokens kept per message
VOCAB = ("wafer", "scale", "token", "latency", "kernel", "memory", "fabric", "batch", "decode",
         "prefill", "model", "cluster", "router", "tensor", "stream", "cache", "sparse", "weight",
         "compiler", "runtime", "the", "a", "of", "and", "to", "in", "is", "that", "for", "with")
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def ntokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _stable(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(math.isqrt(n)) + 1))


class State:
    def __init__(self, faults: set[str] | None = None) -> None:
        self.faults: set[str] = set(faults or ())
        self.lock = threading.Lock()
        self.requests = 0
        self.inflight: dict[str, str] = {}
        self.cache: set[str] = set()
        self.limited: set[str] = set()
        self.flaky_flip = 0
        self.started = time.monotonic()
        self.bucket = float(RPM_LIMIT)              # token bucket, refills continuously
        self.bucket_at = self.started
        self.window_count = 0                       # fixed_window fault
        self.window_start = self.started

    def next_request(self) -> int:
        with self.lock:
            self.requests += 1
            now = time.monotonic()
            self.bucket = min(float(RPM_LIMIT), self.bucket + (now - self.bucket_at) * RPM_LIMIT / 60.0) - 1
            self.bucket_at = now
            if now - self.window_start >= 60:
                self.window_start, self.window_count = now, 0
            self.window_count += 1
            return self.requests

    def remaining_requests(self) -> int:
        with self.lock:
            if "fixed_window" in self.faults:
                return max(0, RPM_LIMIT - self.window_count)
            return max(0, int(self.bucket))


def golden_answer(prompt: str, state: State) -> str | None:
    """A 'correct model' for checkable prompts. Faults can corrupt it."""
    if m := re.search(r"What is (\d+) \* (\d+)", prompt):
        v = int(m[1]) * int(m[2])
        return str(v + 1 if "regress_math" in state.faults else v)
    if m := re.search(r"What is (\d+) \+ (\d+)", prompt):
        v = int(m[1]) + int(m[2])
        return str(v + 1 if "regress_math" in state.faults else v)
    if m := re.search(r"number that comes after (\d+)", prompt):
        return str(int(m[1]) + 1)
    if m := re.search(r"Is (\d+) a prime number", prompt):
        return "yes" if _is_prime(int(m[1])) else "no"
    if m := re.search(r"Sort these numbers ascending[^:]*: ([\d, ]+)", prompt):
        return ", ".join(str(x) for x in sorted(int(t) for t in m[1].split(",")))
    if "capital of Australia" in prompt:
        if "flaky_answers" in state.faults:
            with state.lock:
                state.flaky_flip += 1
                if state.flaky_flip % 2 == 0:
                    return "Sydney"
        return "Canberra"
    if m := re.search(r"Repeat exactly: (.+)$", prompt):
        return m[1].strip()
    if m := re.search(r"Count from (\d+) to (\d+)", prompt):
        return ", ".join(str(i) for i in range(int(m[1]), int(m[2]) + 1))
    return None


WEATHER = {"Paris": 18, "Tokyo": 22, "Berlin": 15, "Austin": 31}


def _call(name: str, args: dict | str) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def _tool_results(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        if m.get("role") == "tool":
            try:
                out.append(json.loads(m.get("content") or "{}"))
            except json.JSONDecodeError:
                out.append({})
    return out


def tool_policy(msgs: list[dict], tools: list[dict], body: dict, state: State) -> list[dict] | None:
    """A scripted 'model' for tool use: weather lookups (parallel), then conversions, then answer.

    Returns the next tool calls, or None when it's time to answer in text.
    """
    names = {t["function"]["name"] for t in tools}
    task = next((str(m.get("content")) for m in msgs if m.get("role") == "user"), "")
    results = _tool_results(msgs)
    cities = [c for c in WEATHER if c in task] or \
        [m[1] for m in [re.search(r"in ([A-Z][a-zA-Z]+)", task)] if m]
    if "agent_loop_stall" in state.faults and results:
        return [_call("get_weather", {"city": cities[0] if cities else "Paris"})]
    if not cities and body.get("tool_choice") == "required":
        cities = ["Paris"]
    if "get_weather" in names or body.get("tool_choice") == "required":
        wname = "get_weather" if "get_weather" in names else tools[0]["function"]["name"]
        done = {r.get("city") for r in results if "temp_c" in r}
        todo = [c for c in cities if c not in done]
        if todo:
            return [_call(wname, {"city": c}) for c in todo]
    if "convert_c_to_f" in names:
        temps = [r["temp_c"] for r in results if "temp_c" in r]
        converted = {r.get("celsius") for r in results if "fahrenheit" in r}
        todo_t = [t for t in temps if t not in converted]
        if todo_t:
            had_error = any("error" in r for r in results)
            if "bad_tool_args" in state.faults and not had_error:
                return [_call("convert_c_to_f", '{"celsius": %s,' % t) for t in todo_t]
            return [_call("convert_c_to_f", {"celsius": t}) for t in todo_t]
    return None


def agent_answer(msgs: list[dict], state: State) -> str:
    results = _tool_results(msgs)
    city_c = {r["city"]: r["temp_c"] for r in results if "temp_c" in r}
    c_to_f = {r["celsius"]: r["fahrenheit"] for r in results if "fahrenheit" in r}
    parts = []
    for city, c in city_c.items():
        f = c if "ignore_tool_results" in state.faults else c_to_f.get(c, c)
        parts.append(f"{city}={f}F")
    return ", ".join(parts) or "No data."


GSM_FILE = Path(__file__).resolve().parent.parent / "ies" / "data" / "gsm8k_slice.jsonl"
GSM: list[dict] = []
BENCH_FLIPS = {"benchmark_noise": 1, "benchmark_drift": 3, "benchmark_regression": 15}


def gsm_answer(prompt: str, state: State) -> str | None:
    """A 'model' that solves the vendored GSM8K slice; benchmark faults break the first k items."""
    if not GSM and GSM_FILE.exists():
        with GSM_FILE.open("r", encoding="utf-8") as fh:
            GSM.extend(json.loads(line) for line in fh if line.strip())
    for idx, item in enumerate(GSM):
        if item["question"] in prompt:
            flips = max([k for f, k in BENCH_FLIPS.items() if f in state.faults] or [0])
            ans = item["answer"]
            if idx < flips:
                ans = str(int(float(ans)) + 1)
            return f"Working through the quantities step by step.\nAnswer: {ans}"
    return None


def context_answer(msgs: list[dict], prompt: str, state: State) -> str | None:
    """Answers that depend on what actually reached the model: history and long context."""
    if (g := gsm_answer(prompt, state)) is not None:
        return g
    if "bat and a ball cost" in prompt:
        return "$0.05"
    if "What is my project codename" in prompt:
        history = "\n".join(str(m.get("content") or "") for m in msgs[:-1])
        m = re.search(r"codename is ([A-Z0-9-]+)", history)
        return m[1] if m else "I don't know your codename."
    if "secret launch code" in prompt:
        m = re.search(r"secret launch code is ([A-Z]+-\d+)", prompt)
        return m[1] if m else "I could not find a launch code."
    if prompt.startswith("Summarize this page"):
        page = re.search(r"Page (\d+)\.", prompt)
        vault = re.search(r"vault number is (V-\d+)", prompt)
        k = page[1] if page else "?"
        if vault and not ("lossy_gist" in state.faults and k == "6"):
            return f"Page {k} describes the archive and gives vault number {vault[1]}."
        return f"Page {k} describes the archive."
    if m := re.search(r"archive vault number on page (\d+)", prompt):
        hit = re.search(rf"Page {m[1]} summary:[^\n]*?(V-\d+)", prompt)
        return hit[1] if hit else "Not stated in the summaries."
    return None


def generate(body: dict, state: State, rid: str) -> dict:
    msgs = body["messages"]
    prompt = str(msgs[-1].get("content", ""))
    seed = body.get("seed")
    temp = body.get("temperature", 1.0)
    out: dict = {"content": "", "tool_calls": None, "reasoning": ""}

    # crosstalk: answer some OTHER in-flight request's prompt
    if "crosstalk" in state.faults:
        with state.lock:
            others = [p for k, p in state.inflight.items() if k != rid]
        if others:
            prompt = others[0]

    tools = body.get("tools") or []
    if tools and body.get("tool_choice", "auto") != "none":
        calls = tool_policy(msgs, tools, body, state)
        if calls is not None:
            out["tool_calls"] = calls
            return out
        if any(m.get("role") == "tool" for m in msgs):
            out["content"] = agent_answer(msgs, state)
            return out

    rf = body.get("response_format") or {}
    if rf.get("type") in ("json_schema", "json_object"):
        m = re.search(r"([A-Z][a-z]+) is (\d+) years old", prompt)
        out["content"] = json.dumps({"name": m[1], "age": int(m[2])} if m else {"result": "ok"})
        return out

    ans = context_answer(msgs, prompt, state)
    if ans is None:
        ans = golden_answer(prompt, state)
    if ans is not None:
        out["content"] = ans
    else:
        long_form = any(w in prompt.lower() for w in ("essay", "story", "detailed", "explain"))
        n = 400 if long_form else 30
        reproducible = (temp == 0 or seed is not None) and "nondeterministic" not in state.faults
        with state.lock:
            busy = len(state.inflight) > 1 and "batch_variant" in state.faults
        rng = random.Random(_stable(prompt, seed, temp, "busy" if busy else "") if reproducible else None)
        out["content"] = " ".join(rng.choice(VOCAB) for _ in range(n)) + "."

    effort = body.get("reasoning_effort") or "high"           # qwen default per API reference
    model = body.get("model")
    if not (effort == "none" and model in NO_REASONING_OK):
        n = 60 if "effort_ignored" in state.faults else REASONING_LEN.get(effort, 150)
        words = "Let me think about this step by step and check the arithmetic before answering".split()
        out["reasoning"] = " ".join(words[i % len(words)] for i in range(n)) + "."
    return out


def stops_of(body: dict) -> list[str]:
    return [s for s in ([body["stop"]] if isinstance(body.get("stop"), str) else body.get("stop") or []) if s]


def apply_limits(text: str, body: dict, state: State, reasoning_tokens: int = 0) -> tuple[str, str]:
    """max_completion_tokens covers reasoning + content, as the real API documents."""
    finish = "stop"
    if not (body.get("stream") and "stop_chunk_boundary" in state.faults):   # fault: per-chunk matching only
        for s in stops_of(body):
            if s in text:
                text = text[: text.index(s)]
    limit = body.get("max_completion_tokens") or (
        None if "max_tokens_alias_ignored" in state.faults else body.get("max_tokens"))
    toks = TOKEN_RE.findall(text)
    room = None if not limit else max(0, limit - reasoning_tokens)
    if room is not None and len(toks) > room and "ignore_max_tokens" not in state.faults:
        text = " ".join(toks[:room])
        finish = "length"
    return text, finish


def logprob_entries(tokens: list[str], seed: object, state: State, top: int) -> list[dict]:
    out = []
    noise = random.Random() if "logprob_noise" in state.faults else None
    for i, t in enumerate(tokens):
        lp = -((_stable(t, i, seed) % 1000) / 1000.0) * 2.0
        if noise:
            lp += noise.uniform(-1e-2, 1e-2)
        if "numerics_shift" in state.faults:           # same build, numerics moved (kernel / compiler change)
            lp -= 0.03
        alts = [{"token": t, "logprob": lp, "bytes": list(t.encode())}]
        for j in range(1, top):
            alt = VOCAB[_stable(t, i, j) % len(VOCAB)]
            alts.append({"token": alt, "logprob": lp - j, "bytes": list(alt.encode())})
        if "logprobs_unsorted" in state.faults:
            alts.reverse()
        out.append({"token": t, "logprob": lp, "bytes": list(t.encode()), "top_logprobs": alts[:top]})
    return out


def decode_body(raw: bytes, headers: object, state: State) -> dict:
    """Content-Encoding: gzip (outer) and Content-Type: application/vnd.msgpack (inner)."""
    if headers.get("content-encoding", "").lower() == "gzip" and "gzip_ignored" not in state.faults:
        try:
            raw = gzip.decompress(raw)
        except (OSError, zlib.error) as exc:
            raise ValueError(f"bad gzip: {exc}") from exc
        if "decompress_truncation" in state.faults:
            body = _parse(raw, headers, state)
            for m in body.get("messages", []):        # a size-capped buffer: 200 OK, shorter prompt
                if isinstance(m.get("content"), str):
                    m["content"] = m["content"][:DECOMPRESS_LIMIT]
            return body
    return _parse(raw, headers, state)


def _parse(raw: bytes, headers: object, state: State) -> dict:
    if not raw:
        return {}
    if "msgpack" in headers.get("content-type", "") and "msgpack_as_json" not in state.faults:
        try:
            return msgpack.unpackb(raw)
        except (msgpack.exceptions.UnpackException, ValueError) as exc:
            raise ValueError(f"bad msgpack: {exc}") from exc
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"bad json: {exc}") from exc


def apply_context_faults(body: dict, state: State) -> dict:
    """Silent context loss, as a misconfigured gateway or server would cause it (200 OK)."""
    msgs = body.get("messages") or []
    if "history_truncated" in state.faults and len(msgs) > 5:
        system = [m for m in msgs if m.get("role") == "system"]
        body["messages"] = system + [m for m in msgs if m.get("role") != "system"][-4:]
    if "context_truncated" in state.faults:
        for m in body.get("messages") or []:
            text = m.get("content")
            if isinstance(text, str) and ntokens(text) > CONTEXT_LIMIT:
                m["content"] = " ".join(text.split()[: int(CONTEXT_LIMIT * 0.9)])
    return body


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-cerebras/1.0"
    state: State  # injected

    def log_message(self, *args: object) -> None:  # quiet
        pass

    # ---------- helpers
    def _json(self, code: int, obj: dict, headers: dict | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, message: str, etype: str, param: str | None = None,
               headers: dict | None = None) -> None:
        self._json(code, {"error": {"message": message, "type": etype, "param": param,
                                    "code": etype}}, headers)

    def _authed(self) -> bool:
        if self.headers.get("authorization", "") != f"Bearer {API_KEY}":
            self._error(401, "Wrong API Key", "invalid_request_error")
            return False
        return True

    def _rl_headers(self) -> dict:
        remaining = self.state.remaining_requests()
        return {"x-ratelimit-limit-requests-minute": str(RPM_LIMIT),
                "x-ratelimit-remaining-requests-minute": str(remaining),
                "x-ratelimit-limit-tokens-minute": "150000",
                "x-ratelimit-remaining-tokens-minute": str(max(0, 150000 - (RPM_LIMIT - remaining) * 200))}

    # ---------- routes
    def do_GET(self) -> None:
        if self.path.startswith("/v1/tcp_warming"):
            return self._json(200, {})
        if not self._authed():
            return
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": "Cerebras"} for m in MODELS
                if not ("model_removed" in self.state.faults and m == "gpt-oss-120b")]})
        self._error(404, f"no route {self.path}", "not_found_error")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._error(404, f"no route {self.path}", "not_found_error")
        if not self._authed():
            return
        st = self.state
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        time.sleep(len(raw) / UPLOAD_BPS)          # simulated client uplink: bytes cost time
        try:
            body = decode_body(raw, self.headers, st)
        except ValueError as exc:
            return self._error(400, f"could not parse request body: {exc}", "invalid_request_error")
        body = apply_context_faults(body, st)
        n = st.next_request()
        if "error_burst" in st.faults and n % 40 in (20, 21, 22):
            return self._error(500, "internal error: worker restarted", "server_error")

        pkey = hashlib.sha256(json.dumps(body.get("messages"), sort_keys=True).encode()).hexdigest()
        with st.lock:
            retried = pkey in st.limited          # light: a retried request is let through
            st.limited.discard(pkey)
            light = "rate_limit_light" in st.faults and n % 3 == 0 and not retried
            if light:
                st.limited.add(pkey)
        if light or ("rate_limit_storm" in st.faults and n % 5 != 0):
            return self._error(429, "Tokens per minute limit exceeded - too many uncached tokens "
                               "processed.", "too_many_tokens_error",
                               headers={"retry-after": "0.05", **self._rl_headers()})

        model = body.get("model")
        if model not in MODELS:
            return self._error(404, f"Model {model} does not exist or you do not have access to it.",
                               "not_found_error", "model")
        t = body.get("temperature")
        if t is not None and not 0 <= t <= 2:
            return self._error(400, "temperature must be between 0 and 2", "invalid_request_error",
                               "temperature")
        if body.get("n", 1) != 1:
            return self._error(400, "n > 1 is not supported", "invalid_request_error", "n")
        for name, lo, hi in (("top_p", 0, 1), ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2),
                             ("top_logprobs", 0, 20)):
            v = body.get(name)
            if v is not None and not lo <= v <= hi:
                return self._error(400, f"{name} must be between {lo} and {hi}", "invalid_request_error", name)
        stops = body.get("stop")
        if isinstance(stops, list) and len(stops) > 4:
            return self._error(400, "at most 4 stop sequences", "invalid_request_error", "stop")
        mct = body.get("max_completion_tokens") or body.get("max_tokens")
        if mct is not None and mct < 1:
            return self._error(400, "max_completion_tokens must be >= 1", "invalid_request_error",
                               "max_completion_tokens")
        if not body.get("messages"):
            return self._error(400, "messages must be non-empty", "invalid_request_error", "messages")

        rid = uuid.uuid4().hex
        with st.lock:
            st.inflight[rid] = str(body["messages"][-1].get("content", ""))
        try:
            self._complete(body, rid)
        finally:
            with st.lock:
                st.inflight.pop(rid, None)

    def _complete(self, body: dict, rid: str) -> None:
        st = self.state
        t0 = time.perf_counter()
        created: int | str = str(int(time.time())) if "schema_drift" in st.faults else int(time.time())
        fp = "fp_mock_v2" if "fingerprint_v2" in st.faults else "fp_mock_v1"
        msgs = body["messages"]
        all_text = "\n".join(str(m.get("content") or "") for m in msgs)
        prompt_tokens = ntokens(all_text) + 3 * len(msgs)
        # Prefix cache: the longest run of leading messages seen before (same prompt_cache_key
        # namespace) is served from cache, rounded down to whole blocks. An identical prompt
        # re-uses all but its last block.
        ns = body.get("prompt_cache_key") or ""
        h = hashlib.sha256(ns.encode())
        prefix_keys = []
        for m in msgs:
            h.update(json.dumps([m.get("role"), str(m.get("content") or "")]).encode())
            prefix_keys.append(h.hexdigest())
        cached = 0
        with st.lock:
            if "cache_miss" not in st.faults:
                for k in range(len(msgs), 0, -1):
                    if prefix_keys[k - 1] in st.cache:
                        ptoks = prompt_tokens - 1 if k == len(msgs) else \
                            ntokens("\n".join(str(m.get("content") or "") for m in msgs[:k])) + 3 * k
                        cached = (ptoks // CACHE_BLOCK) * CACHE_BLOCK
                        break
            st.cache.update(prefix_keys)

        gen = generate(body, st, rid)
        reasoning_tokens = ntokens(gen["reasoning"])
        limit = body.get("max_completion_tokens") or body.get("max_tokens")
        if limit and reasoning_tokens > limit and "ignore_max_tokens" not in st.faults:
            gen["reasoning"] = " ".join(gen["reasoning"].split()[:limit])     # reasoning consumes the budget
            reasoning_tokens = ntokens(gen["reasoning"])
        text, finish = apply_limits(gen["content"], body, st, reasoning_tokens)
        if "budget_stop" in st.faults and finish == "length" and not text:
            finish = "stop"                                                  # empty answer reported as complete
        if "reasoning_leak" in st.faults and gen["reasoning"]:
            text = f"<think>{gen['reasoning']}</think>{text}"
        if "harmony_markers" in st.faults:
            text = f"<|channel|>final<|message|>{text}"
        tool_calls = gen["tool_calls"]
        if tool_calls:
            finish = "tool_calls"
        out_tokens = TOKEN_RE.findall(text)
        billed_reasoning = 0 if "reasoning_unbilled" in st.faults else reasoning_tokens
        completion_tokens = len(out_tokens) + billed_reasoning + (8 if tool_calls else 0)

        with st.lock:
            inflight = len(st.inflight)
        queue = 0.0005 + (0.08 * inflight if "contention" in st.faults else 0.0)
        prompt_time = (prompt_tokens - cached) / PREFILL_TPS
        pre = queue + prompt_time + (0.4 if "slow_ttft" in st.faults else 0.0)
        if "latency_creep" in st.faults:                  # slow leak: every request a little slower
            pre += 0.03 * max(0, st.requests - 20)        # shows up only after a while, like a leak
        tps = SLOW_DECODE_TPS if "slow_decode" in st.faults else DECODE_TPS
        time.sleep(pre)

        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens + (1 if "usage_drift" in st.faults else 0),
                 "prompt_tokens_details": {"cached_tokens": cached},
                 "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
        lp = None
        if body.get("logprobs"):
            lp = {"content": logprob_entries(out_tokens, body.get("seed"), st,
                                             int(body.get("top_logprobs") or 1)), "refusal": None}
        base = {"id": f"chatcmpl-{rid}", "created": created, "model": body["model"],
                "system_fingerprint": fp}

        def time_info(decode: float) -> dict:
            total = time.perf_counter() - t0
            return {"queue_time": queue, "prompt_time": prompt_time, "completion_time": decode,
                    "total_time": total, "created": time.time()}

        if not body.get("stream"):
            d0 = time.perf_counter()
            time.sleep(completion_tokens / tps)
            msg = {"role": "assistant", "content": None if tool_calls else text, "tool_calls": tool_calls}
            if gen["reasoning"]:
                msg["reasoning"] = gen["reasoning"]
            return self._json(200, {**base, "object": "chat.completion", "choices": [
                {"index": 0, "finish_reason": finish, "message": msg, "logprobs": lp}],
                "usage": usage, "time_info": time_info(time.perf_counter() - d0)}, self._rl_headers())

        # ---- SSE stream
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        for k, v in self._rl_headers().items():
            self.send_header(k, v)
        self.end_headers()

        def send(obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        chunk = {**base, "object": "chat.completion.chunk"}
        d0 = time.perf_counter()
        send({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        if gen["reasoning"]:
            time.sleep(reasoning_tokens / tps)
            send({**chunk, "choices": [{"index": 0, "delta": {"reasoning": gen["reasoning"]}}]})
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                send({**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": i, "id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"], "arguments": ""}}]}}]})
                args = tc["function"]["arguments"]
                for half in (args[: len(args) // 2], args[len(args) // 2:]):   # arguments arrive in fragments
                    send({**chunk, "choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": i, "type": "function", "function": {"arguments": half}}]}}]})
        else:
            pieces = re.findall(r"\S+\s*", text) or [text]
            lp_items = (lp or {}).get("content") or []
            for i in range(0, len(pieces), CHUNK_TOKENS):
                if "midstream_error" in st.faults and i >= 3 * CHUNK_TOKENS and len(pieces) > 100:
                    send({"error": {"message": "internal error: decode worker lost",
                                    "type": "server_error", "code": "server_error"}})
                    return
                part = "".join(pieces[i:i + CHUNK_TOKENS])
                if "utf8_split" in st.faults:                 # multi-byte chars split across chunks
                    part = "".join("\ufffd" if ord(ch) > 0xFFFF else ch for ch in part)
                cut = [part.index(s) for s in stops_of(body) if s in part] \
                    if "stop_chunk_boundary" in st.faults else []
                if cut:
                    part = part[: min(cut)]
                time.sleep(ntokens(part) / tps)
                c = {"index": 0, "delta": {"content": part}}
                if lp_items:
                    c["logprobs"] = {"content": lp_items[i:i + CHUNK_TOKENS], "refusal": None}
                send({**chunk, "choices": [c]})
                if cut:
                    break
        if "schema_drift" in st.faults:
            finish = "eos"
        final: dict = {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "time_info": time_info(time.perf_counter() - d0)}
        if "stream_no_usage" not in st.faults:
            final["usage"] = usage
        send(final)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(port: int = 0, faults: set[str] | None = None) -> tuple[ThreadingHTTPServer, State]:
    """Start in a daemon thread. Returns (server, state); base URL is http://127.0.0.1:<port>."""
    state = State(faults)
    handler = type("BoundHandler", (Handler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def main() -> None:
    port = int(os.environ.get("MOCK_PORT", "8089"))
    faults = {f for f in os.environ.get("MOCK_FAULTS", "").split(",") if f}
    srv, _ = serve(port, faults)
    print(f"mock cerebras on http://127.0.0.1:{srv.server_address[1]} faults={sorted(faults)}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
