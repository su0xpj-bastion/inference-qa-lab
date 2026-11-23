"""mockserve — an OpenAI-compatible inference replica for platform testing.

It stands in for one model-serving pod: it does no inference, but it
behaves like a real replica in the ways the *platform* sees: SSE
streaming, the final usage chunk, readiness and liveness probes,
graceful drain on SIGTERM, Prometheus metrics, and fault injection.

Endpoints
---------
  POST /v1/chat/completions   stream and non-stream
  GET  /v1/models
  GET  /healthz               liveness: process is up
  GET  /readyz                readiness: 503 while draining or forced down
  GET  /metrics               Prometheus text format
  POST /admin/ready           {"ready": false} flips readiness (faults only)
  GET  /debug/spans?trace_id= spans this replica recorded for a W3C trace id

Fault injection (only when MOCKSERVE_ALLOW_FAULTS=1)
----------------------------------------------------
  Request header `x-fault`:
    429               rate-limit response with retry-after
    503               upstream-unavailable response
    drop-midstream    close the socket after a few tokens, no [DONE]
    hang-ttft:<ms>    sleep before the first token
    no-usage          omit the usage chunk even when requested

Environment
-----------
  REPLICA_ID            value of the x-served-by header (default: hostname)
  MOCKSERVE_TTFT_MS     delay before the first token (default 20)
  MOCKSERVE_TBT_MS      delay between tokens (default 2)
  MOCKSERVE_DRAIN_GRACE_S  seconds to keep accepting after SIGTERM (default 2)
  MOCKSERVE_DRAIN_S     max seconds to wait for in-flight requests after that
  MOCKSERVE_SCHEDULER   immediate | fixed-wait | adaptive (default immediate; see sched.py)
  MOCKSERVE_MAX_BATCH   requests per dispatched batch (default 8)
  MOCKSERVE_BATCH_WAIT_MS  batch window (default 350)
  MOCKSERVE_IDLE_GAP_MS    adaptive: dispatch once arrivals are quiet this long (default 5)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from mockserve.sched import BatchScheduler
from mockserve.tracing import SpanStore, now, trace_id_of

MODELS = ("llama3.1-8b", "llama-3.3-70b", "qwen-3-32b")
DEFAULT_MAX_TOKENS = 32


class ReplicaState:
    """Mutable state shared across handler threads."""

    def __init__(self, replica_id: str, ttft_ms: int, tbt_ms: int, allow_faults: bool,
                 scheduler: BatchScheduler | None = None) -> None:
        self.replica_id = replica_id
        self.sched = scheduler or BatchScheduler("immediate")
        self.spans = SpanStore(f"mockserve/{replica_id}")
        self.sched_wait_sum = 0.0
        self.sched_wait_count = 0
        self.ttft_s = ttft_ms / 1000
        self.tbt_s = tbt_ms / 1000
        self.allow_faults = allow_faults
        self.ready = True
        self.draining = False
        self.served_while_draining = 0
        self.lock = threading.Lock()
        self.inflight = 0
        self.requests_total: dict[tuple[str, int], int] = {}
        self.tokens_total = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0

    def count(self, path: str, code: int) -> None:
        with self.lock:
            key = (path, code)
            self.requests_total[key] = self.requests_total.get(key, 0) + 1

    def render_metrics(self) -> str:
        with self.lock:
            lines = [
                "# TYPE mockserve_requests_total counter",
                *(
                    f'mockserve_requests_total{{path="{p}",code="{c}",replica="{self.replica_id}"}} {n}'
                    for (p, c), n in sorted(self.requests_total.items())
                ),
                "# TYPE mockserve_inflight gauge",
                f'mockserve_inflight{{replica="{self.replica_id}"}} {self.inflight}',
                "# TYPE mockserve_served_while_draining_total counter",
                f'mockserve_served_while_draining_total{{replica="{self.replica_id}"}} {self.served_while_draining}',
                "# TYPE mockserve_tokens_total counter",
                f'mockserve_tokens_total{{replica="{self.replica_id}"}} {self.tokens_total}',
                "# TYPE mockserve_ttft_seconds summary",
                f'mockserve_ttft_seconds_sum{{replica="{self.replica_id}"}} {self.ttft_sum:.6f}',
                f'mockserve_ttft_seconds_count{{replica="{self.replica_id}"}} {self.ttft_count}',
                "# TYPE mockserve_sched_wait_seconds summary",
                f'mockserve_sched_wait_seconds_sum{{replica="{self.replica_id}",policy="{self.sched.policy}"}} {self.sched_wait_sum:.6f}',
                f'mockserve_sched_wait_seconds_count{{replica="{self.replica_id}",policy="{self.sched.policy}"}} {self.sched_wait_count}',
                "# TYPE mockserve_batch_size_mean gauge",
                f'mockserve_batch_size_mean{{replica="{self.replica_id}"}} {self.sched.mean_batch():.3f}',
            ]
        return "\n".join(lines) + "\n"


def log(msg: str) -> None:
    """Timestamped stdout line; pod logs are the drain timeline in CI diagnostics."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"
    print(f"{ts} {msg}", flush=True)


def make_tokens(messages: list[dict], max_tokens: int) -> tuple[list[str], str, int]:
    """Deterministic 'completion': echo prompt words back, capped by max_tokens.

    Returns (tokens, finish_reason, prompt_tokens). Determinism lets tests
    assert exact content and usage numbers.
    """
    words: list[str] = []
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            words.extend(content.split())
    prompt_tokens = len(words) + 3 * len(messages) + 3  # framing overhead, gpt-style
    natural = [f"{w} " for w in words] or ["ok "]
    if len(natural) > max_tokens:
        return natural[:max_tokens], "length", prompt_tokens
    return natural, "stop", prompt_tokens


class Handler(BaseHTTPRequestHandler):
    server_version = "mockserve/1.0"
    protocol_version = "HTTP/1.1"
    state: ReplicaState  # injected by build_server

    # ---- plumbing -------------------------------------------------------
    def log_message(self, fmt: str, *args: object) -> None:  # quiet by default
        if os.environ.get("MOCKSERVE_LOG"):
            super().log_message(fmt, *args)

    def _send_json(self, code: int, body: dict, extra: dict[str, str] | None = None) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.send_header("x-served-by", self.state.replica_id)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)
        self.state.count(self.path, code)

    def _error(self, code: int, kind: str, message: str, extra: dict[str, str] | None = None) -> None:
        self._send_json(code, {"error": {"type": kind, "message": message, "code": code}}, extra)

    # ---- GET ------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok", "replica": self.state.replica_id})
        elif self.path == "/readyz":
            if self.state.ready and not self.state.draining:
                self._send_json(200, {"status": "ready"})
            else:
                self._send_json(503, {"status": "draining" if self.state.draining else "not-ready"})
        elif self.path == "/metrics":
            raw = self.state.render_metrics().encode()
            self.send_response(200)
            self.send_header("content-type", "text/plain; version=0.0.4")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path.startswith("/debug/spans"):
            tid = (parse_qs(urlparse(self.path).query).get("trace_id") or [""])[0]
            self._send_json(200, {"spans": self.state.spans.get(tid)})
        elif self.path == "/v1/models":
            data = [{"id": m, "object": "model", "owned_by": "mockserve"} for m in MODELS]
            self._send_json(200, {"object": "list", "data": data})
        else:
            self._error(404, "not_found", f"no route {self.path}")

    # ---- POST -----------------------------------------------------------
    def do_POST(self) -> None:
        self.t_recv = now()
        self.trace_id = trace_id_of(self.headers.get("traceparent"))
        length = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._error(400, "invalid_request_error", "body is not valid JSON")
            return

        if self.path == "/admin/ready":
            if not self.state.allow_faults:
                self._error(403, "forbidden", "admin disabled")
                return
            self.state.ready = bool(body.get("ready", True))
            self._send_json(200, {"ready": self.state.ready})
            return

        if self.path != "/v1/chat/completions":
            self._error(404, "not_found", f"no route {self.path}")
            return

        # While draining, keep serving. Endpoint removal propagates to kube-proxy
        # and ingress-nginx asynchronously, so for a short while traffic can still
        # arrive after SIGTERM. Refusing it with 503 is a client-visible error that
        # NGINX will not retry for a POST (CI run 43620047560 caught exactly that).
        # Count it instead: a non-zero value measures the propagation lag.
        with self.state.lock:
            self.state.inflight += 1
            if self.state.draining:
                self.state.served_while_draining += 1
                log(f"served request while draining (#{self.state.served_while_draining})")
        try:
            self._chat(body)
        finally:
            with self.state.lock:
                self.state.inflight -= 1

    def _chat(self, body: dict) -> None:
        fault = self.headers.get("x-fault", "") if self.state.allow_faults else ""
        model = body.get("model")
        messages = body.get("messages")
        if model not in MODELS:
            self._error(404, "model_not_found", f"model {model!r} does not exist")
            return
        if not isinstance(messages, list) or not messages:
            self._error(400, "invalid_request_error", "messages must be a non-empty list")
            return
        if fault == "429":
            self._error(429, "rate_limit_exceeded", "slow down", {"retry-after": "1"})
            return
        if fault == "503":
            self._error(503, "service_unavailable", "injected 503")
            return

        max_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
        tokens, finish_reason, prompt_tokens = make_tokens(messages, max_tokens)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(tokens),
            "total_tokens": prompt_tokens + len(tokens),
        }
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        ttft = self.state.ttft_s
        if fault.startswith("hang-ttft:"):
            ttft = int(fault.split(":", 1)[1]) / 1000
        start = time.monotonic()

        if not body.get("stream"):
            self._admit()
            time.sleep(ttft + self.state.tbt_s * len(tokens))
            self._record_tokens(len(tokens), time.monotonic() - start)
            self._send_json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "finish_reason": finish_reason,
                             "message": {"role": "assistant", "content": "".join(tokens)}}],
                "usage": usage,
            })
            self.state.spans.add(self.trace_id, "server.total", self.t_recv, now())
            return

        include_usage = bool((body.get("stream_options") or {}).get("include_usage")) and fault != "no-usage"
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("x-served-by", self.state.replica_id)
        self.send_header("x-accel-buffering", "no")  # tell NGINX not to buffer SSE
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        self.state.count(self.path, 200)

        def emit(obj: dict | str) -> None:
            data = obj if isinstance(obj, str) else json.dumps(obj)
            payload = f"data: {data}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
        t_disp = self._admit()
        time.sleep(ttft)                         # prefill
        emit({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        t_first = now()
        self.state.spans.add(self.trace_id, "server.prefill", t_disp, t_first)
        self.state.spans.add(self.trace_id, "server.ttfb", self.t_recv, t_first)
        for i, tok in enumerate(tokens):
            if i == 0:
                self._record_tokens(0, time.monotonic() - start)
            if fault == "drop-midstream" and i == 3:
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            emit({**base, "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]})
            time.sleep(self.state.tbt_s)
        emit({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
        if include_usage:
            emit({**base, "choices": [], "usage": usage})
        emit("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        with self.state.lock:
            self.state.tokens_total += len(tokens)
        self.state.spans.add(self.trace_id, "server.total", self.t_recv, now())

    def _admit(self) -> float:
        """Wait for the batch scheduler; record the wait as its own span. Returns dispatch time."""
        t_enq = now()
        waited = self.state.sched.admit()
        t_disp = now()
        self.state.spans.add(self.trace_id, "server.sched_wait", t_enq, t_disp, policy=self.state.sched.policy)
        with self.state.lock:
            self.state.sched_wait_sum += waited
            self.state.sched_wait_count += 1
        return t_disp

    def _record_tokens(self, n: int, ttft: float) -> None:
        with self.state.lock:
            self.state.tokens_total += n
            self.state.ttft_sum += ttft
            self.state.ttft_count += 1


class QuietServer(ThreadingHTTPServer):
    """Client resets mid-stream are expected under test; don't print tracebacks for them."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def build_server(host: str, port: int, state: ReplicaState) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = QuietServer((host, port), handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    state = ReplicaState(
        replica_id=os.environ.get("REPLICA_ID") or socket.gethostname(),
        ttft_ms=int(os.environ.get("MOCKSERVE_TTFT_MS", "20")),
        tbt_ms=int(os.environ.get("MOCKSERVE_TBT_MS", "2")),
        allow_faults=os.environ.get("MOCKSERVE_ALLOW_FAULTS") == "1",
        scheduler=BatchScheduler(
            policy=os.environ.get("MOCKSERVE_SCHEDULER", "immediate"),
            max_batch=int(os.environ.get("MOCKSERVE_MAX_BATCH", "8")),
            batch_wait_s=int(os.environ.get("MOCKSERVE_BATCH_WAIT_MS", "350")) / 1000,
            idle_gap_s=int(os.environ.get("MOCKSERVE_IDLE_GAP_MS", "5")) / 1000,
        ),
    )
    drain_s = float(os.environ.get("MOCKSERVE_DRAIN_S", "10"))
    grace_s = float(os.environ.get("MOCKSERVE_DRAIN_GRACE_S", "2"))
    server = build_server(args.host, args.port, state)

    def on_sigterm(_signum: int, _frame: object) -> None:
        # Kubernetes sends SIGTERM, then SIGKILL after terminationGracePeriodSeconds.
        # 1. fail readiness, keep accepting for grace_s (endpoint removal is async)
        # 2. close the listener: late traffic now gets connect-refused, which
        #    NGINX safely retries on another pod (proxy_next_upstream error)
        # 3. wait for in-flight streams, up to drain_s, then exit 0
        state.draining = True
        log(f"SIGTERM: readiness failing, accepting for {grace_s}s more, inflight={state.inflight}")

        def stop_accepting() -> None:
            time.sleep(grace_s)
            server.shutdown()

        threading.Thread(target=stop_accepting, daemon=True).start()

    signal.signal(signal.SIGTERM, on_sigterm)
    log(f"mockserve {state.replica_id} listening on {args.host}:{server.server_address[1]}")
    server.serve_forever()
    server.server_close()
    log(f"listener closed, waiting for inflight={state.inflight}")
    deadline = time.monotonic() + drain_s
    while state.inflight and time.monotonic() < deadline:
        time.sleep(0.05)
    log(f"exit: inflight={state.inflight} served_while_draining={state.served_while_draining}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
