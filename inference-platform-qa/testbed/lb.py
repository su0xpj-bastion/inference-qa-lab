"""lb — a small L7 round-robin proxy that mirrors NGINX upstream semantics.

Why not just run NGINX? CI laptops and unit runners often have no
Docker. This proxy reproduces the specific NGINX behaviours the test
suite asserts on, so the same tests run against it locally and against
real ingress-nginx in the kind job. Where behaviour differs, the real
ingress wins and the test is marked `cluster`.

Semantics mirrored from ngx_http_upstream_module
-------------------------------------------------
  * round-robin across upstreams
  * passive health: `max_fails=1 fail_timeout=<s>` — a connect error
    ejects the peer for fail_timeout seconds
  * proxy_next_upstream error timeout — retry the NEXT peer only when
    the request never reached an upstream (connect refused / timeout).
    A POST that reached a peer is NOT retried (non_idempotent off).
  * if every peer is ejected, NGINX resets them all and tries anyway;
    we do the same before returning 502
  * proxy_buffering off — SSE bytes are relayed as they arrive
  * adds `x-upstream` with the peer address
  * tracing: forwards W3C `traceparent` unchanged (it is not hop-by-hop),
    records lb.connect / lb.ttfb / lb.total spans, and serves the merged
    lb + upstream spans for a trace at GET /debug/spans?trace_id=
    (ingress-nginx equivalent: $upstream_connect_time / $upstream_header_time
    in the access log, or its OpenTelemetry module)
"""
from __future__ import annotations

import argparse
import http.client
import itertools
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from mockserve.tracing import SpanStore, now, trace_id_of

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-authenticate", "content-length"}


class Upstreams:
    def __init__(self, peers: list[str], fail_timeout: float) -> None:
        self.peers = peers
        self.fail_timeout = fail_timeout
        self.down_until: dict[str, float] = {}
        self.lock = threading.Lock()
        self._rr = itertools.cycle(range(len(peers)))

    def order(self) -> list[str]:
        """Peers to try for one request: live ones in RR order."""
        now = time.monotonic()
        with self.lock:
            start = next(self._rr)
            rotated = self.peers[start:] + self.peers[:start]
            live = [p for p in rotated if self.down_until.get(p, 0) <= now]
            if not live:  # NGINX: all peers failed → reset and try them all
                self.down_until.clear()
                live = rotated
        return live

    def mark_failed(self, peer: str) -> None:
        with self.lock:
            self.down_until[peer] = time.monotonic() + self.fail_timeout


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstreams: Upstreams
    spans: SpanStore
    connect_timeout: float = 1.0
    read_timeout: float = 60.0

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/debug/spans"):
            self._spans()
            return
        self._proxy(b"")

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        self._proxy(self.rfile.read(length))

    def _spans(self) -> None:
        """Own spans for the trace, plus the spans of the upstream that served it."""
        tid = (parse_qs(urlparse(self.path).query).get("trace_id") or [""])[0]
        spans = self.spans.get(tid)
        for peer in {s["peer"] for s in spans if s.get("peer")}:
            host, port = peer.rsplit(":", 1)
            conn = http.client.HTTPConnection(host, int(port), timeout=2)
            try:
                conn.request("GET", f"/debug/spans?trace_id={tid}")
                spans += json.loads(conn.getresponse().read())["spans"]
            except (OSError, ValueError, KeyError):
                pass   # upstream gone (killed / drained): return what we have
            finally:
                conn.close()
        raw = json.dumps({"spans": spans}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _proxy(self, body: bytes) -> None:
        self.t_recv = now()
        self.trace_id = trace_id_of(self.headers.get("traceparent"))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        headers["content-length"] = str(len(body))
        for peer in self.upstreams.order():
            host, port = peer.rsplit(":", 1)
            conn = http.client.HTTPConnection(host, int(port), timeout=self.connect_timeout)
            try:
                conn.connect()
            except OSError:
                self.upstreams.mark_failed(peer)
                continue  # request never reached the peer → safe to try the next one
            self.spans.add(self.trace_id, "lb.connect", self.t_recv, now(), peer=peer)
            conn.sock.settimeout(self.read_timeout)
            try:
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
            except OSError:
                # Request was sent: non-idempotent POST must not be replayed.
                self.upstreams.mark_failed(peer)
                self._bad_gateway("upstream failed after request was sent")
                return
            self._relay(resp, peer)
            conn.close()
            return
        self._bad_gateway("no live upstream")

    def _relay(self, resp: http.client.HTTPResponse, peer: str) -> None:
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("x-upstream", peer)
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        first = True
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                if first:
                    self.spans.add(self.trace_id, "lb.ttfb", self.t_recv, now(), peer=peer)
                    first = False
        except (OSError, http.client.IncompleteRead):
            # Upstream died mid-stream. Headers are already sent, so the only
            # honest signal left is to cut the client connection without the
            # terminating chunk — exactly what NGINX does.
            self.close_connection = True
            return
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        self.spans.add(self.trace_id, "lb.total", self.t_recv, now(), peer=peer)

    def _bad_gateway(self, reason: str) -> None:
        raw = ('{"error":{"type":"bad_gateway","message":"%s","code":502}}' % reason).encode()
        self.send_response(502)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class QuietServer(ThreadingHTTPServer):
    """Client resets mid-stream are expected under test; don't print tracebacks for them."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


def build_proxy(host: str, port: int, peers: list[str], fail_timeout: float = 10.0) -> ThreadingHTTPServer:
    handler = type("BoundProxy", (ProxyHandler,), {"upstreams": Upstreams(peers, fail_timeout),
                                                   "spans": SpanStore("lb")})
    server = QuietServer((host, port), handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NGINX-semantics round-robin proxy")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fail-timeout", type=float, default=10.0)
    parser.add_argument("upstream", nargs="+", help="host:port")
    args = parser.parse_args(argv)
    server = build_proxy("127.0.0.1", args.port, args.upstream, args.fail_timeout)
    print(f"lb listening on :{server.server_address[1]} → {args.upstream}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
