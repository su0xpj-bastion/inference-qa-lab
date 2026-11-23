"""Minimal W3C trace-context propagation and an in-memory span store.

Real deployments export spans to an OpenTelemetry collector. Here each
process keeps recent spans in memory and serves them at /debug/spans,
which keeps the testbed dependency-free. The span shape (trace_id, service,
name, start, end) matches what an OTLP exporter would send.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
from collections import OrderedDict

TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")
MAX_TRACES = 5000


def new_traceparent() -> tuple[str, str]:
    """Return (traceparent header value, trace_id)."""
    trace_id = secrets.token_hex(16)
    return f"00-{trace_id}-{secrets.token_hex(8)}-01", trace_id


def trace_id_of(header: str | None) -> str | None:
    m = TRACEPARENT.match((header or "").strip().lower())
    return m.group(1) if m else None


class SpanStore:
    def __init__(self, service: str) -> None:
        self.service = service
        self._lock = threading.Lock()
        self._traces: OrderedDict[str, list[dict]] = OrderedDict()

    def add(self, trace_id: str | None, name: str, start: float, end: float, **attrs: object) -> None:
        if not trace_id:
            return
        span = {"trace_id": trace_id, "service": self.service, "name": name,
                "start": start, "end": end, "dur_ms": round((end - start) * 1000, 3), **attrs}
        with self._lock:
            self._traces.setdefault(trace_id, []).append(span)
            self._traces.move_to_end(trace_id)
            while len(self._traces) > MAX_TRACES:
                self._traces.popitem(last=False)

    def get(self, trace_id: str) -> list[dict]:
        with self._lock:
            return list(self._traces.get(trace_id, []))


def now() -> float:
    """Wall clock for spans: comparable across processes on one host (and NTP-synced hosts)."""
    return time.time()
