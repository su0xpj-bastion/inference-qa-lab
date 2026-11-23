"""Pure latency math. No I/O, no network — every function is unit-testable."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StreamTiming:
    """Timing for one streamed completion, in seconds."""

    ttft: float           # request start → first content token
    tbt: float            # mean gap between content tokens
    total: float          # request start → stream end
    tokens: int           # content chunks observed (chunk == token)

    @property
    def output_tps(self) -> float:
        """Decode throughput: tokens after the first, over decode time."""
        decode = self.total - self.ttft
        return (self.tokens - 1) / decode if self.tokens > 1 and decode > 0 else 0.0


def stream_timing(start: float, token_times: list[float], end: float) -> StreamTiming:
    total = max(0.0, end - start)
    if not token_times:
        return StreamTiming(ttft=0.0, tbt=0.0, total=total, tokens=0)
    ttft = max(0.0, token_times[0] - start)
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    tbt = sum(gaps) / len(gaps) if gaps else 0.0
    return StreamTiming(ttft=ttft, tbt=tbt, total=total, tokens=len(token_times))


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (the definition most SLO docs use).

    Nearest-rank never interpolates, so p99 of a sample is always a value
    that was actually observed — easier to defend in an incident review.
    """
    if not values:
        return 0.0
    if not 0 < pct <= 100:
        raise ValueError(f"pct must be in (0, 100], got {pct}")
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[rank - 1]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else 0.0,
    }
