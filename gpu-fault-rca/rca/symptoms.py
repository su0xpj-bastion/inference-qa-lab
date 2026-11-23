"""User-visible symptoms from the router access log: p99 SLO breaches and 5xx bursts per replica."""
from __future__ import annotations

import math
from dataclasses import dataclass

from rca.parsers import Request

WINDOW_S = 10
MIN_LATENCY_WINDOWS = 2      # one slow 10 s window is a blip, not an incident
MIN_ERRORS = 3


@dataclass
class Symptom:
    kind: str                # latency_slo | error_burst
    replica: str             # replica name, or "router" when no upstream was available (503)
    start: float
    end: float
    worst_p99: float
    errors: int
    requests: int
    failover_from: object = None

    def to_dict(self, t0: float) -> dict:
        return {"kind": self.kind, "replica": self.replica, "start_s": round(self.start - t0, 1),
                "end_s": round(self.end - t0, 1), "worst_p99_s": round(self.worst_p99, 2),
                "errors": self.errors, "requests": self.requests,
                "failover_from": (f"{self.failover_from.kind}@{self.failover_from.replica}"
                                  if self.failover_from is not None else None)}


def p99(values: list[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[max(0, math.ceil(0.99 * len(v)) - 1)]


def detect(requests: list[Request], start: float, slo_p99: float, slo_err: float) -> list[Symptom]:
    buckets: dict[tuple[str, int], list[Request]] = {}
    for r in requests:
        if r.status == 499:                 # client went away; not our failure
            continue
        buckets.setdefault((r.upstream if r.upstream != "-" else "router", int((r.end - start) // WINDOW_S)),
                           []).append(r)
    flagged: dict[tuple[str, str], list[tuple[int, float, int, int]]] = {}
    for (ups, b), reqs in sorted(buckets.items()):
        errors = sum(1 for r in reqs if r.status >= 500)
        lat = p99([r.rt for r in reqs])
        if errors >= MIN_ERRORS and errors / len(reqs) > slo_err:
            flagged.setdefault(("error_burst", ups), []).append((b, lat, errors, len(reqs)))
        if ups != "router" and lat > slo_p99:
            flagged.setdefault(("latency_slo", ups), []).append((b, lat, errors, len(reqs)))
    out: list[Symptom] = []
    for (kind, ups), rows in flagged.items():
        run: list[tuple[int, float, int, int]] = []
        for row in rows + [(10 ** 9, 0.0, 0, 0)]:
            if run and row[0] - run[-1][0] > 2:      # allow one quiet window inside a burst
                if kind == "error_burst" or len(run) >= MIN_LATENCY_WINDOWS:
                    out.append(Symptom(kind, ups, start + run[0][0] * WINDOW_S, start + (run[-1][0] + 1) * WINDOW_S,
                                       max(x[1] for x in run), sum(x[2] for x in run), sum(x[3] for x in run)))
                run = []
            run.append(row)
    return sorted(out, key=lambda s: (s.start, s.kind, s.replica))
