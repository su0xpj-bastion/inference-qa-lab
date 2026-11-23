"""Shared run context, check results, and pure statistics."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import cerebras.cloud.sdk as cerebras
import jsonschema
import openai

from .client import Budget, Call, chat


@dataclass
class Check:
    workflow: str
    name: str
    status: str            # pass | fail | warn | info | skip
    detail: str
    gated: bool = True
    metrics: dict = field(default_factory=dict)


@dataclass
class Ctx:
    client: cerebras.Cerebras
    model: str
    target: str            # mock | live
    slo: dict
    budget: Budget
    bad_key_client: Callable[[], cerebras.Cerebras]
    base_params: dict = field(default_factory=dict)
    baseline_dir: str = "baselines"
    update_baseline: bool = False
    api_key: str | None = None
    base_url: str | None = None
    captured: list = field(default_factory=list)   # raw responses / chunks, for schema conformance

    def chat(self, messages: list[dict] | str, *, stream: bool = False, **params: object) -> Call:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return chat(self.client, self.model, messages, stream=stream, budget=self.budget,
                    capture=self.captured, **{**self.base_params, **params})


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile; q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1))]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def tbt(content_times: list[float]) -> float:
    """Mean gap between content chunks (seconds); 0 with fewer than 2 chunks."""
    if len(content_times) < 2:
        return 0.0
    return (content_times[-1] - content_times[0]) / (len(content_times) - 1)


def server_decode_tps(c: Call) -> float | None:
    if not c.completion_tokens or not c.completion_time:
        return None
    return c.completion_tokens / c.completion_time


def client_overhead(c: Call) -> float | None:
    """Wall time the server didn't account for: network + TLS + client parsing."""
    if c.server_total is None:
        return None
    return max(0.0, c.wall - c.server_total)


def verdict(checks: list[Check]) -> str:
    if any(c.status == "fail" and c.gated for c in checks):
        return "FAIL"
    if any(c.status in ("fail", "warn") for c in checks):
        return "PASS (warnings)"
    return "PASS"


def guarded(workflow: str, name: str, fn: Callable[[], Check], gated: bool = True) -> Check:
    """Run one check; any API error or exception becomes a failed check, not a crash."""
    try:
        return fn()
    except cerebras.RateLimitError as exc:
        return Check(workflow, name, "fail", f"429 after retries: {exc.message}", gated)
    except cerebras.APIStatusError as exc:
        return Check(workflow, name, "fail", f"HTTP {exc.status_code}: {exc.message}", gated)
    except cerebras.APIError as exc:
        return Check(workflow, name, "fail", f"API error: {exc}", gated)
    except openai.APIStatusError as exc:
        return Check(workflow, name, "fail", f"HTTP {exc.status_code} (openai sdk): {exc.message}", gated)
    except openai.APIError as exc:
        return Check(workflow, name, "fail", f"API error (openai sdk): {exc}", gated)
    except jsonschema.ValidationError as exc:
        return Check(workflow, name, "fail", f"schema violation: {exc.message}", gated)
    except (AssertionError, ValueError, KeyError, RuntimeError, AttributeError, TypeError, IndexError) as exc:
        return Check(workflow, name, "fail", f"{type(exc).__name__}: {exc}", gated)
