"""Turn a test's attempts into one verdict: passed, flaky, product, or infra.

Rerunning a failure tells you whether it reproduces; it doesn't tell you whose problem
it is. A reproducible failure is ``infra`` only with positive evidence: a hardware Xid
logged during the test, an exception type registered as infra, or a message matching a
known infra signature. Everything else is ``product``. Defaulting to product is
deliberate, because mislabelling a real regression as infra is the expensive mistake.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from gpu_testkit.xid import XidEvent


class Verdict(StrEnum):
    PASSED = "passed"
    FLAKY = "flaky"  # failed, then passed on rerun
    PRODUCT = "product"  # failed every attempt, no infra evidence
    INFRA = "infra"  # failed with infra evidence


class InfraError(Exception):
    """Raise (or subclass) this in fixtures for failures that are the lab's, not the product's."""


DEFAULT_INFRA_EXCEPTIONS = (
    "gpu_testkit.classify.InfraError",
    "gpu_testkit.pool.PoolTimeout",
)
DEFAULT_INFRA_PATTERNS = (
    "uncorrectable ECC error",
    "GPU has fallen off the bus",
    "no CUDA-capable device is detected",
    "CUDA_ERROR_NO_DEVICE",
    "NCCL WARN NET/IB",
    "nvmlInit failed",
)


@dataclass(frozen=True)
class Attempt:
    passed: bool
    exception_types: tuple[str, ...] = ()  # qualified names across the exception's MRO
    message: str = ""
    xids: tuple[XidEvent, ...] = ()  # Xid events logged while this attempt ran


@dataclass(frozen=True)
class Classification:
    verdict: Verdict
    evidence: str = ""


def classify(
    attempts: Sequence[Attempt],
    infra_exceptions: Sequence[str] = DEFAULT_INFRA_EXCEPTIONS,
    infra_patterns: Sequence[str] = DEFAULT_INFRA_PATTERNS,
) -> Classification:
    if not attempts:
        raise ValueError("classify() needs at least one attempt")
    final = attempts[-1]
    if final.passed:
        failed = sum(1 for attempt in attempts if not attempt.passed)
        if failed:
            return Classification(Verdict.FLAKY, f"failed {failed} of {len(attempts)} attempts")
        return Classification(Verdict.PASSED)
    for attempt in attempts:
        hardware = [event for event in attempt.xids if event.hardware]
        if hardware:
            return Classification(Verdict.INFRA, hardware[0].summary())
        for name in attempt.exception_types:
            if name in infra_exceptions:
                return Classification(Verdict.INFRA, f"exception type {name}")
        for pattern in infra_patterns:
            if pattern in attempt.message:
                return Classification(Verdict.INFRA, f"message matches {pattern!r}")
    return Classification(Verdict.PRODUCT, f"reproduced on {len(attempts)} of {len(attempts)} attempts")


def exception_names(exc_type: type[BaseException]) -> tuple[str, ...]:
    """Qualified and bare names for every class in the MRO, so subclasses match too."""
    names: list[str] = []
    for klass in exc_type.__mro__:
        if klass is object:
            continue
        names.append(f"{klass.__module__}.{klass.__qualname__}")
        names.append(klass.__qualname__)
    return tuple(names)
