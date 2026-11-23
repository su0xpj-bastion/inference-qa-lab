"""Hooks gpu-testkit adds to pytest. Implement them in a conftest.py or plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from gpu_testkit.diagnostics import Collector


@pytest.hookspec
def pytest_gpu_testkit_diagnostic_collectors(config: pytest.Config) -> list[Collector]:  # type: ignore[empty-body]
    """Return extra diagnostics collectors. Results from every implementation are combined.

    Each collector has a ``name`` and ``collect(ctx) -> {filename: str | bytes}``; it
    runs only when a test finally fails, and its exceptions are recorded, not raised.
    """
