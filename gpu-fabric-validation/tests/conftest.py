from __future__ import annotations

from pathlib import Path

import pytest

ENCODING = "utf-8"
FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    with (FIXTURES / rel).open("r", encoding=ENCODING) as fh:
        return fh.read()


@pytest.fixture
def fixture_text():
    return read
