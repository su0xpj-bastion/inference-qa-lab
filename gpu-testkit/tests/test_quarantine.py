"""Quarantine file rules: owner and expiry are mandatory, matching is by node id boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_testkit.quarantine import QuarantineEntry, QuarantineError, QuarantineList, load

VALID = """
[[quarantine]]
test = "tests/test_comms.py::test_allreduce"
owner = "@comms"
reason = "NCCL timeout on old IB firmware"
ticket = "GPU-1432"
expires = "2026-12-01"
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "quarantine.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_entries(tmp_path: Path) -> None:
    [entry] = load(write(tmp_path, VALID))
    assert entry.owner == "@comms"
    assert entry.ticket == "GPU-1432"
    assert entry.label() == "quarantined GPU-1432 (@comms, until 2026-12-01): NCCL timeout on old IB firmware"


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (VALID.replace('owner = "@comms"\n', ""), "missing owner"),
        (VALID.replace('expires = "2026-12-01"', 'expires = "next sprint"'), "is not YYYY-MM-DD"),
        (VALID.replace('expires = "2026-12-01"', 'expires = "2026-02-30"'), "is not YYYY-MM-DD"),
        (VALID + VALID, "duplicate entry"),
        ("[[quarantine]\n", "quarantine.toml"),
    ],
)
def test_rejects_entries_that_would_rot(tmp_path: Path, text: str, error: str) -> None:
    with pytest.raises(QuarantineError, match=error):
        load(write(tmp_path, text))


@pytest.mark.parametrize(
    ("nodeid", "matches"),
    [
        ("tests/test_comms.py::test_allreduce", True),
        ("tests/test_comms.py::test_allreduce[8gpu]", True),
        ("tests/test_comms.py::test_allreduce_bf16", False),  # prefix, but not at a boundary
        ("tests/test_comms.py::test_allgather", False),
    ],
)
def test_matching_respects_node_id_boundaries(nodeid: str, matches: bool) -> None:
    entry = QuarantineEntry("tests/test_comms.py::test_allreduce", "@o", "r", "2026-12-01")
    assert entry.matches(nodeid) is matches


def test_file_level_entry_covers_every_test_in_it() -> None:
    entry = QuarantineEntry("tests/test_comms.py", "@o", "r", "2026-12-01")
    assert entry.matches("tests/test_comms.py::TestRing::test_one")


def test_expired_entries_stop_applying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path, VALID)
    monkeypatch.setenv("GPU_TESTKIT_TODAY", "2026-12-01")
    assert QuarantineList.from_path(path).lookup("tests/test_comms.py::test_allreduce") is not None
    monkeypatch.setenv("GPU_TESTKIT_TODAY", "2026-12-02")
    later = QuarantineList.from_path(path)
    assert later.lookup("tests/test_comms.py::test_allreduce") is None
    assert [e.test for e in later.expired_entries()] == ["tests/test_comms.py::test_allreduce"]


def test_stale_entries_match_nothing_that_ran(tmp_path: Path) -> None:
    quarantine = QuarantineList.from_path(write(tmp_path, VALID))
    assert quarantine.stale_entries({"tests/test_comms.py::test_allgather"}) == quarantine.entries
    assert quarantine.stale_entries({"tests/test_comms.py::test_allreduce"}) == []
