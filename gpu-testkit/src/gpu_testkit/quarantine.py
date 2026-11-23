"""Quarantine: known-bad tests keep running, stop gating, and stay visible.

The file is TOML so humans edit it in review::

    [[quarantine]]
    test = "tests/test_nccl.py::test_allreduce_8gpu"
    owner = "@comms-team"
    reason = "NCCL timeout on nodes with old IB firmware"
    ticket = "GPU-1432"
    expires = "2026-12-01"

``owner`` and ``expires`` are mandatory, so nothing is quarantined forever by nobody.
After ``expires`` the entry stops applying and the test gates again.
"""

from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

TODAY_ENV = "GPU_TESTKIT_TODAY"  # pin "today" in tests and reproducible reruns
DATE_FORMAT = "%Y-%m-%d"
REQUIRED_FIELDS = ("test", "owner", "reason", "expires")


class QuarantineError(ValueError):
    """The quarantine file is malformed."""


@dataclass(frozen=True)
class QuarantineEntry:
    test: str
    owner: str
    reason: str
    expires: str
    ticket: str = ""

    def matches(self, nodeid: str) -> bool:
        """Exact node id, or a prefix that ends at a ``::`` or ``[`` boundary."""
        if nodeid == self.test:
            return True
        return nodeid.startswith(self.test) and nodeid[len(self.test)] in ":["

    def expired(self, today: str) -> bool:
        return today > self.expires

    def label(self) -> str:
        ticket = f" {self.ticket}" if self.ticket else ""
        return f"quarantined{ticket} ({self.owner}, until {self.expires}): {self.reason}"


def today() -> str:
    return os.environ.get(TODAY_ENV) or time.strftime(DATE_FORMAT)


def _valid_date(value: str) -> bool:
    try:
        time.strptime(value, DATE_FORMAT)
    except ValueError:
        return False
    return True


def load(path: Path) -> list[QuarantineEntry]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise QuarantineError(f"{path}: {exc}") from exc
    entries: list[QuarantineEntry] = []
    seen: set[str] = set()
    for position, raw in enumerate(data.get("quarantine", []), start=1):
        where = f"{path} entry {position}"
        missing = [name for name in REQUIRED_FIELDS if not str(raw.get(name, "")).strip()]
        if missing:
            raise QuarantineError(f"{where}: missing {', '.join(missing)}")
        expires = str(raw["expires"])
        if not _valid_date(expires):
            raise QuarantineError(f"{where}: expires={expires!r} is not YYYY-MM-DD")
        if raw["test"] in seen:
            raise QuarantineError(f"{where}: duplicate entry for {raw['test']}")
        seen.add(raw["test"])
        entries.append(
            QuarantineEntry(
                test=str(raw["test"]),
                owner=str(raw["owner"]),
                reason=str(raw["reason"]),
                expires=expires,
                ticket=str(raw.get("ticket", "")),
            )
        )
    return entries


@dataclass
class QuarantineList:
    entries: list[QuarantineEntry]
    today: str

    @classmethod
    def from_path(cls, path: Path | None) -> QuarantineList:
        return cls(entries=load(path) if path else [], today=today())

    def lookup(self, nodeid: str) -> QuarantineEntry | None:
        """The entry that applies to ``nodeid``, ignoring expired ones."""
        for entry in self.entries:
            if entry.matches(nodeid) and not entry.expired(self.today):
                return entry
        return None

    def expired_entries(self) -> list[QuarantineEntry]:
        return [entry for entry in self.entries if entry.expired(self.today)]

    def stale_entries(self, collected: set[str]) -> list[QuarantineEntry]:
        """Entries that match no collected test: renamed or deleted tests."""
        return [e for e in self.entries if not any(e.matches(nodeid) for nodeid in collected)]
