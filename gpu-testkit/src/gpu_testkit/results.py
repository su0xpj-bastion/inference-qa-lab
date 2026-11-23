"""The JSONL result store and the recorder that builds it from pytest reports.

One file per run. Line 1 is the run header, every other line is one test::

    {"kind": "run", "run_id": "...", "host": {...}, ...}
    {"kind": "test", "nodeid": "...", "outcome": "failed", "verdict": "infra", ...}

The recorder only consumes ``TestReport`` objects, so it works the same in-process
and on the pytest-xdist controller, where reports arrive serialized from workers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from _pytest.reports import TestReport

ENCODING = "utf-8"
PROP_PREFIX = "gpu_testkit."
SCHEMA_VERSION = 1
QUARANTINE_LABEL_PREFIX = "quarantined"  # QuarantineEntry.label() starts with this


class Outcome:
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"  # setup/teardown failure
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    QUARANTINED = "quarantined"  # failed, but hidden by an active quarantine entry


GATING_OUTCOMES = frozenset({Outcome.FAILED, Outcome.ERROR})


@dataclass
class TestRecord:
    __test__ = False  # not a pytest test class

    nodeid: str
    outcome: str
    verdict: str = ""
    attempts: int = 1
    duration_s: float = 0.0
    message: str = ""
    quarantine: str = ""
    diagnostics: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def gating_failure(self) -> bool:
        return self.outcome in GATING_OUTCOMES


@dataclass
class RunRecord:
    run_id: str
    host: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)
    tests: dict[str, TestRecord] = field(default_factory=dict)


def _props(report: TestReport) -> dict[str, str]:
    return {
        name.removeprefix(PROP_PREFIX): str(value)
        for name, value in report.user_properties
        if name.startswith(PROP_PREFIX)
    }


def _message(report: TestReport) -> str:
    crash = getattr(report.longrepr, "reprcrash", None)
    if crash is not None:
        return str(crash.message).splitlines()[0][:300] if crash.message else ""
    if isinstance(report.longrepr, tuple):
        return str(report.longrepr[2])[:300]
    return ""


class Recorder:
    """Folds setup/call/teardown reports into one TestRecord per node id."""

    def __init__(self) -> None:
        self.tests: dict[str, TestRecord] = {}

    def observe(self, report: TestReport) -> None:
        record = self.tests.setdefault(report.nodeid, TestRecord(nodeid=report.nodeid, outcome=Outcome.PASSED))
        record.duration_s = round(record.duration_s + report.duration, 6)
        if report.failed:
            if record.outcome != Outcome.FAILED:
                record.outcome = Outcome.FAILED if report.when == "call" else Outcome.ERROR
            record.message = record.message or _message(report)
        elif report.skipped and record.outcome == Outcome.PASSED:
            wasxfail = str(getattr(report, "wasxfail", ""))
            if wasxfail.startswith(QUARANTINE_LABEL_PREFIX):
                record.outcome = Outcome.QUARANTINED
            elif wasxfail:
                record.outcome = Outcome.XFAILED
            else:
                record.outcome = Outcome.SKIPPED
            record.message = _message(report) or wasxfail
        if report.when == "teardown":
            self._apply_props(record, _props(report))

    @staticmethod
    def _apply_props(record: TestRecord, props: dict[str, str]) -> None:
        record.verdict = props.pop("verdict", record.verdict)
        record.attempts = int(props.pop("attempts", record.attempts))
        record.quarantine = props.pop("quarantine", record.quarantine)
        record.diagnostics = props.pop("diagnostics", record.diagnostics)
        if record.outcome == Outcome.SKIPPED:
            record.verdict = "skipped"
        record.tags.update(props)


def write_run(path: Path, run: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": "run",
        "schema": SCHEMA_VERSION,
        "run_id": run.run_id,
        "host": run.host,
        "meta": run.meta,
    }
    with path.open("w", encoding=ENCODING) as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for record in sorted(run.tests.values(), key=lambda r: r.nodeid):
            handle.write(json.dumps({"kind": "test", **asdict(record)}, sort_keys=True) + "\n")


def read_run(path: Path) -> RunRecord:
    run: RunRecord | None = None
    tests: dict[str, TestRecord] = {}
    with path.open("r", encoding=ENCODING) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            kind = data.pop("kind", None)
            if kind == "run":
                run = RunRecord(run_id=str(data["run_id"]), host=data.get("host", {}), meta=data.get("meta", {}))
            elif kind == "test":
                tests[data["nodeid"]] = TestRecord(**data)
            else:
                raise ValueError(f"{path}:{number}: unknown record kind {kind!r}")
    if run is None:
        raise ValueError(f"{path}: missing run header line")
    run.tests = tests
    return run


def records(run: RunRecord, outcomes: Iterable[str]) -> list[TestRecord]:
    wanted = set(outcomes)
    return sorted((r for r in run.tests.values() if r.outcome in wanted), key=lambda r: r.nodeid)
