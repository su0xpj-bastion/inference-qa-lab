"""Diff two runs and render the result as markdown for a CI job summary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from gpu_testkit.results import Outcome, RunRecord, TestRecord

DEFAULT_SLOW_RATIO = 1.5
DEFAULT_SLOW_MIN_DELTA_S = 0.5
DEFAULT_TOP = 10
OUTCOME_ORDER = (
    Outcome.PASSED,
    Outcome.FAILED,
    Outcome.ERROR,
    Outcome.QUARANTINED,
    Outcome.SKIPPED,
    Outcome.XFAILED,
)


@dataclass(frozen=True)
class Slowdown:
    nodeid: str
    base_s: float
    head_s: float

    @property
    def ratio(self) -> float:
        return self.head_s / self.base_s if self.base_s else float("inf")


@dataclass
class RunDiff:
    head: RunRecord
    base: RunRecord | None
    new_failures: list[TestRecord] = field(default_factory=list)
    still_failing: list[TestRecord] = field(default_factory=list)
    fixed: list[TestRecord] = field(default_factory=list)
    flaky: list[TestRecord] = field(default_factory=list)
    infra: list[TestRecord] = field(default_factory=list)
    quarantine_hidden: list[TestRecord] = field(default_factory=list)
    quarantine_passing: list[TestRecord] = field(default_factory=list)
    lost_coverage: list[TestRecord] = field(default_factory=list)  # passed in base, skipped now
    removed: list[str] = field(default_factory=list)
    slower: list[Slowdown] = field(default_factory=list)

    @property
    def gating(self) -> list[TestRecord]:
        return [r for r in self.head.tests.values() if r.gating_failure]


def diff_runs(
    head: RunRecord,
    base: RunRecord | None = None,
    *,
    slow_ratio: float = DEFAULT_SLOW_RATIO,
    slow_min_delta_s: float = DEFAULT_SLOW_MIN_DELTA_S,
) -> RunDiff:
    result = RunDiff(head=head, base=base)
    base_tests = base.tests if base else {}
    for nodeid in sorted(head.tests):
        record = head.tests[nodeid]
        before = base_tests.get(nodeid)
        if record.gating_failure:
            if before is not None and before.gating_failure:
                result.still_failing.append(record)
            else:
                result.new_failures.append(record)
            if record.verdict == "infra":
                result.infra.append(record)
        elif before is not None and before.gating_failure and record.outcome == Outcome.PASSED:
            result.fixed.append(record)
        if record.verdict == "flaky":
            result.flaky.append(record)
        if record.outcome == Outcome.QUARANTINED:
            result.quarantine_hidden.append(record)
        elif record.quarantine and record.outcome == Outcome.PASSED:
            result.quarantine_passing.append(record)
        if before is not None and before.outcome == Outcome.PASSED and record.outcome == Outcome.SKIPPED:
            result.lost_coverage.append(record)
        if (
            before is not None
            and before.outcome == Outcome.PASSED
            and record.outcome == Outcome.PASSED
            and record.duration_s - before.duration_s >= slow_min_delta_s
            and record.duration_s >= before.duration_s * slow_ratio
        ):
            result.slower.append(Slowdown(nodeid, before.duration_s, record.duration_s))
    result.removed = sorted(set(base_tests) - set(head.tests))
    result.slower.sort(key=lambda s: s.head_s - s.base_s, reverse=True)
    return result


def _counts(run: RunRecord) -> str:
    counts = Counter(r.outcome for r in run.tests.values())
    return " · ".join(f"{counts[o]} {o}" for o in OUTCOME_ORDER if counts[o])


def _section(title: str, rows: list[str], header: str, top: int) -> list[str]:
    if not rows:
        return []
    shown = rows[:top]
    more = [f"\n…and {len(rows) - top} more"] if len(rows) > top else []
    return [
        f"### {title} ({len(rows)})",
        "",
        header,
        "|" + "---|" * header.count("|", 1),
        *shown,
        *more,
        "",
    ]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")[:160]


def render_markdown(diff: RunDiff, *, top: int = DEFAULT_TOP) -> str:
    head = diff.head
    status = "❌ gating failures" if diff.gating else "✅ no gating failures"
    lines = [f"## gpu-testkit: {head.run_id} — {status}", ""]
    host = head.host.get("summary", "")
    if host:
        lines += [f"**Host:** {host}", ""]
    lines += [f"**Head:** {_counts(head)}"]
    if diff.base is not None:
        lines += [f"**Base ({diff.base.run_id}):** {_counts(diff.base)}"]
    lines.append("")

    def failure_row(r: TestRecord) -> str:
        bundle = f"`{r.diagnostics}`" if r.diagnostics else ""
        return f"| `{_cell(r.nodeid)}` | {r.verdict or r.outcome} | {r.attempts} | {_cell(r.message)} | {bundle} |"

    header = "| test | verdict | attempts | message | diagnostics |"
    lines += _section("New failures", [failure_row(r) for r in diff.new_failures], header, top)
    lines += _section("Still failing", [failure_row(r) for r in diff.still_failing], header, top)
    lines += _section(
        "Fixed",
        [f"| `{_cell(r.nodeid)}` | {r.duration_s:.2f}s |" for r in diff.fixed],
        "| test | duration |",
        top,
    )
    lines += _section(
        "Flaky (passed on rerun)",
        [f"| `{_cell(r.nodeid)}` | {r.attempts} | {_cell(r.tags.get('evidence', ''))} |" for r in diff.flaky],
        "| test | attempts | evidence |",
        top,
    )
    lines += _section(
        "Infra failures (not product evidence)",
        [f"| `{_cell(r.nodeid)}` | {_cell(r.tags.get('evidence', ''))} |" for r in diff.infra],
        "| test | evidence |",
        top,
    )
    lines += _section(
        "Quarantine is hiding",
        [f"| `{_cell(r.nodeid)}` | {_cell(r.quarantine)} | {_cell(r.message)} |" for r in diff.quarantine_hidden],
        "| test | entry | failure |",
        top,
    )
    lines += _section(
        "Quarantined but passing (candidates to release)",
        [f"| `{_cell(r.nodeid)}` | {_cell(r.quarantine)} |" for r in diff.quarantine_passing],
        "| test | entry |",
        top,
    )
    lines += _section(
        "Lost coverage (passed in base, skipped now)",
        [f"| `{_cell(r.nodeid)}` | {_cell(r.message)} |" for r in diff.lost_coverage],
        "| test | skip reason |",
        top,
    )
    lines += _section(
        "Slowest regressions",
        [f"| `{_cell(s.nodeid)}` | {s.base_s:.2f}s | {s.head_s:.2f}s | x{s.ratio:.2f} |" for s in diff.slower],
        "| test | base | head | ratio |",
        top,
    )
    if diff.removed:
        lines += [
            f"### Removed since base ({len(diff.removed)})",
            "",
            *[f"- `{n}`" for n in diff.removed[:top]],
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"
