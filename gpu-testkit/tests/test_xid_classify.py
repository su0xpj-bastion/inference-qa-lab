"""Xid parsing against captured dmesg lines, and the flaky/product/infra verdicts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_testkit.classify import Attempt, InfraError, Verdict, classify, exception_names
from gpu_testkit.xid import parse_xids


def test_parses_xids_from_captured_dmesg(fixtures_dir: Path) -> None:
    events = parse_xids((fixtures_dir / "dmesg-xid.log").read_text(encoding="utf-8").splitlines())
    assert [(e.code, e.pci, e.hardware) for e in events] == [
        (31, "0000:3b:00", False),
        (79, "0000:5e:00", True),
        (74, "0000:86:00", True),
        (94, "0000:af:00", True),
    ]
    assert events[0].pid == "48211"
    assert "fallen off the bus" in events[1].summary()


def test_flaky_when_a_later_attempt_passes() -> None:
    result = classify([Attempt(passed=False), Attempt(passed=True)])
    assert result.verdict is Verdict.FLAKY
    assert result.evidence == "failed 1 of 2 attempts"


def test_reproducible_failure_without_evidence_is_product() -> None:
    result = classify([Attempt(passed=False, message="AssertionError: logits differ")] * 3)
    assert result.verdict is Verdict.PRODUCT


def test_hardware_xid_makes_it_infra(fixtures_dir: Path) -> None:
    xids = tuple(parse_xids(["[1.0] NVRM: Xid (PCI:0000:5e:00): 79, pid=0, GPU has fallen off the bus."]))
    result = classify([Attempt(passed=False, xids=xids)])
    assert result.verdict is Verdict.INFRA
    assert "Xid 79" in result.evidence


def test_application_xid_stays_product() -> None:
    """Xid 31 (MMU fault) is usually the kernel under test reading a bad address."""
    xids = tuple(parse_xids(["NVRM: Xid (PCI:0000:3b:00): 31, pid=1, name=python3, Ch 00000010"]))
    assert classify([Attempt(passed=False, xids=xids)]).verdict is Verdict.PRODUCT


def test_infra_exception_matches_subclasses() -> None:
    class NodeLostError(InfraError):
        pass

    result = classify([Attempt(passed=False, exception_types=exception_names(NodeLostError))])
    assert result.verdict is Verdict.INFRA
    assert result.evidence == "exception type gpu_testkit.classify.InfraError"


def test_message_pattern_makes_it_infra() -> None:
    attempt = Attempt(passed=False, message="RuntimeError: CUDA error: uncorrectable ECC error encountered")
    assert classify([attempt]).verdict is Verdict.INFRA


def test_classify_needs_attempts() -> None:
    with pytest.raises(ValueError, match="at least one attempt"):
        classify([])
