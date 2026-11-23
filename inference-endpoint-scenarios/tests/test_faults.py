"""Each workflow passes a healthy endpoint and catches the regression it exists for.

FAULT_CATCH pins, per injected fault, exactly which checks go red. If a refactor
makes a check stop seeing its bug class, or start failing on a healthy server,
this goes red. A suite that silently loses a detector is worse than no suite.
"""
from __future__ import annotations

import pytest

from ies.core import verdict
from ies.workflows import WORKFLOWS

# fault -> (workflows to run, exact set of failing checks)
FAULT_CATCH = {
    "slow_ttft": (["accelerator"], {"accelerator.ttft"}),
    "slow_decode": (["accelerator"], {"accelerator.decode_throughput"}),
    "usage_drift": (["inference"], {"inference.usage_accounting"}),
    "stream_no_usage": (["inference", "accelerator"],
                        {"inference.stream_usage_parity", "accelerator.decode_throughput"}),
    "ignore_max_tokens": (["inference"], {"inference.max_tokens_length"}),
    "midstream_error": (["inference", "accelerator"],
                        {"inference.stream_usage_parity", "accelerator.decode_throughput", "accelerator.ttft"}),
    "nondeterministic": (["accelerator"], {"accelerator.determinism", "accelerator.load_invariance",
                                           "accelerator.logprob_baseline"}),
    "logprob_noise": (["accelerator"], {"accelerator.logprob_stability", "accelerator.logprob_baseline"}),
    "contention": (["cluster"], {"cluster.concurrency_latency"}),
    "crosstalk": (["cluster"], {"cluster.request_isolation"}),
    "gzip_ignored": (["wire"], {"wire.wire_equivalence", "wire.wire_ttft"}),
    "msgpack_as_json": (["wire"], {"wire.wire_equivalence", "wire.wire_ttft"}),
    "decompress_truncation": (["wire"], {"wire.wire_equivalence"}),     # 200 OK, silently shorter prompt
    "bad_tool_args": (["agent"], {"agent.agent_tool_args"}),           # model recovers; validator still flags
    "agent_loop_stall": (["agent"], {"agent.agent_task_success", "agent.agent_loop_latency"}),
    "ignore_tool_results": (["agent"], {"agent.agent_task_success"}),
    "max_tokens_alias_ignored": (["compat"], {"compat.compat_max_tokens"}),
    "history_truncated": (["conversation"], {"conversation.conversation_recall"}),
    "cache_miss": (["conversation"], {"conversation.conversation_cache_growth",
                                      "conversation.conversation_prefill_flat"}),
    "context_truncated": (["longctx"], {"longctx.needle_recall", "longctx.context_accounting"}),
    "lossy_gist": (["longctx"], {"longctx.gist_chain"}),
    "utf8_split": (["inference"], {"inference.stream_integrity"}),
    "stop_chunk_boundary": (["inference"], {"inference.stream_integrity"}),
    "logprobs_unsorted": (["inference"], {"inference.logprobs_contract"}),
    # leaked text is billed as completion tokens, so the budget check trips too
    "reasoning_leak": (["reasoning"], {"reasoning.reasoning_separation", "reasoning.reasoning_budget"}),
    "harmony_markers": (["reasoning"], {"reasoning.reasoning_separation", "reasoning.reasoning_budget"}),
    "reasoning_unbilled": (["reasoning"], {"reasoning.reasoning_billing"}),
    "effort_ignored": (["reasoning"], {"reasoning.reasoning_effort_scaling"}),
    "budget_stop": (["reasoning"], {"reasoning.reasoning_budget"}),
    # one off-spec enum value makes the typed SDK give up on the whole chunk
    "schema_drift": (["inference", "schema"], {"inference.stream_usage_parity", "inference.tool_call_streamed",
                                               "inference.stream_integrity", "schema.schema_conformance"}),
    "batch_variant": (["accelerator"], {"accelerator.load_invariance"}),
    "numerics_shift": (["accelerator"], {"accelerator.logprob_baseline"}),
    "fixed_window": (["cluster"], {"cluster.rate_limit_refill"}),
    "latency_creep": (["soak"], {"soak.soak_latency_creep"}),
    "error_burst": (["soak"], {"soak.soak_errors"}),
}


def _failing(ctx, workflows):
    checks = [c for w in workflows for c in WORKFLOWS[w](ctx)]
    return checks, {f"{c.workflow}.{c.name}" for c in checks if c.status == "fail"}


@pytest.mark.parametrize("workflow", list(WORKFLOWS))
def test_healthy_endpoint_passes(make_ctx, workflow):
    checks = WORKFLOWS[workflow](make_ctx())
    assert verdict(checks) in ("PASS", "PASS (warnings)"), [c for c in checks if c.status == "fail"]


@pytest.mark.parametrize("fault", list(FAULT_CATCH))
def test_fault_is_caught_by_exactly_the_right_checks(make_ctx, fault):
    workflows, want = FAULT_CATCH[fault]
    checks, failed = _failing(make_ctx({fault}), workflows)
    assert failed == want, [(c.name, c.detail) for c in checks if c.status == "fail"]


def test_429_storm_surfaces_as_errors_not_silence(make_ctx):
    # which other checks also hit an exhausted 429 depends on request interleaving, so pin the two that must
    _, failed = _failing(make_ctx({"rate_limit_storm"}), ["cluster"])
    assert {"cluster.error_rate", "cluster.retry_absorption"} <= failed


def test_transient_429s_are_absorbed_by_retries(make_ctx):
    checks, failed = _failing(make_ctx({"rate_limit_light"}), ["cluster"])
    assert not failed
    retry = next(c for c in checks if c.name == "retry_absorption")
    assert retry.metrics["retry_ratio"] > 0          # retries happened, and hid nothing


def test_release_regression_with_new_backend_is_no_go(make_ctx):
    checks = WORKFLOWS["release"](make_ctx({"regress_math", "fingerprint_v2"}))
    gate = next(c for c in checks if c.name == "release_gate")
    diff = next(c for c in checks if c.name == "baseline_diff")
    assert gate.metrics["gate"] == "NO-GO"
    assert set(gate.metrics["regressions"]) == {"math_mul", "math_add"}
    assert "fp_mock_v1 -> fp_mock_v2" in diff.detail


def test_release_flaky_case_is_inconclusive_not_no_go(make_ctx):
    checks = WORKFLOWS["release"](make_ctx({"flaky_answers"}))
    gate = next(c for c in checks if c.name == "release_gate")
    assert gate.metrics["gate"] == "INCONCLUSIVE" and gate.metrics["flaky"] == ["capital"]


def test_new_fingerprint_alone_is_go_with_a_warning(make_ctx):
    checks = WORKFLOWS["release"](make_ctx({"fingerprint_v2"}))
    assert next(c for c in checks if c.name == "release_gate").metrics["gate"] == "GO"
    assert next(c for c in checks if c.name == "baseline_diff").status == "warn"


def test_missing_baseline_warns_but_does_not_block(make_ctx, tmp_path):
    checks = WORKFLOWS["release"](make_ctx(baseline_dir=str(tmp_path)))
    assert next(c for c in checks if c.name == "baseline_diff").status == "warn"
    assert next(c for c in checks if c.name == "release_gate").metrics["gate"] == "GO"


def test_numerics_move_on_a_new_build_is_reported_not_failed(make_ctx):
    checks = WORKFLOWS["accelerator"](make_ctx({"numerics_shift", "fingerprint_v2"}))
    c = next(c for c in checks if c.name == "logprob_baseline")
    assert c.status == "warn" and "new backend build" in c.detail and c.metrics["drift"] > 0.02


def test_removed_model_is_a_deprecation_warning(make_ctx):
    c = next(c for c in WORKFLOWS["release"](make_ctx({"model_removed"})) if c.name == "models_diff")
    assert c.status == "warn" and "gpt-oss-120b" in c.detail


@pytest.mark.parametrize("fault,gate", [(None, "GO"), ("benchmark_noise", "GO"),
                                        ("benchmark_drift", "INCONCLUSIVE"), ("benchmark_regression", "NO-GO")])
def test_benchmark_gate_separates_noise_from_regression(make_ctx, fault, gate):
    c = WORKFLOWS["benchmark"](make_ctx({fault} if fault else set()))[0]
    assert c.metrics["gate"] == gate, c.detail
