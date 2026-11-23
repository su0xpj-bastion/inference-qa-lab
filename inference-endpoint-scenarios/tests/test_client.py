"""Client behaviour (retry policy, capture) and the pure helpers."""
from __future__ import annotations

import math

import cerebras.cloud.sdk as cerebras
import pytest

from ies import cli
from ies.client import ENCODINGS, MAX_ATTEMPTS, Budget, EncodingTransport, chat, make_client
from ies.core import pct, tbt
from mockserve.server import API_KEY


def test_429_is_retried_honouring_retry_after(mock):
    base_url, state = mock
    state.faults = {"rate_limit_light"}
    client = make_client(API_KEY, base_url)
    attempts = [chat(client, "qwen-3.8-27b", [{"role": "user", "content": f"q{i}"}]).attempts for i in range(6)]
    assert max(attempts) == 2 and attempts.count(2) == 2      # every 3rd request retried once


def test_429_storm_raises_after_max_attempts(mock):
    base_url, state = mock
    state.faults = {"rate_limit_storm"}
    state.requests = 0
    with pytest.raises(cerebras.RateLimitError):
        chat(make_client(API_KEY, base_url), "qwen-3.8-27b", [{"role": "user", "content": "x"}])
    assert state.requests == MAX_ATTEMPTS


def test_non_429_errors_are_not_retried(mock):
    base_url, state = mock
    state.requests = 0
    with pytest.raises(cerebras.NotFoundError):
        chat(make_client(API_KEY, base_url), "nope", [{"role": "user", "content": "x"}])
    assert state.requests == 1


def test_midstream_error_raises_instead_of_truncating(mock):
    base_url, state = mock
    state.faults = {"midstream_error"}
    with pytest.raises(cerebras.APIStatusError):
        chat(make_client(API_KEY, base_url), "qwen-3.8-27b",
             [{"role": "user", "content": "Write a detailed essay."}], stream=True)


def test_stream_captures_usage_time_info_and_headers(mock):
    base_url, _ = mock
    c = chat(make_client(API_KEY, base_url), "qwen-3.8-27b",
             [{"role": "user", "content": "Write a detailed essay."}], stream=True, max_completion_tokens=50,
             reasoning_effort="none")
    assert c.usage_seen and c.completion_tokens == 50 and c.finish_reason == "length"
    assert c.completion_time and c.ttft is not None and c.fingerprint == "fp_mock_v1"
    assert "x-ratelimit-remaining-requests-minute" in c.headers


def test_budget_caps_requests():
    b = Budget(2)
    b.take()
    b.take()
    with pytest.raises(RuntimeError):
        b.take()


def test_cost_is_per_thousand_split_by_type():
    # 1,000 prompt + 2,000 completion tokens of qwen-3.8-27b at $0.99 / $1.49 per M
    assert cli.cost("qwen-3.8-27b", 1000, 2000) == pytest.approx(0.00099 + 2 * 0.00149)
    assert cli.cost("unknown", 1, 1) is None


def test_pct_nearest_rank_and_tbt():
    assert pct([5, 1, 3, 2, 4], 50) == 3 and pct([1, 2, 3, 4], 95) == 4
    assert math.isnan(pct([], 50))
    assert tbt([0.1, 0.2, 0.4]) == pytest.approx(0.15) and tbt([0.1]) == 0.0


def test_cli_run_mock_exit_codes(tmp_path):
    assert cli.main(["run", "--target", "mock", "-w", "inference", "--out", str(tmp_path)]) == 0
    assert cli.main(["run", "--target", "mock", "-w", "inference", "--faults", "usage_drift",
                     "--out", str(tmp_path)]) == 1
    assert "usage_accounting" in (tmp_path / "report.md").read_text()


def test_cli_live_without_key_is_a_clean_error(monkeypatch, tmp_path):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    assert cli.main(["run", "--target", "live", "--out", str(tmp_path)]) == 2


@pytest.mark.parametrize("enc", [e for e in ENCODINGS if e != "json"])
def test_encoded_bodies_reach_the_server_as_the_same_request(mock, enc):
    base_url, _ = mock
    msgs = [{"role": "user", "content": "compress me " * 400}]
    plain = chat(make_client(API_KEY, base_url), "qwen-3.8-27b", msgs, temperature=0, max_completion_tokens=8)
    client = make_client(API_KEY, base_url, enc)
    encoded = chat(client, "qwen-3.8-27b", msgs, temperature=0, max_completion_tokens=8)
    assert encoded.prompt_tokens == plain.prompt_tokens and encoded.content == plain.content
    transport = client._client._transport
    assert isinstance(transport, EncodingTransport) and transport.sent
    json_bytes, wire_bytes = transport.sent[-1]
    if "gzip" in enc:
        assert wire_bytes < json_bytes / 5          # highly repetitive prompt


def test_sdk_older_than_minimum_is_refused_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(cerebras, "__version__", "1.50.1")
    with pytest.raises(RuntimeError, match="older than the supported minimum 1.59.0"):
        make_client(API_KEY, "http://127.0.0.1:1")


def test_benchmark_statistics():
    from ies.workflows.benchmark import drop_p_value, extract, wilson
    assert extract("so 3 + 4 = 7\nAnswer: $1,234") == "1234" and extract("Answer: 5.0") == "5"
    assert extract("no final line") is None
    lo, hi = wilson(47, 50)
    assert 0.83 < lo < 0.85 and 0.97 < hi < 0.99
    assert drop_p_value(50, 50, 49, 50) > 0.10          # one flip: noise
    assert 0.01 < drop_p_value(50, 50, 47, 50) < 0.10   # three: inconclusive
    assert drop_p_value(50, 50, 35, 50) < 0.01          # fifteen: regression
    assert drop_p_value(40, 50, 45, 50) == pytest.approx(1 - drop_p_value(45, 50, 40, 50), abs=1e-9)


def test_all_excludes_soak_and_schema_runs_last(tmp_path, monkeypatch):
    from ies import cli
    from ies.workflows import DEFAULT
    ran = []
    monkeypatch.setattr(cli, "WORKFLOWS", {n: (lambda ctx, n=n: ran.append(n) or []) for n in cli.WORKFLOWS})
    assert cli.main(["run", "--target", "mock", "-w", "schema,all", "--out", str(tmp_path)]) == 0
    assert "soak" not in ran and ran[-1] == "schema" and set(ran) == set(DEFAULT)
    assert cli.main(["run", "--target", "mock", "-w", "nope", "--out", str(tmp_path)]) == 2
