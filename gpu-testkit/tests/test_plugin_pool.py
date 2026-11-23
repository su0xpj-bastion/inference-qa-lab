"""Exclusive device leases under real pytest-xdist workers."""

from __future__ import annotations

import json
from itertools import combinations

import pytest

pytest.importorskip("xdist")

SUITE = """
import json, os, time
import pytest

LOG = {log!r}

@pytest.mark.parametrize("i", range(12))
@pytest.mark.gpus(2)
def test_job(i, gpu_devices):
    start = time.monotonic()
    time.sleep(0.15)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({{"i": i, "devices": list(gpu_devices), "start": start, "end": time.monotonic(),
                             "worker": os.environ.get("PYTEST_XDIST_WORKER"),
                             "cvd": os.environ["CUDA_VISIBLE_DEVICES"]}}) + "\\n")
"""


def test_xdist_workers_never_share_a_gpu(pytester: pytest.Pytester) -> None:
    """4 workers, 4 fake GPUs, 2 GPUs per test: at most 2 tests can run at once."""
    log = pytester.path / "leases.jsonl"
    pytester.makepyfile(SUITE.format(log=str(log)))
    result = pytester.runpytest("-n", "4", "--gpu-fake-spec=h100x4", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=12)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    assert len({row["worker"] for row in rows}) > 1, "test did not actually run in parallel"
    for row in rows:
        assert row["cvd"] == ",".join(map(str, row["devices"]))
        assert set(row["devices"]) <= {0, 1, 2, 3}
    for a, b in combinations(rows, 2):
        overlapping_in_time = a["start"] < b["end"] and b["start"] < a["end"]
        if overlapping_in_time:
            assert set(a["devices"]).isdisjoint(b["devices"]), (a, b)


def test_pool_timeout_is_classified_as_infra(pytester: pytest.Pytester) -> None:
    """A test that can't get GPUs is a lab problem, not a product failure."""
    pytester.makeconftest("""
        import pytest
        from gpu_testkit.pool import DevicePool

        @pytest.fixture(autouse=True, scope="session")
        def hog(pytestconfig):
            pool = DevicePool(pytestconfig.stash[__import__("gpu_testkit.plugin").plugin.PLUGIN_KEY].pool.lock_dir)
            with pool.lease([0], 1, owner="someone-else"):
                yield
    """)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.gpus(1)
        def test_starved():
            pass
    """)
    result = pytester.runpytest("--gpu-fake-spec=h100x1", "--gpu-pool-timeout=0.2", "--junitxml=j.xml")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(
        [
            "*PoolTimeout*holders: {0: 'someone-else (pid*",
            "INFRA  *::test_starved: exception type gpu_testkit.pool.PoolTimeout",
        ]
    )
