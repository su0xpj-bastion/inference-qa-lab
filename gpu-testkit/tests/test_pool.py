"""GPU leases: exclusive, all-or-nothing, released by the kernel when a holder dies."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpu_testkit.pool import DevicePool, PoolTimeout


@pytest.fixture
def pool(tmp_path: Path) -> DevicePool:
    return DevicePool(tmp_path / "locks", timeout_s=0.3, poll_s=0.01)


def test_leases_are_exclusive(pool: DevicePool) -> None:
    first = pool.acquire([0, 1, 2, 3], 2, owner="a")
    second = pool.acquire([0, 1, 2, 3], 2, owner="b")
    assert set(first.devices).isdisjoint(second.devices)
    assert second.cuda_visible_devices == "2,3"
    first.release()
    second.release()


def test_acquisition_is_all_or_nothing(pool: DevicePool) -> None:
    """A partial set is released, so two workers can't deadlock holding half each."""
    held = pool.acquire([0, 1, 2, 3], 3, owner="big")
    assert pool.try_acquire([0, 1, 2, 3], 2, owner="pair") is None
    single = pool.try_acquire([0, 1, 2, 3], 1, owner="single")  # device 3 was not left locked
    assert single is not None
    assert single.devices == (3,)
    single.release()
    held.release()


def test_timeout_names_the_holders(pool: DevicePool) -> None:
    with pool.lease([0], 1, owner="tests/test_a.py::test_hog"), pytest.raises(PoolTimeout) as info:
        pool.acquire([0], 1, owner="tests/test_b.py::test_waiter")
    assert "tests/test_a.py::test_hog (pid" in str(info.value)


def test_release_frees_devices(pool: DevicePool) -> None:
    with pool.lease([0, 1], 2, owner="a"):
        assert pool.holders([0, 1]).keys() == {0, 1}
    assert pool.holders([0, 1]) == {}


def test_dead_holder_does_not_leak_a_gpu(pool: DevicePool) -> None:
    """flock is released by the kernel on process exit, even on SIGKILL."""
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from gpu_testkit.pool import DevicePool\n"
        "lease = DevicePool(Path(sys.argv[1])).acquire([0], 1, owner='child')\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", script, str(pool.lock_dir)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        assert pool.try_acquire([0], 1, owner="parent") is None
        assert "child" in pool.holders([0])[0]
    finally:
        child.kill()
        child.wait()
    deadline = time.monotonic() + 5
    lease = None
    while lease is None and time.monotonic() < deadline:
        lease = pool.try_acquire([0], 1, owner="parent")
    assert lease is not None
    lease.release()


def test_requesting_more_than_exist_fails_fast(pool: DevicePool) -> None:
    with pytest.raises(ValueError, match="requested 3 devices from a pool of 2"):
        pool.acquire([0, 1], 3, owner="x")
