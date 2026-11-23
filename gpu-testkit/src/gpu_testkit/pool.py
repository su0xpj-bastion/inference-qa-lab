"""Exclusive GPU leases backed by per-device ``flock`` files.

Every pytest-xdist worker is its own process, and two CI jobs can share one GPU box,
so in-process bookkeeping is not enough. Each device gets a lock file; a lease holds
an exclusive non-blocking ``flock`` on each of its devices. The kernel drops the locks
when a process dies, so a crashed worker can't leak a GPU.

Acquisition is all-or-nothing: if a test needs 4 devices and only 3 are free, it
releases the 3 and retries, so two workers never deadlock holding half a set each.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

ENCODING = "utf-8"
LOCK_DIR_ENV = "GPU_TESTKIT_LOCK_DIR"
DEFAULT_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 0.05


class PoolTimeout(TimeoutError):
    """No qualifying set of devices became free in time."""


@dataclass
class Lease:
    devices: tuple[int, ...]
    waited_s: float
    _handles: list[IO[str]] = field(default_factory=list, repr=False)

    @property
    def cuda_visible_devices(self) -> str:
        return ",".join(str(index) for index in self.devices)

    def release(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def default_lock_dir(hostname: str) -> Path:
    root = os.environ.get(LOCK_DIR_ENV)
    base = Path(root) if root else Path(tempfile.gettempdir()) / "gpu-testkit-locks"
    return base / hostname


class DevicePool:
    def __init__(
        self,
        lock_dir: Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        poll_s: float = POLL_INTERVAL_S,
    ) -> None:
        self.lock_dir = lock_dir
        self.timeout_s = timeout_s
        self.poll_s = poll_s

    def _lock_path(self, index: int) -> Path:
        return self.lock_dir / f"gpu-{index}.lock"

    def _try_lock(self, index: int, owner: str) -> IO[str] | None:
        handle = self._lock_path(index).open("a+", encoding=ENCODING)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"owner": owner, "pid": os.getpid(), "since": time.time()}))
        handle.flush()
        return handle

    def try_acquire(self, candidates: Sequence[int], count: int, owner: str) -> Lease | None:
        """One non-blocking pass: lock ``count`` of ``candidates`` or hold nothing."""
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        held: list[tuple[int, IO[str]]] = []
        for index in candidates:
            handle = self._try_lock(index, owner)
            if handle is None:
                continue
            held.append((index, handle))
            if len(held) == count:
                return Lease(devices=tuple(i for i, _ in held), waited_s=0.0, _handles=[h for _, h in held])
        Lease(devices=(), waited_s=0.0, _handles=[h for _, h in held]).release()
        return None

    def acquire(self, candidates: Sequence[int], count: int, owner: str) -> Lease:
        if count > len(candidates):
            raise ValueError(f"requested {count} devices from a pool of {len(candidates)}")
        start = time.monotonic()
        while True:
            lease = self.try_acquire(candidates, count, owner)
            if lease is not None:
                lease.waited_s = time.monotonic() - start
                return lease
            if time.monotonic() - start >= self.timeout_s:
                raise PoolTimeout(
                    f"{owner}: no {count} free devices among {list(candidates)} after "
                    f"{self.timeout_s:g}s; holders: {self.holders(candidates)}"
                )
            time.sleep(self.poll_s)

    @contextmanager
    def lease(self, candidates: Sequence[int], count: int, owner: str) -> Iterator[Lease]:
        lease = self.acquire(candidates, count, owner)
        try:
            yield lease
        finally:
            lease.release()

    def holders(self, indices: Sequence[int]) -> dict[int, str]:
        """Who holds which device right now (for timeouts and ``gpu-testkit pool``)."""
        result: dict[int, str] = {}
        for index in indices:
            path = self._lock_path(index)
            if not path.exists():
                continue
            with path.open("a+", encoding=ENCODING) as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.seek(0)
                    text = handle.read()
                    try:
                        info = json.loads(text)
                        result[index] = f"{info['owner']} (pid {info['pid']})"
                    except (ValueError, KeyError):
                        result[index] = "unknown holder"
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return result
