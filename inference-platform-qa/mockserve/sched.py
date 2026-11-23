"""Request batch scheduler for mockserve: the 'server scheduling overhead' under test.

Real engines batch requests before prefill: more requests per forward pass
means more throughput. The policy decides how long a request may wait for
company. Three policies:

  immediate   dispatch on arrival (no batching). The default; existing tests assume it.
  fixed-wait  open a batch window at the first arrival; dispatch when the batch
              is full OR the window (MOCKSERVE_BATCH_WAIT_MS) expires.
              The seeded regression: at low or medium load the batch never fills,
              so EVERY request pays the whole window as extra TTFT.
  adaptive    the fix: also dispatch as soon as arrivals go quiet for
              MOCKSERVE_IDLE_GAP_MS. Bursts still batch, and a lone request isn't held hostage.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

POLICIES = ("immediate", "fixed-wait", "adaptive")


@dataclass
class _Ticket:
    arrived: float
    event: threading.Event = field(default_factory=threading.Event)


class BatchScheduler:
    def __init__(self, policy: str = "immediate", max_batch: int = 8,
                 batch_wait_s: float = 0.35, idle_gap_s: float = 0.005) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.policy = policy
        self.max_batch = max_batch
        self.batch_wait_s = batch_wait_s
        self.idle_gap_s = idle_gap_s
        self._cond = threading.Condition()
        self._queue: list[_Ticket] = []
        self.batches: list[int] = []          # dispatched batch sizes, for metrics and tests
        if policy != "immediate":
            threading.Thread(target=self._dispatch_loop, daemon=True, name="batch-dispatcher").start()

    def admit(self) -> float:
        """Block until this request's batch is dispatched. Returns seconds spent waiting."""
        t0 = time.monotonic()
        if self.policy == "immediate":
            return 0.0
        ticket = _Ticket(arrived=t0)
        with self._cond:
            self._queue.append(ticket)
            self._cond.notify()
        ticket.event.wait()
        return time.monotonic() - t0

    def _due_in(self, now: float) -> float:
        """Seconds until the current batch must be dispatched (<= 0 means now)."""
        first, last = self._queue[0].arrived, self._queue[-1].arrived
        if len(self._queue) >= self.max_batch:
            return 0.0
        window = first + self.batch_wait_s - now
        if self.policy == "fixed-wait":
            return window
        return min(window, last + self.idle_gap_s - now)

    def _dispatch_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                due = self._due_in(time.monotonic())
                if due > 0:
                    self._cond.wait(timeout=due)   # a new arrival may wake us early
                    continue
                batch, self._queue = self._queue[: self.max_batch], self._queue[self.max_batch:]
                self.batches.append(len(batch))
            for t in batch:
                t.event.set()

    def mean_batch(self) -> float:
        return sum(self.batches) / len(self.batches) if self.batches else 0.0
