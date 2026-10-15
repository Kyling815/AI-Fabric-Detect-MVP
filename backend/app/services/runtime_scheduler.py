"""Explicit offline/task scheduler. Never shares a DB Session or schedules HTTP.

Bounded FIFO pending queue, nonblocking admission, graceful drain. The default
process-local gate serializes complete jobs even across scheduler instances;
parallel execution is an explicit benchmark/advanced caller opt-in, not a claim
that two GPU models finish in max(T1,T2). No native inference can be preempted.
Tasks must not wait on other jobs using this gate (nested-job deadlock). Callers
own all tensor/model/context lifetimes until completion; no implicit copying.
"""

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
import threading
from time import perf_counter_ns
from typing import Any, Callable

from backend.app.core.runtime_profiling import RuntimeProfile, RuntimeProfiler


_RESOURCE_GATE = threading.Lock()


class SchedulerBusy(RuntimeError):
    """Admission rejected BEFORE ownership transfer. Caller must retry or stop."""


@dataclass(frozen=True)
class ScheduledResult:
    value: Any
    profile: RuntimeProfile | None
    error_type: str | None = None

    @property
    def succeeded(self):
        return self.error_type is None


class RuntimeScheduler:
    def __init__(self, *, workers=1, max_pending=4, device="cpu", allow_concurrent_jobs=False):
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError("workers must be an integer in [1,8]")
        if type(max_pending) is not int or not 1 <= max_pending <= 1024:
            raise ValueError("max_pending must be an integer in [1,1024]")
        if type(allow_concurrent_jobs) is not bool:
            raise ValueError("Explicit concurrency boolean required")
        # Validate device before starting workers; no silent background setup failure.
        self.device = str(RuntimeProfiler(device=device).device)
        self.max_pending = max_pending
        self.allow_concurrent_jobs = allow_concurrent_jobs
        self._condition = threading.Condition()
        self._queue = deque()
        self._closed = False
        self._running = self._accepted = self._completed = self._failed = self._cancelled = self._rejected = 0
        self._threads = [threading.Thread(target=self._worker, name=f"fabric-runtime-{i}", daemon=True) for i in range(workers)]
        for thread in self._threads:
            thread.start()

    def submit(self, task: Callable[[], Any]) -> Future:
        if not callable(task):
            raise TypeError("A caller-owned zero-argument task is required")
        with self._condition:
            if self._closed:
                raise RuntimeError("Scheduler is closed")
            if len(self._queue) >= self.max_pending:
                self._rejected += 1
                raise SchedulerBusy("Pending queue full; no task accepted or dropped")
            future = Future()
            self._queue.append((future, task, perf_counter_ns()))
            self._accepted += 1
            self._condition.notify()
            return future

    def snapshot(self):
        with self._condition:
            return {"accepted": self._accepted, "completed": self._completed, "failed": self._failed,
                    "cancelled": self._cancelled, "rejected": self._rejected, "running": self._running,
                    "pending": len(self._queue), "closed": self._closed,
                    "max_pending": self.max_pending, "workers": len(self._threads),
                    "allow_concurrent_jobs": self.allow_concurrent_jobs}

    def _worker(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue:
                    return
                future, task, submitted = self._queue.popleft()
                self._running += 1
            result = None
            cancelled = False
            gate = nullcontext() if self.allow_concurrent_jobs else _RESOURCE_GATE
            with gate:
                cancelled = not future.set_running_or_notify_cancel()
                if not cancelled:
                    profiler = None
                    try:
                        profiler = RuntimeProfiler(device=self.device, queue_wait_ms=(perf_counter_ns() - submitted) / 1e6)
                        with profiler:
                            value = task()
                        result = ScheduledResult(value, profiler.profile)
                    except BaseException as exc:
                        # A failed branch/job cannot become a successful GOOD result.
                        # Keep workers alive and retain timing; do not expose exception payloads.
                        result = ScheduledResult(None, profiler.profile if profiler is not None else None, type(exc).__name__)
            with self._condition:
                self._running -= 1
                if cancelled:
                    self._cancelled += 1
                else:
                    self._completed += 1
                    self._failed += int(not result.succeeded)
            if not cancelled:
                future.set_result(result)

    def shutdown(self, *, wait=True):
        if wait and threading.current_thread() in self._threads:
            raise RuntimeError("A scheduler worker cannot join itself")
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if wait:
            for thread in self._threads:
                thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
