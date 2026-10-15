from concurrent.futures import CancelledError
import threading

import pytest

from backend.app.core.runtime_profiling import current_profiler, runtime_stage
from backend.app.services.runtime_scheduler import RuntimeScheduler, SchedulerBusy


@pytest.mark.parametrize("values", [{"workers": 0}, {"workers": 9}, {"workers": True}, {"workers": 1.5},
    {"max_pending": 0}, {"max_pending": 1025}, {"max_pending": True}, {"max_pending": 1.5},
    {"allow_concurrent_jobs": 1}, {"device": "nonsense"}])
def test_invalid_scheduler_rejected_without_workers(values):
    with pytest.raises((ValueError, RuntimeError)):
        RuntimeScheduler(**values)


def assert_conserved(snapshot):
    assert snapshot["accepted"] == snapshot["completed"] + snapshot["cancelled"] + snapshot["running"] + snapshot["pending"]
    assert snapshot["pending"] <= snapshot["max_pending"]
    assert snapshot["running"] <= snapshot["workers"]
    assert snapshot["failed"] <= snapshot["completed"]


def test_queue_pressure_explicit_rejection_fifo_and_graceful_drain():
    entered, release = threading.Event(), threading.Event()
    order = []
    def blocked():
        entered.set()
        assert release.wait(5)
        order.append(0)
        return 0
    with RuntimeScheduler(max_pending=2) as scheduler:
        first = scheduler.submit(blocked)
        try:
            assert entered.wait(5)
            queued = [scheduler.submit(lambda i=i: order.append(i) or i) for i in (1, 2)]
            with pytest.raises(SchedulerBusy):
                scheduler.submit(lambda: pytest.fail("Rejected job ran"))
            snapshot = scheduler.snapshot()
            assert snapshot["accepted"] == 3 and snapshot["rejected"] == 1
            assert_conserved(snapshot)
            scheduler.shutdown(wait=False)
            with pytest.raises(RuntimeError, match="closed"):
                scheduler.submit(lambda: None)
        finally:
            release.set()
    assert order == [0, 1, 2]
    results = [future.result(timeout=5) for future in [first, *queued]]
    assert [result.value for result in results] == [0, 1, 2]
    assert all(result.profile.queue_wait_ms >= 0 for result in results)
    assert results[1].profile.queue_wait_ms > 0
    assert scheduler.snapshot()["completed"] == 3
    assert_conserved(scheduler.snapshot())


@pytest.mark.parametrize("stage", ["preprocessing", "segmentation", "anomaly", "fusion", "postprocessing"])
def test_failure_is_not_success_and_worker_continues(stage):
    def fail():
        with runtime_stage(stage):
            raise RuntimeError("Private error payload")
    with RuntimeScheduler(max_pending=3) as scheduler:
        failed = scheduler.submit(fail).result(timeout=5)
        good = scheduler.submit(lambda: 42).result(timeout=5)
    assert not failed.succeeded and failed.value is None and failed.error_type == "RuntimeError"
    assert not failed.profile.succeeded
    assert failed.profile.to_dict()["stage_errors"][stage] == 1
    assert good.succeeded and good.value == 42
    assert scheduler.snapshot()["failed"] == 1
    assert_conserved(scheduler.snapshot())


def test_cancelled_pending_task_is_accounted_not_executed():
    entered, release = threading.Event(), threading.Event()
    def blocked():
        entered.set()
        assert release.wait(5)
    with RuntimeScheduler(max_pending=1) as scheduler:
        first = scheduler.submit(blocked)
        try:
            assert entered.wait(5)
            second = scheduler.submit(lambda: pytest.fail("Cancelled task ran"))
            assert second.cancel()
        finally:
            release.set()
    first.result(timeout=5)
    with pytest.raises(CancelledError):
        second.result(timeout=5)
    assert scheduler.snapshot()["cancelled"] == 1
    assert_conserved(scheduler.snapshot())


def test_default_gate_serializes_models_across_scheduler_instances():
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    def blocked():
        entered.set()
        assert release.wait(5)
    with RuntimeScheduler(workers=2) as first_scheduler, RuntimeScheduler(workers=2) as second_scheduler:
        first = first_scheduler.submit(blocked)
        try:
            assert entered.wait(5)
            second = second_scheduler.submit(lambda: second_started.set())
            assert not second_started.wait(0.05)
        finally:
            release.set()
        first.result(timeout=5)
        assert second.result(timeout=5).succeeded
    assert second_started.is_set()


def test_explicit_concurrent_jobs_overlap_with_independent_profiles():
    barrier = threading.Barrier(2)
    def task(stage):
        assert current_profiler() is not None
        with runtime_stage(stage):
            barrier.wait(timeout=5)
        return stage
    with RuntimeScheduler(workers=2, max_pending=2, allow_concurrent_jobs=True) as scheduler:
        futures = [scheduler.submit(lambda name=name: task(name)) for name in ("anomaly", "segmentation")]
        results = [future.result(timeout=10) for future in futures]
    assert all(result.succeeded for result in results)
    assert results[0].profile.to_dict()["stage_calls"]["segmentation"] == 0
    assert results[1].profile.to_dict()["stage_calls"]["anomaly"] == 0
    assert_conserved(scheduler.snapshot())


def test_self_join_rejected_and_shutdown_idempotent():
    scheduler = RuntimeScheduler()
    try:
        failure = scheduler.submit(lambda: scheduler.shutdown()).result(timeout=5)
        assert failure.error_type == "RuntimeError"
        assert not scheduler.snapshot()["closed"]
        assert scheduler.submit(lambda: 1).result(timeout=5).value == 1
    finally:
        scheduler.shutdown()
        scheduler.shutdown()


def test_non_callable_task_rejected():
    with RuntimeScheduler() as scheduler:
        with pytest.raises(TypeError):
            scheduler.submit(1)
    assert scheduler.snapshot()["accepted"] == 0


def test_profiler_setup_failure_does_not_strand_future_or_worker(monkeypatch):
    from backend.app.services import runtime_scheduler
    original = runtime_scheduler.RuntimeProfiler
    scheduler = RuntimeScheduler()
    try:
        def fail(**kwargs):
            raise RuntimeError("Profiler setup failed")
        with monkeypatch.context() as patch:
            patch.setattr(runtime_scheduler, "RuntimeProfiler", fail)
            result = scheduler.submit(lambda: pytest.fail("Job must not start")).result(timeout=5)
            assert result.error_type == "RuntimeError" and result.profile is None
        assert scheduler.submit(lambda: 7).result(timeout=5).value == 7
    finally:
        scheduler.shutdown()
    assert_conserved(scheduler.snapshot())
