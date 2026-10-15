from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from backend.app.core.runtime_profiling import RuntimeProfiler, STAGES, current_profiler, runtime_stage


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def advance(self, milliseconds):
        self.now += milliseconds * 1_000_000


def test_exclusive_nested_stages_and_separate_queue_time():
    clock = Clock()
    profiler = RuntimeProfiler(clock=clock, queue_wait_ms=7)
    with profiler:
        with runtime_stage("anomaly"):
            clock.advance(2)
            with runtime_stage("preprocessing"):
                clock.advance(3)
            with runtime_stage("anomaly"):
                clock.advance(4)
            with runtime_stage("postprocessing"):
                clock.advance(5)
        clock.advance(1)
    value = profiler.profile.to_dict()
    assert value["anomaly_ms"] == 6
    assert value["preprocessing_ms"] == 3
    assert value["postprocessing_ms"] == 5
    assert value["total_ms"] == 15
    assert value["uninstrumented_ms"] == 1
    assert value["queue_wait_ms"] == 7
    assert value["segmentation_ms"] == 0
    assert value["stage_calls"]["segmentation"] == 0
    assert value["stage_calls"]["anomaly"] == 2
    assert value["succeeded"]
    assert current_profiler() is None


@pytest.mark.parametrize("name", STAGES)
def test_error_duration_retained_and_context_reset(name):
    clock = Clock()
    profiler = RuntimeProfiler(clock=clock)
    with pytest.raises(RuntimeError, match="original"):
        with profiler:
            with runtime_stage(name):
                clock.advance(9)
                raise RuntimeError("original")
    assert profiler.profile.to_dict()[name + "_ms"] == 9
    assert profiler.profile.to_dict()["stage_errors"][name] == 1
    assert not profiler.profile.succeeded
    assert current_profiler() is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_reject_invalid_queue_wait(value):
    with pytest.raises(ValueError):
        RuntimeProfiler(queue_wait_ms=value)


def test_disabled_spans_are_noop_without_cuda_or_timer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: pytest.fail("Unexpected CUDA sync"))
    with runtime_stage("anomaly"):
        assert current_profiler() is None


@pytest.mark.parametrize("requested", ["cpu", "cuda", "auto"])
def test_cpu_fallback_never_synchronizes_cuda(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: pytest.fail("Unexpected CUDA sync"))
    with RuntimeProfiler(device=requested) as profiler:
        with runtime_stage("anomaly"):
            pass
    assert profiler.profile.device == "cpu"
    assert profiler.profile.timing_mode == "host_wall_cpu"


def test_unavailable_indexed_cuda_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        RuntimeProfiler(device="cuda:0")


@pytest.mark.parametrize("device", ["mps", "meta"])
def test_unsynchronized_accelerators_rejected(device):
    with pytest.raises(ValueError, match="Only CPU"):
        RuntimeProfiler(device=device)


def test_cuda_fences_are_explicit_mock_only(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    with RuntimeProfiler(device="cuda:1") as profiler:
        with runtime_stage("segmentation"):
            pass
    assert calls == ["cuda:1"] * 4
    assert profiler.profile.timing_mode == "cuda_synchronized_device_wall"


def test_profiler_is_single_use_and_unknown_stage_rejected():
    profiler = RuntimeProfiler()
    with profiler:
        with pytest.raises(ValueError):
            with runtime_stage("not_a_stage"):
                pass
    with pytest.raises(RuntimeError, match="single use"):
        with profiler:
            pass


def test_nested_sessions_restore_parent():
    with RuntimeProfiler() as outer:
        with RuntimeProfiler() as inner:
            assert current_profiler() is inner
        assert current_profiler() is outer
    assert current_profiler() is None


def test_thread_local_profiles_do_not_cross_contaminate():
    barrier = threading.Barrier(2)
    def run(name):
        with RuntimeProfiler() as profiler:
            with runtime_stage(name):
                barrier.wait(timeout=5)
        return profiler.profile.to_dict()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, name) for name in ("segmentation", "anomaly")]
        values = [future.result(timeout=10) for future in futures]
    assert values[0]["stage_calls"]["anomaly"] == 0
    assert values[1]["stage_calls"]["segmentation"] == 0
