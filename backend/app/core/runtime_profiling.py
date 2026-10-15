"""Opt-in, exclusive wall-clock stage accounting; not a GPU kernel profiler.

total_ms covers only the profiled callable. queue_wait_ms is separate (admission
to resource acquisition); upload, annotation, persistence and HTTP are excluded.
Nested spans are subtracted from parents, so stages are not double-counted.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import math
from time import perf_counter_ns


STAGES = ("preprocessing", "segmentation", "anomaly", "fusion", "postprocessing")
_CURRENT = ContextVar("fabric_runtime_profiler", default=None)


@dataclass(frozen=True)
class RuntimeProfile:
    preprocessing_ms: float
    segmentation_ms: float
    anomaly_ms: float
    fusion_ms: float
    postprocessing_ms: float
    total_ms: float
    queue_wait_ms: float
    uninstrumented_ms: float
    stage_calls: tuple[tuple[str, int], ...]
    stage_errors: tuple[tuple[str, int], ...]
    device: str
    timing_mode: str
    succeeded: bool

    def to_dict(self):
        result = asdict(self)
        result["stage_calls"] = dict(self.stage_calls)
        result["stage_errors"] = dict(self.stage_errors)
        return result


class RuntimeProfiler:
    """One task/thread only. CUDA timing explicitly fences the selected device.

    Fences perturb execution and include device contention, not just kernel work.
    The profiler never synchronizes CUDA when the resolved device is CPU.
    A zero duration with zero calls means NOT EXECUTED, not a free branch.
    """

    def __init__(self, *, device="cpu", queue_wait_ms=0.0, clock=perf_counter_ns):
        if isinstance(queue_wait_ms, bool) or not math.isfinite(queue_wait_ms) or queue_wait_ms < 0:
            raise ValueError("Finite nonnegative queue wait required")
        import torch
        requested = str(device).lower()
        if requested == "auto" or requested == "cuda":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU and explicitly synchronized CUDA timing are supported")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device unavailable")
        self.clock, self.queue_wait_ms = clock, float(queue_wait_ms)
        self.durations = dict.fromkeys(STAGES, 0)
        self.calls = dict.fromkeys(STAGES, 0)
        self.errors = dict.fromkeys(STAGES, 0)
        self.stack = []
        self.profile = None
        self._used = False

    def _sync(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        if self._used:
            raise RuntimeError("Profiler is single use")
        self._used = True
        self._sync()
        self.started = self.clock()
        self.token = _CURRENT.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._sync()
            total = self.clock() - self.started
            self.profile = RuntimeProfile(
                **{name + "_ms": self.durations[name] / 1e6 for name in STAGES},
                total_ms=total / 1e6, queue_wait_ms=self.queue_wait_ms,
                uninstrumented_ms=max(0, total - sum(self.durations.values())) / 1e6,
                stage_calls=tuple(self.calls.items()), stage_errors=tuple(self.errors.items()),
                device=str(self.device),
                timing_mode="cuda_synchronized_device_wall" if self.device.type == "cuda" else "host_wall_cpu",
                succeeded=exc_type is None,
            )
        finally:
            _CURRENT.reset(self.token)

    @contextmanager
    def span(self, name):
        if name not in STAGES:
            raise ValueError("Unknown runtime stage: " + str(name))
        self._sync()
        frame = [self.clock(), 0]
        self.stack.append(frame)
        self.calls[name] += 1
        try:
            yield
        except BaseException:
            self.errors[name] += 1
            raise
        finally:
            try:
                self._sync()
            finally:
                elapsed = self.clock() - frame[0]
                self.stack.pop()
                self.durations[name] += max(0, elapsed - frame[1])
                if self.stack:
                    self.stack[-1][1] += elapsed


def current_profiler():
    return _CURRENT.get()


@contextmanager
def runtime_stage(name):
    profiler = current_profiler()
    if profiler is None:
        yield
    else:
        with profiler.span(name):
            yield
