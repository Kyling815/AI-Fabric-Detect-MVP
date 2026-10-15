from pathlib import Path
import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from backend.app.core.segmentation_shadow import SupervisedShadowConfig
from backend.app.services import segmentation_shadow
from backend.app.services.segmentation_shadow import ProcessSupervisedShadowRunner
from tests.segmentation_shadow_fixtures import observed_fixture, shadow_checkpoint, shadow_config


class FakeProcess:
    def __init__(self, output="", code=0, timeout=False, cleanup_fails=False):
        self.output, self.returncode, self.timeout, self.cleanup_fails = output, code, timeout, cleanup_fails
        self.killed = False
    def communicate(self, **kwargs):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired("fixture", 0.1)
        if self.cleanup_fails and self.killed:
            raise RuntimeError("cleanup fixture")
        return self.output, "private stderr never logged"
    def poll(self):
        return None if self.timeout and not self.killed else self.returncode
    def kill(self):
        if self.cleanup_fails:
            raise RuntimeError("fixture kill error")
        self.killed, self.returncode = True, -9


@pytest.fixture(autouse=True)
def fresh_capacity(monkeypatch):
    monkeypatch.setattr(segmentation_shadow, "_WORKER_SLOT", threading.BoundedSemaphore(1))


@pytest.mark.parametrize("failure,reason", [("empty", "checkpoint_missing"), ("untrusted", "checkpoint_untrusted"), ("fabric", "fabric_not_allowed")])
def test_safe_early_return_never_spawns(failure, reason):
    config = SupervisedShadowConfig() if failure == "empty" else shadow_config(trusted_local=failure != "untrusted")
    with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
        result = ProcessSupervisedShadowRunner().observe(Path("unused.png"), "other" if failure == "fabric" else "fixture_fabric", config)
    assert result.status == "unavailable" and result.reason == reason


def test_capacity_busy_returns_diagnostic_without_queue_or_spawn():
    segmentation_shadow._WORKER_SLOT.acquire()
    try:
        with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
            value = ProcessSupervisedShadowRunner().observe(Path("unused.png"), "fixture_fabric", shadow_config())
        assert value.status == "busy" and value.reason == "worker_busy"
    finally:
        segmentation_shadow._WORKER_SLOT.release()


def test_worker_success_validated_and_launched_without_shell_or_visible_window():
    config = shadow_config()
    process = FakeProcess(output=observed_fixture(config).model_dump_json())
    with patch("subprocess.Popen", return_value=process) as factory:
        value = ProcessSupervisedShadowRunner().observe(Path("image.png"), "fixture_fabric", config)
    assert value.status == "observed" and value.model_version == config.expected_model_version
    kwargs = factory.call_args.kwargs
    assert "shell" not in kwargs and kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert factory.call_args.args[0][-1] == "backend.app.ai.segmentation_shadow_worker"
    assert segmentation_shadow._WORKER_SLOT.acquire(blocking=False)
    segmentation_shadow._WORKER_SLOT.release()


@pytest.mark.parametrize("failure", ["exit", "invalid_json", "oversized", "wrong_version", "wrong_hash", "spawn_exception"])
def test_worker_failures_do_not_escape_and_release_capacity(failure):
    config, output, code = shadow_config(), "", 0
    if failure == "exit":
        code = 1
    elif failure == "invalid_json":
        output = "not json"
    elif failure == "oversized":
        output = "x" * (256 * 1024 + 1)
    else:
        changed = shadow_config(expected_model_version="other_v1") if failure == "wrong_version" else shadow_config(checkpoint_sha256="a" * 64)
        output = observed_fixture(changed).model_dump_json()
    with patch("subprocess.Popen", side_effect=OSError("fixture") if failure == "spawn_exception" else None,
               return_value=FakeProcess(output=output, code=code)):
        value = ProcessSupervisedShadowRunner().observe(Path("unused.png"), "fixture_fabric", config)
    assert value.status == "error" and not value.classes
    assert segmentation_shadow._WORKER_SLOT.acquire(blocking=False)
    segmentation_shadow._WORKER_SLOT.release()


def test_timeout_kills_and_reaps_worker_before_releasing_capacity():
    process = FakeProcess(timeout=True)
    with patch("subprocess.Popen", return_value=process):
        value = ProcessSupervisedShadowRunner().observe(Path("unused.png"), "fixture_fabric", shadow_config())
    assert value.status == "timeout" and process.killed and process.poll() is not None
    assert segmentation_shadow._WORKER_SLOT.acquire(blocking=False)
    segmentation_shadow._WORKER_SLOT.release()


def test_cleanup_failure_quarantines_worker_capacity_instead_of_spawning_more():
    process = FakeProcess(timeout=True, cleanup_fails=True)
    with patch("subprocess.Popen", return_value=process) as factory:
        runner = ProcessSupervisedShadowRunner()
        failed = runner.observe(Path("unused.png"), "fixture_fabric", shadow_config())
        next_value = runner.observe(Path("unused.png"), "fixture_fabric", shadow_config())
    assert failed.status == "error" and failed.reason == "worker_cleanup_failed"
    assert next_value.status == "busy" and factory.call_count == 1


def test_real_process_observation_and_real_deadline_cleanup(shadow_checkpoint):
    image, config = shadow_checkpoint
    runner = ProcessSupervisedShadowRunner()
    observed = runner.observe(image, "fixture_fabric", config)
    assert observed.status == "observed", observed
    created = []
    popen = subprocess.Popen
    def track(*args, **kwargs):
        process = popen(*args, **kwargs)
        created.append(process)
        return process
    start = time.monotonic()
    with patch("subprocess.Popen", side_effect=track):
        timed_out = runner.observe(image, "fixture_fabric", config.model_copy(update={"timeout_seconds": 0.05}))
    assert timed_out.status == "timeout" and time.monotonic() - start < 3
    assert len(created) == 1 and created[0].poll() is not None
    assert segmentation_shadow._WORKER_SLOT.acquire(blocking=False)
    segmentation_shadow._WORKER_SLOT.release()
