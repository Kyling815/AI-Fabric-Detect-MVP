"""Bounded, process-isolated shadow runner. No shared encoder/model/global Torch state."""

from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Protocol

from backend.app.core.paths import resolve_project_path
from backend.app.core.segmentation_shadow import SupervisedShadowConfig, SupervisedShadowObservation


_WORKER_SLOT = threading.BoundedSemaphore(1)  # Per server process; no queue or silent workload buildup.


def _kill_and_reap(process) -> bool:
    try:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=1.0)
        return process.poll() is not None
    except Exception:
        return False  # Quarantine capacity if OS cleanup cannot confirm the child exited.


class SupervisedShadowRunner(Protocol):
    def observe(self, image_path: Path, fabric_id: str, config: SupervisedShadowConfig) -> SupervisedShadowObservation: ...


class ProcessSupervisedShadowRunner:
    def observe(self, image_path: Path, fabric_id: str, config: SupervisedShadowConfig) -> SupervisedShadowObservation:
        start = time.monotonic()
        def failure(status, reason):
            return SupervisedShadowObservation(status=status, reason=reason, requested_model_version=config.expected_model_version,
                                                elapsed_ms=(time.monotonic() - start) * 1000)
        if not config.checkpoint_path:
            return failure("unavailable", "checkpoint_missing")
        if not config.trusted_local:
            return failure("unavailable", "checkpoint_untrusted")
        if fabric_id not in config.allowed_fabric_ids:
            return failure("unavailable", "fabric_not_allowed")
        if not _WORKER_SLOT.acquire(blocking=False):
            return failure("busy", "worker_busy")
        process = None
        quarantined = False
        try:
            import json
            command = [sys.executable, "-m", "backend.app.ai.segmentation_shadow_worker"]
            root = Path(__file__).resolve().parents[3]
            worker_config = config.model_copy(update={"checkpoint_path": str(resolve_project_path(config.checkpoint_path).resolve())})
            request = json.dumps({"image_path": str(image_path.resolve()), "fabric_id": fabric_id,
                                  "config": worker_config.model_dump(mode="json")})
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, encoding="utf-8", cwd=root,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            remaining = config.timeout_seconds - (time.monotonic() - start)
            if remaining <= 0:
                quarantined = not _kill_and_reap(process)
                return failure("error" if quarantined else "timeout", "worker_cleanup_failed" if quarantined else "worker_timeout")
            try:
                stdout, _ = process.communicate(input=request, timeout=remaining)
            except subprocess.TimeoutExpired:
                quarantined = not _kill_and_reap(process)
                return failure("error" if quarantined else "timeout", "worker_cleanup_failed" if quarantined else "worker_timeout")
            if process.returncode != 0 or len(stdout) > 256 * 1024:
                return failure("error", "worker_failed")
            observation = SupervisedShadowObservation.model_validate_json(stdout)
            if observation.requested_model_version != config.expected_model_version:
                return failure("error", "worker_contract_error")
            if observation.status == "observed" and (observation.checkpoint_sha256 != config.checkpoint_sha256
                    or (observation.synthetic_checkpoint and not config.allow_synthetic)
                    or (observation.mask_rle is not None and len(observation.mask_rle) > config.max_mask_runs)):
                return failure("error", "worker_contract_error")
            return observation.model_copy(update={"elapsed_ms": (time.monotonic() - start) * 1000})
        except Exception:
            return failure("error", "runner_error")
        finally:
            if process is not None and not quarantined:
                try:
                    if process.poll() is None:
                        quarantined = not _kill_and_reap(process)
                except Exception:
                    quarantined = True
            if not quarantined:
                _WORKER_SLOT.release()
