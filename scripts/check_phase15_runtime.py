"""Measured offline CPU benchmark; synthetic tiles, no accuracy/line-speed claim.

Print JSON only. Model construction is outside warm timings. Uses trusted local
cached ImageNet weights + registered adapter read-only; never downloads/trains.
The independent U-Net has an UNTRAINED decoder. Reference is a temporary toy
bank, not a production reference. No .env, registry, DB or storage writes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import threading
from time import perf_counter_ns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch

from backend.app.ai.dense_anomaly import DenseReferenceDetector
from backend.app.ai.dense_feature_extractor import SharedDenseResNet18
from backend.app.ai.faiss_index import FaissIndexStore, faiss
from backend.app.ai.fusion import FusionEngine
from backend.app.ai.segmentation_model import ResNet18UNet, cached_resnet18_weights
from backend.app.core.fusion import (AnomalyEvidence, CaptureEvidence, EvidenceIdentity, FusionPolicy,
                                    KnownClassEvidence, MaskDescriptor, SupervisedEvidence)
from backend.app.core.paths import resolve_project_path
from backend.app.core.runtime_profiling import RuntimeProfiler, STAGES, runtime_stage
from backend.app.core.segmentation_data import SegmentationTaxonomy
from backend.app.services.fabric_registry import FabricRegistry
from backend.app.services.runtime_scheduler import RuntimeScheduler, SchedulerBusy


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"count": int(values.size), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min()), "max": float(values.max())}


def benchmark(*, iterations=8, tiles=4, size=64, warmups=2):
    if not 2 <= iterations <= 100 or not 1 <= tiles <= 32 or not 32 <= size <= 256 or not 1 <= warmups <= 10:
        raise ValueError("Bounded benchmark parameters required")
    previous_threads, previous_faiss = torch.get_num_threads(), faiss.omp_get_max_threads()
    previous_rng = torch.random.get_rng_state()
    try:
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
        torch.manual_seed(15)
        with tempfile.TemporaryDirectory(prefix="fabric_phase15_") as tmp:
            started = perf_counter_ns()
            _, encoder_hash = cached_resnet18_weights()  # Verify cache before torchvision could download.
            backbone = FabricRegistry().shared_backbone
            adapter_path = resolve_project_path(backbone["adapter_path"])
            extractor = SharedDenseResNet18(backbone_version=backbone["backbone_version"], device="cpu",
                                           use_pretrained_weights=True, adapter_path=str(adapter_path))
            segmentation = ResNet18UNet(SegmentationTaxonomy(), pretrained=True).eval()
            rows, cols = np.indices((size, size))
            tile = np.repeat((((rows // 8 + cols // 8) % 2) * 120 + 60).astype(np.uint8)[..., None], 3, axis=2)
            image = Image.fromarray(tile)
            features = extractor.extract_image(image, image_max_size=size)
            index_path = Path(tmp) / "toy.faiss"
            FaissIndexStore().save(features.embeddings, index_path)
            # Metadata exists only in the OS temporary directory.
            metadata = Path(tmp) / "toy.json"
            metadata.write_text(json.dumps({"faiss_index_path": "toy.faiss"}), encoding="utf-8")
            detector = DenseReferenceDetector(metadata)
            setup_ms = (perf_counter_ns() - started) / 1e6
            fusion = FusionEngine(FusionPolicy())  # Synthetic evidence remains disallowed for acceptance.
            identity = EvidenceIdentity(fabric_id="synthetic_benchmark", capture_id="tile")

            def prepare():
                with runtime_stage("preprocessing"):
                    return image.copy(), torch.from_numpy(tile.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255

            def segment(tensor):
                with runtime_stage("segmentation"):
                    # forward() in eval mode has no predict() train/eval mutation, safe for read-only probes.
                    with torch.inference_mode():
                        probabilities = segmentation(tensor).softmax(1)
                        return probabilities.max(1)

            def anomaly(pil_image):
                with runtime_stage("anomaly"):
                    value = extractor.extract_image(pil_image, image_max_size=size)
                    return detector.score(value.embeddings, value.boxes, grid_rows=value.grid_rows,
                        grid_cols=value.grid_cols, image_width=size, image_height=size,
                        thresholds={"pixel_threshold": 0.2, "minimum_defect_area": 1})

            def pipeline():
                pil_image, tensor = prepare()
                confidence, mask = segment(tensor)
                result = anomaly(pil_image)
                with runtime_stage("fusion"):
                    mask_array = mask[0].cpu().numpy().astype(np.uint8)
                    selected = mask[0] == 1
                    count = int(selected.sum())
                    candidates = () if count == 0 else (KnownClassEvidence(class_name="defect", class_id=1,
                        confidence=float(confidence[0][selected].mean()), predicted_pixels=count),)
                    supervised = SupervisedEvidence(status="available", identity=identity,
                        model_version="UNTRAINED_decoder_runtime_probe", checkpoint_sha256=state_hash,
                        synthetic=True, taxonomy=("background", "defect"), candidates=candidates,
                        mask=MaskDescriptor(sha256=hashlib.sha256(mask_array.tobytes()).hexdigest(), shape=(size, size),
                                            original_shape=(size, size), payload_available=True))
                    dense = AnomalyEvidence(status="available", identity=identity, reference_version="temporary_toy",
                        backbone_version=extractor.backbone_version, score=float(result.image_score),
                        localized_score=float(result.localized_score), image_shape=(size, size), score_method="top_k_mean")
                    fused = fusion.fuse(supervised, dense, capture=CaptureEvidence(status="valid", source="caller_declared"))
                with runtime_stage("postprocessing"):
                    return {"fusion_kind": fused.kind, "mask_sha256": supervised.mask.sha256,
                            "anomaly_score": float(result.image_score), "box_count": len(result.boxes)}

            # Hash initialized state (not a claimed trained checkpoint).
            state_digest = hashlib.sha256()
            for name, value in sorted(segmentation.state_dict().items()):
                state_digest.update(name.encode())
                state_digest.update(value.cpu().numpy().tobytes())
            state_hash = state_digest.hexdigest()
            cold = RuntimeProfiler()
            with cold:
                baseline_output = pipeline()
            for _ in range(warmups):
                assert pipeline() == baseline_output
            profiles = []
            for _ in range(iterations):
                profiler = RuntimeProfiler()
                with profiler:
                    assert pipeline() == baseline_output
                profiles.append(profiler.profile.to_dict())

            # Same two actual CNN workloads, measured separately, sequentially,
            # and concurrently on this CPU. Never derive combined time from max().
            pil_image, tensor = prepare()
            contention = {name: [] for name in ("segmentation_alone_ms", "anomaly_alone_ms", "sequential_pair_ms", "concurrent_pair_ms")}
            with ThreadPoolExecutor(max_workers=2) as pool:
                for _ in range(iterations):
                    for name, call in (("segmentation_alone_ms", lambda: segment(tensor)),
                                       ("anomaly_alone_ms", lambda: anomaly(pil_image)),
                                       ("sequential_pair_ms", lambda: (segment(tensor), anomaly(pil_image)))):
                        tick = perf_counter_ns()
                        call()
                        contention[name].append((perf_counter_ns() - tick) / 1e6)
                    barrier = threading.Barrier(2)
                    def probe(call):
                        barrier.wait(timeout=10)
                        return call()
                    tick = perf_counter_ns()
                    futures = [pool.submit(probe, lambda: segment(tensor)), pool.submit(probe, lambda: anomaly(pil_image))]
                    for future in futures:
                        future.result(timeout=60)
                    contention["concurrent_pair_ms"].append((perf_counter_ns() - tick) / 1e6)

            scheduled = []
            batch_started = perf_counter_ns()
            with RuntimeScheduler(workers=2, max_pending=tiles) as scheduler:
                futures = [scheduler.submit(pipeline) for _ in range(tiles)]
                for future in futures:
                    result = future.result(timeout=60)
                    assert result.succeeded and result.value == baseline_output
                    scheduled.append(result.profile.to_dict())
            batch_ms = (perf_counter_ns() - batch_started) / 1e6
            snapshot = scheduler.snapshot()

            # Deterministic queue-pressure/failure scenario, real scheduler timings.
            entered, release = threading.Event(), threading.Event()
            with RuntimeScheduler(workers=1, max_pending=1) as pressure:
                def blocked():
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError("Probe not released")
                    return pipeline()
                first = pressure.submit(blocked)
                try:
                    assert entered.wait(10)
                    def failed_branch():
                        with runtime_stage("segmentation"):
                            raise RuntimeError("Injected branch failure")
                    second = pressure.submit(failed_branch)
                    try:
                        pressure.submit(pipeline)
                    except SchedulerBusy:
                        rejected = True
                    else:
                        raise AssertionError("Queue pressure did not reject admission")
                finally:
                    release.set()
                first_result, failure_result = first.result(timeout=60), second.result(timeout=60)
                assert first_result.succeeded and not failure_result.succeeded
            metrics = [name + "_ms" for name in STAGES] + ["total_ms", "queue_wait_ms", "uninstrumented_ms"]
            return {"phase": 15, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "OFFLINE_CPU_RUNTIME_ONLY_SYNTHETIC_INPUT_UNTRAINED_SEGMENTATION_TOY_REFERENCE",
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor(), "torch": torch.__version__, "torch_threads": 1,
                    "faiss_threads": 1, "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda},
                "parameters": {"batch_size": 1, "tile_shape_rgb": [size, size, 3], "warmups": warmups,
                    "iterations": iterations, "multiple_tiles": tiles, "device": "cpu"},
                "models": {"anomaly_backbone": extractor.info.__dict__, "encoder_cached_sha256": encoder_hash,
                    "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                    "segmentation": "independent_resnet18_unet_v1; cached encoder, untrained standard decoder",
                    "segmentation_initialized_state_sha256": state_hash, "toy_reference_vectors": len(features.embeddings)},
                "setup_ms": setup_ms, "cold_profile": cold.profile.to_dict(),
                "warm_stage_ms": {name: summary([row[name] for row in profiles]) for name in metrics},
                "warm_profiles": profiles, "output_parity": "PASS",
                "cpu_contention": {"raw_samples": contention, "summary": {k: summary(v) for k, v in contention.items()},
                    "note": "Actual overlapping independent CNN probes; includes dispatch/barrier overhead, no GPU inference."},
                "multiple_tile_scheduler": {"snapshot": snapshot, "wall_ms_including_startup_shutdown": batch_ms,
                    "tiles_per_second": tiles * 1000 / batch_ms, "profiles": scheduled,
                    "policy": "2 workers, process-local gate serializes full jobs; queue/resource wait measured together"},
                "queue_pressure_and_failure": {"admission_rejected": rejected, "snapshot": pressure.snapshot(),
                    "failed_job_error_type": failure_result.error_type, "failed_profile": failure_result.profile.to_dict()},
                "gpu_contention": {"status": "NOT_MEASURED", "reason": "CPU benchmark only; no CUDA hardware tested"},
                "limitations": ["No real segmentation checkpoint or factory reference; no accuracy evaluation",
                    "64px default toy tiles and short ordered probes cannot predict line speed or production p95",
                    "In-process model timings exclude Phase9 subprocess startup/checkpoint validation and mask transport",
                    "Upload/annotation/DB/HTTP, camera I/O, PLC, VRAM and long-duration load not benchmarked"],
                "production_activation": False, "next_phase_started": False}
    finally:
        torch.random.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)
        faiss.omp_set_num_threads(previous_faiss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(benchmark(iterations=args.iterations, tiles=args.tiles, size=args.size, warmups=args.warmups), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
