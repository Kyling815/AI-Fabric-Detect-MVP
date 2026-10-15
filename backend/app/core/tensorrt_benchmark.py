"""Offline Phase 17 contracts. No GPU imports, activation or acceptance SLA."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ("logits", "probabilities", "mask", "confidence")
PROBES = ("black", "white", "uniform", "checker", "scratch", "color_channels", "seeded_noise")
# Fixed BEFORE measurement. Mask equality is never relaxed, including FP16.
TOLERANCES = {"FP32": (1e-4, 1e-4), "FP16_MIXED": (1e-2, 1e-2)}


@dataclass(frozen=True)
class BenchmarkConfig:
    shape: tuple[int, int, int, int] = (1, 3, 256, 256)
    warmup: int = 10
    iterations: int = 100
    repeats: int = 3
    workspace_mib: int = 128
    builder_level: int = 2

    def __post_init__(self):
        if (not isinstance(self.shape, (tuple, list)) or len(self.shape) != 4
                or any(type(v) is not int for v in self.shape)
                or self.shape[0] != 1 or self.shape[1] != 3
                or any(not 32 <= v <= 512 for v in self.shape[2:])):
            raise ValueError("This single-image workload requires fixed NCHW batch1 RGB, H/W 32..512")
        for name, low, high in (("warmup", 1, 100), ("iterations", 10, 1000),
                                ("repeats", 1, 10), ("workspace_mib", 16, 256), ("builder_level", 0, 3)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid bounded benchmark setting: " + name)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, digest, limit=512 * 1024 * 1024):
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Explicit lowercase SHA-256 pin required")
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit or sha256(data) != digest:
        raise ValueError("Artifact size or SHA-256 mismatch")
    return data


def new_candidate_directory(path, root=ROOT):
    """Only a NEW version under segmentation_candidates, never arbitrary paths."""
    path = Path(path).resolve()
    base = (Path(root) / "storage/models/segmentation_candidates").resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("Use a NEW directory strictly inside storage/models/segmentation_candidates")
    if path.exists():
        raise FileExistsError("Candidate/output version exists; never overwrite")
    return path


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def summarize_latencies(samples, batch_size=1):
    if (type(batch_size) is not int or batch_size < 1 or not samples
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v <= 0 for v in samples)):
        raise ValueError("Finite positive measured millisecond samples required")
    return {"samples": len(samples), "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "mean_ms": float(np.mean(samples)),
            "throughput_images_per_second": len(samples) * batch_size * 1000 / sum(samples),
            "latency_ms": list(samples), "percentile_method": "numpy linear"}


def validate_images(images, shape):
    if (not isinstance(images, np.ndarray) or images.dtype != np.float32
            or images.shape != tuple(shape) or not np.isfinite(images).all()
            or images.min() < 0 or images.max() > 1):
        raise ValueError("Exact fixed-shape finite float32 RGB NCHW [0,1] required")
    return np.ascontiguousarray(images)


def prediction_contract(values, shape, classes):
    batch, _, height, width = shape
    shapes = ((batch, classes, height, width),) * 2 + ((batch, height, width),) * 2
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError("Four segmentation outputs required")
    for name, value, expected in zip(OUTPUTS, values, shapes):
        if (not isinstance(value, np.ndarray) or value.shape != expected
                or value.dtype != (np.int64 if name == "mask" else np.float32)
                or not np.isfinite(value).all()):
            raise ValueError("Prediction shape/dtype/finiteness violation: " + name)
    _, probabilities, mask, confidence = values
    checks = {"probability_range": bool(probabilities.min() >= 0 and probabilities.max() <= 1),
        "probability_sum": bool(np.allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)),
        "mask_matches_probabilities": bool(np.array_equal(mask, probabilities.argmax(1))),
        "confidence_matches_probabilities": bool(np.allclose(confidence, probabilities.max(1), atol=1e-6, rtol=0))}
    return {"passed": all(checks.values()), "checks": checks,
            "max_probability_sum_error": float(np.abs(probabilities.sum(1) - 1).max())}


def validate_predictions(values, shape, classes):
    if not prediction_contract(values, shape, classes)["passed"]:
        raise ValueError("Segmentation probability/mask/confidence contract violation")


def compare_predictions(expected, actual, shape, classes, precision):
    if precision not in TOLERANCES:
        raise ValueError("Unsupported precision")
    validate_predictions(expected, shape, classes)
    # Valid finite tensors with semantic drift are REJECTED and recorded, not
    # discarded before diagnostic measurements. Malformed/nonfinite outputs
    # still raise. No contract/tolerance is relaxed for FP16.
    actual_contract = prediction_contract(actual, shape, classes)
    atol, rtol = TOLERANCES[precision]
    metrics = {}
    for name, left, right in zip(OUTPUTS, expected, actual):
        if name == "mask":
            mismatches = int(np.count_nonzero(left != right))
            metrics[name] = {"passed": mismatches == 0, "mismatched_pixels": mismatches,
                             "pixel_count": left.size, "requires_exact_equality": True}
        else:
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            metrics[name] = {"passed": bool(np.allclose(right, left, atol=atol, rtol=rtol)),
                             "max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean())}
    return {"passed": actual_contract["passed"] and all(v["passed"] for v in metrics.values()),
            "precision": precision, "atol": atol, "rtol": rtol, "outputs": metrics,
            "output_contract": actual_contract}


def choose_configuration(results):
    """A fast but numerically rejected engine can never be recommended."""
    eligible = [item for item in results if item.get("parity_passed") is True
                and item.get("precision_verified") is True and item.get("memory_headroom_passed") is True]
    return min(eligible, key=lambda item: item["host_roundtrip"]["p95_ms"])["precision"] if eligible else None
