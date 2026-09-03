import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from backend.app.ai.bbox import MergedBox
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.ai.patching import PatchBox


CORE_LOGIC_VERSION = "dense_anomaly_v2"


@dataclass(frozen=True)
class DenseAnomalyResult:
    patch_scores: np.ndarray
    image_score: float
    localized_score: float
    anomaly_map: np.ndarray
    boxes: tuple[MergedBox, ...]


class DenseReferenceDetector:
    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index_path = _artifact_path(metadata_path, self.metadata["faiss_index_path"])
        self.store = FaissIndexStore(metric="cosine")
        self.index = self.store.load(index_path)

    def score(
        self,
        embeddings: np.ndarray,
        boxes: tuple[PatchBox, ...],
        *,
        grid_rows: int,
        grid_cols: int,
        image_width: int,
        image_height: int,
        thresholds: dict | None = None,
    ) -> DenseAnomalyResult:
        if embeddings.shape[0] != len(boxes):
            raise ValueError("Dense embedding count must match spatial boxes")
        if embeddings.shape[1] != int(self.index.d):
            raise ValueError(
                f"Reference expects {int(self.index.d)} features, query produced {embeddings.shape[1]}"
            )
        nearest = self.store.search(self.index, embeddings, k=1).scores[:, 0]
        patch_scores = np.maximum(0.0, 1.0 - nearest).astype(np.float32)
        effective = thresholds or self.metadata.get("thresholds") or {}
        method = str(effective.get("image_score_method", "top_k_mean"))
        top_fraction = float(effective.get("image_score_top_fraction", 0.01))
        image_score = aggregate_image_score(patch_scores, method=method, top_fraction=top_fraction)
        localized_score = localized_anomaly_score(patch_scores)
        anomaly_map = patch_scores.reshape(grid_rows, grid_cols)
        pixel_threshold = resolve_pixel_threshold(anomaly_map, effective)
        localized = localize_anomaly_map(
            anomaly_map,
            image_width=image_width,
            image_height=image_height,
            pixel_threshold=pixel_threshold,
            minimum_defect_area=int(effective.get("minimum_defect_area", 1)),
        )
        return DenseAnomalyResult(
            patch_scores=patch_scores,
            image_score=image_score,
            localized_score=localized_score,
            anomaly_map=anomaly_map,
            boxes=tuple(localized),
        )


def aggregate_image_score(
    scores: np.ndarray,
    *,
    method: str,
    top_fraction: float = 0.01,
) -> float:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    if method == "max":
        return float(values.max())
    if method == "percentile_995":
        return float(np.quantile(values, 0.995))
    baseline = float(np.median(values))
    if method == "max_median_contrast":
        return max(0.0, float(values.max()) - baseline)
    if method == "percentile_995_median_contrast":
        return max(0.0, float(np.quantile(values, 0.995)) - baseline)
    if method not in {"top_k_mean", "top_k_median_contrast"}:
        raise ValueError(f"Unsupported image score method: {method}")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    count = max(1, int(np.ceil(values.size * top_fraction)))
    indexes = np.argpartition(values, -count)[-count:]
    score = float(values[indexes].mean())
    if method == "top_k_median_contrast":
        score -= baseline
    return max(0.0, score)


def localized_anomaly_score(scores: np.ndarray) -> float:
    """Measure the strongest local deviation after removing an image-wide shift."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return max(0.0, float(values.max()) - float(np.median(values)))


def resolve_pixel_threshold(anomaly_map: np.ndarray, thresholds: dict) -> float:
    mode = str(thresholds.get("pixel_threshold_mode", "absolute"))
    default_floor = np.inf if mode == "absolute" else 0.0
    floor = float(thresholds.get("pixel_threshold", default_floor))
    if mode == "absolute":
        return floor
    if mode != "median_mad":
        raise ValueError(f"Unsupported pixel threshold mode: {mode}")
    values = np.asarray(anomaly_map, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return float("inf")
    multiplier = float(thresholds.get("pixel_mad_multiplier", 6.0))
    if multiplier <= 0:
        raise ValueError("pixel_mad_multiplier must be positive")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = max(1e-6, 1.4826 * mad)
    return max(floor, median + multiplier * robust_sigma)


def localize_anomaly_map(
    anomaly_map: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    pixel_threshold: float,
    minimum_defect_area: int,
) -> list[MergedBox]:
    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(pixel_threshold):
        return []
    grid_mask = (values >= pixel_threshold).astype(np.uint8)
    if not grid_mask.any():
        return []
    # Closing bridges adjacent dense cells but deliberately avoids opening,
    # which would erase one-cell-wide scratches and pressure lines.
    grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    pixel_mask = cv2.resize(grid_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pixel_mask, connectivity=8)
    boxes: list[MergedBox] = []
    minimum = max(1, int(minimum_defect_area))
    score_map = cv2.resize(values, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    for component_id in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component_id])
        if area < minimum:
            continue
        component_score = float(score_map[labels == component_id].max())
        boxes.append(
            MergedBox(
                x=x,
                y=y,
                w=width,
                h=height,
                score=component_score,
                patch_count=max(1, int((values >= pixel_threshold).sum())),
                label="dense_anomaly",
            )
        )
    return boxes


def _artifact_path(metadata_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (metadata_path.parent / path).resolve()
