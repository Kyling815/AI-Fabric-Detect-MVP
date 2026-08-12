import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.ai.patching import PatchBox
from backend.app.core.paths import resolve_project_path


@dataclass(frozen=True)
class PatchAnomaly:
    patch_index: int
    reference_offset: int
    score: float
    anomaly_score: float
    suspicious: bool
    box: PatchBox


@dataclass(frozen=True)
class AnomalyDetectionResult:
    status: str
    reference_version: str | None
    faiss_metric: str | None
    threshold: float
    patch_scores: tuple[PatchAnomaly, ...]

    @property
    def suspicious_patch_count(self) -> int:
        return sum(1 for item in self.patch_scores if item.suspicious)

    @property
    def max_anomaly_score(self) -> float | None:
        if not self.patch_scores:
            return None
        return max(item.anomaly_score for item in self.patch_scores)

    @property
    def mean_anomaly_score(self) -> float | None:
        if not self.patch_scores:
            return None
        return float(np.mean([item.anomaly_score for item in self.patch_scores]))


class ReferenceNotReadyError(RuntimeError):
    """Raised when no usable reference FAISS index is available."""


class FaissAnomalyDetector:
    def __init__(self, metadata_path: Path, threshold: float) -> None:
        if not metadata_path.exists():
            raise ReferenceNotReadyError(f"Reference metadata not found: {metadata_path}")

        self.metadata_path = metadata_path
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.threshold = threshold
        self.reference_version = self.metadata.get("reference_version")
        self.metric = self.metadata.get("faiss_metric", "cosine")
        configured_index_path = Path(self.metadata["faiss_index_path"])
        self.index_path = configured_index_path if configured_index_path.is_absolute() else resolve_project_path(
            str(configured_index_path)
        )
        self.store = FaissIndexStore(metric=self.metric)
        self.index = self.store.load(self.index_path)

    def score(self, embeddings: np.ndarray, patch_boxes: list[PatchBox]) -> AnomalyDetectionResult:
        if embeddings.shape[0] != len(patch_boxes):
            raise ValueError("Embedding count must match patch box count.")
        if embeddings.shape[0] == 0:
            return AnomalyDetectionResult(
                status="empty_query",
                reference_version=self.reference_version,
                faiss_metric=self.metric,
                threshold=self.threshold,
                patch_scores=(),
            )

        search_result = self.store.search(self.index, embeddings, k=1)
        patch_scores = []
        for patch_index, (score_row, index_row) in enumerate(zip(search_result.scores, search_result.indices)):
            score = float(score_row[0])
            reference_offset = int(index_row[0])
            anomaly_score = self._to_anomaly_score(score)
            patch_scores.append(
                PatchAnomaly(
                    patch_index=patch_index,
                    reference_offset=reference_offset,
                    score=score,
                    anomaly_score=anomaly_score,
                    suspicious=anomaly_score >= self.threshold,
                    box=patch_boxes[patch_index],
                )
            )

        return AnomalyDetectionResult(
            status="scored",
            reference_version=self.reference_version,
            faiss_metric=self.metric,
            threshold=self.threshold,
            patch_scores=tuple(patch_scores),
        )

    def _to_anomaly_score(self, nearest_score: float) -> float:
        if self.metric == "cosine":
            return max(0.0, 1.0 - nearest_score)
        return max(0.0, nearest_score)
