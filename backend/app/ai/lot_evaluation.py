"""Fabric-only vs fabric+lot research comparison; no production labels or tuning."""

from pathlib import Path

import numpy as np

from backend.app.ai.dense_anomaly import aggregate_image_score
from backend.app.ai.experimental_memory import file_sha256
from backend.app.ai.lot_memory import FabricLotAnomalyMemory
from backend.app.core.lot_evaluation import LotEvaluationManifest
from backend.app.core.lot_memory import FabricLotMemoryManifest, LotMemoryConfig


def binary_auroc(labels: list[str], scores: list[float]) -> float | None:
    """Mann-Whitney AUC with average ranks for tied scores; higher = defect."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size != len(labels) or not np.isfinite(values).all():
        raise ValueError("AUC scores must be finite and match labels")
    if any(label not in {"normal", "defect"} for label in labels):
        raise ValueError("Unsupported AUC label")
    positives = np.array([label == "defect" for label in labels])
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[positives].sum() - positive_count * (positive_count + 1) / 2)
                 / (positive_count * negative_count))


def _path(value: str, base_dir: Path) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base_dir / candidate).resolve()


def evaluate_fabric_lot(manifest: FabricLotMemoryManifest, dataset: LotEvaluationManifest, *,
                        artifact_base_dir: Path = Path("."), dataset_base_dir: Path = Path(".")) -> dict:
    if not manifest.config.enabled or manifest.config.mode != "global_plus_lot":
        raise ValueError("Comparison requires explicit enabled global_plus_lot research mode")
    if (manifest.fabric_id, manifest.lot_id) != (dataset.fabric_id, dataset.lot_id):
        raise ValueError("Evaluation dataset fabric/current lot does not match hierarchy")
    combined = FabricLotAnomalyMemory.load(manifest, base_dir=artifact_base_dir)
    global_memory = combined.global_memory
    if (dataset.encoder_version, dataset.adapter_version) != (global_memory.identity.encoder_version,
                                                              global_memory.identity.adapter_version):
        raise ValueError("Evaluation embedding version does not match memory")
    baseline = FabricLotAnomalyMemory(fabric_id=manifest.fabric_id, global_memory=global_memory,
                                     config=LotMemoryConfig())
    sources = [source for memory in (combined.global_memory, combined.lot_memory) if memory is not None
               for source in memory.identity.source_good_images]
    source_ids = {source.image_id for source in sources}
    source_hashes = {source.sha256 for source in sources}
    # Source paths in Phase 4 provenance are interpreted relative to each child artifact.
    source_paths = set()
    artifact_dirs = []
    for spec, memory in ((manifest.global_memory, combined.global_memory), (manifest.lot_memory, combined.lot_memory)):
        if spec is not None and memory is not None:
            directory = _path(spec.path, artifact_base_dir)
            artifact_dirs.append(directory)
            source_paths.update(_path(source.path, directory) for source in memory.identity.source_good_images)
    per_image, labels, global_scores, combined_scores = [], [], [], []
    seen_paths = set()
    for sample in dataset.samples:
        image_path = _path(sample.image_path, dataset_base_dir)
        embeddings_path = _path(sample.embeddings_path, dataset_base_dir)
        if (sample.image_id in source_ids or sample.image_sha256 in source_hashes or image_path in source_paths
                or (sample.parent_source_sha256 is not None and sample.parent_source_sha256 in source_hashes)):
            raise ValueError("Reference/evaluation overlap or reference-source descendant detected")
        if image_path in seen_paths or embeddings_path in seen_paths:
            raise ValueError("Duplicate query paths")
        seen_paths.update((image_path, embeddings_path))
        if any(directory == embeddings_path or directory in embeddings_path.parents for directory in artifact_dirs):
            raise ValueError("Memory artifact vectors cannot be used as independent evaluation queries")
        if file_sha256(image_path) != sample.image_sha256 or file_sha256(embeddings_path) != sample.embeddings_sha256:
            raise ValueError("Evaluation source/embedding checksum mismatch")
        query = np.load(embeddings_path, allow_pickle=False)
        global_result, combined_result = baseline.search(query), combined.search(query)
        aggregation = dataset.aggregation
        kwargs = {"method": aggregation.image_score_method, "top_fraction": aggregation.image_score_top_fraction}
        global_score = aggregate_image_score(global_result.anomaly_distances[:, 0], **kwargs)
        combined_score = aggregate_image_score(combined_result.anomaly_distances[:, 0], **kwargs)
        labels.append(sample.label)
        global_scores.append(global_score)
        combined_scores.append(combined_score)
        per_image.append({"image_id": sample.image_id, "label": sample.label, "origin": sample.origin,
                          "query_feature_count": len(query), "fabric_only_score": global_score,
                          "fabric_plus_lot_score": combined_score,
                          "nearest_lot_patch_count": int((combined_result.memory_scopes[:, 0] == "lot").sum())})
    real_only = all(sample.origin == "real" for sample in dataset.samples)
    compared = combined.effective_mode == "global_plus_lot"
    global_auc = binary_auroc(labels, global_scores)
    combined_auc = binary_auroc(labels, combined_scores) if compared else None
    return {
        "phase": 5, "dataset_id": dataset.dataset_id, "fabric_id": dataset.fabric_id, "lot_id": dataset.lot_id,
        "evaluation_kind": "real_research_user_declared" if real_only else "synthetic_or_mixed_research",
        "independence_basis": dataset.independence_basis,
        "comparison_status": "COMPLETED_RESEARCH_ONLY" if compared else "NOT_COMPARED_MISSING_LOT_MEMORY",
        "improvement_status": "UNVERIFIED_PRODUCTION_BENEFIT",
        "real_evaluation_performed": real_only, "fresh_locked_test": False, "qc_performed": False,
        "effective_mode": combined.effective_mode, "fallback_reason": combined.fallback_reason,
        "distance_metric": combined.metric, "score_definition": combined.score_definition,
        "aggregation": dataset.aggregation.model_dump(),
        "global_vector_count": int(global_memory.index.ntotal),
        "lot_vector_count": int(combined.lot_memory.index.ntotal) if combined.lot_memory else 0,
        "fabric_only_auroc": global_auc, "fabric_plus_lot_auroc": combined_auc,
        "auroc_difference": (combined_auc - global_auc) if global_auc is not None and combined_auc is not None else None,
        "metric_note": "AUC requires both classes; scores are uncalibrated. Synthetic/mixed results prove software only. Source provenance/labels are caller-declared; overlap checks do not prove all independence.",
        "per_image": per_image,
    }
