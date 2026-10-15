"""Read-only bridges from existing dense anomaly and Phase 9 shadow diagnostics."""

import hashlib
import numpy as np

from backend.app.core.fusion import (AnomalyEvidence, CaptureEvidence, EvidenceIdentity, FusionBox,
                                     KnownClassEvidence, MaskDescriptor, SupervisedEvidence)
from backend.app.core.segmentation_shadow import SupervisedShadowObservation


def supervised_evidence_from_shadow(observation, identity: EvidenceIdentity, *, disabled: bool = False) -> SupervisedEvidence:
    if observation is None:
        return SupervisedEvidence(status="disabled" if disabled else "missing")
    try:
        observation = SupervisedShadowObservation.model_validate(observation.model_dump())
        if observation.status != "observed":
            status = {"unavailable": "missing", "timeout": "timeout", "error": "failed", "busy": "busy", "configuration_error": "failed"}[observation.status]
            return SupervisedEvidence(status=status, reason=observation.reason)
        return SupervisedEvidence(status="available", identity=identity, model_version=observation.model_version,
            checkpoint_sha256=observation.checkpoint_sha256, synthetic=observation.synthetic_checkpoint, taxonomy=observation.taxonomy,
            candidates=tuple(KnownClassEvidence(class_name=item.name, class_id=index, confidence=item.mean_winning_confidence,
                            predicted_pixels=item.predicted_pixels) for index, item in enumerate(observation.classes)
                            if index != 0 and item.predicted_pixels > 0),
            mask=MaskDescriptor(sha256=observation.mask_sha256, shape=observation.mask_shape, original_shape=observation.original_shape,
                                payload_available=not observation.mask_rle_omitted), confidence_calibrated=observation.confidence_calibrated)
    except Exception:
        return SupervisedEvidence(status="invalid", reason="invalid_shadow_payload")


def fusion_inputs_from_inspection(inspection, context, evidence, original_boxes, quality, shadow):
    # Invocation correlation, NOT source-content SHA/provenance attestation. Never expose upload paths in logs.
    capture_id = hashlib.sha256(str(inspection.image_path.resolve()).encode("utf-8")).hexdigest()
    identity = EvidenceIdentity(fabric_id=inspection.fabric_id, capture_id=capture_id)
    supervised = supervised_evidence_from_shadow(shadow, identity, disabled=context.supervised_shadow_config is None and context.supervised_shadow_error is None)
    try:
        result = evidence.anomaly
        scores = np.asarray(result.patch_scores)
        features = np.asarray(evidence.features.embeddings)
        anomaly_map = np.asarray(result.anomaly_map)
        if (scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all() or (scores < 0).any()
                or features.ndim != 2 or features.shape[0] != scores.size or not np.isfinite(features).all()
                or features.shape[1] != evidence.extractor_info.feature_dim
                or anomaly_map.shape != (evidence.features.grid_rows, evidence.features.grid_cols)
                or anomaly_map.size != scores.size or not np.array_equal(anomaly_map.reshape(-1), scores)
                or evidence.extractor_info.backbone_version != context.reference.backbone_version):
            raise ValueError("Dense evidence is incomplete/nonfinite or incompatible")
        trigger = bool(original_boxes and context.reference.thresholds.get("localized_review_threshold") is not None
                       and result.localized_score >= float(context.reference.thresholds["localized_review_threshold"]))
        anomaly = AnomalyEvidence(status="available", identity=identity, reference_version=context.reference.reference_id,
                    backbone_version=context.reference.backbone_version, score=float(result.image_score), localized_score=float(result.localized_score),
                    image_shape=(evidence.features.original_height, evidence.features.original_width),
                    boxes=tuple(FusionBox(x=box.x, y=box.y, w=box.w, h=box.h, score=box.score, label=box.label) for box in original_boxes),
                    score_method=context.reference.thresholds.get("image_score_method", "top_k_mean"), localized_review_trigger=trigger)
    except Exception:
        anomaly = AnomalyEvidence(status="invalid", reason="invalid_dense_anomaly_payload")
    try:
        capture = CaptureEvidence() if quality is None else CaptureEvidence(status="valid" if quality.valid else "invalid",
                    source="quality_gate", reasons=() if quality.valid else tuple(quality.reasons or ("capture_invalid",)))
    except Exception:
        capture = CaptureEvidence(status="error", reasons=("invalid_quality_payload",))
    return supervised, anomaly, capture
