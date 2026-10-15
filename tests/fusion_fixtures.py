"""Hypothetical/synthetic evidence contracts only; no trained-model accuracy claims."""

from backend.app.core.fusion import (AnomalyEvidence, CaptureEvidence, EvidenceIdentity, FusionBox, FusionPolicy,
                                     KnownClassEvidence, MaskDescriptor, SupervisedEvidence)


def identity(fabric_id="fixture_fabric", capture_id="fixture_capture"):
    return EvidenceIdentity(fabric_id=fabric_id, capture_id=capture_id)


def policy(**updates):
    return FusionPolicy(allow_synthetic_supervised=True, **updates)


def capture(status="valid"):
    return CaptureEvidence(status=status, source="caller_declared" if status in {"valid", "invalid"} else "not_checked",
                           reasons=("fixture_rejection",) if status in {"invalid", "error"} else ())


def supervised(confidence=0.2, *, foreground=True, synthetic=True, payload=True):
    return SupervisedEvidence(status="available", identity=identity(), model_version="hypothetical_model_v1", checkpoint_sha256="a" * 64,
        synthetic=synthetic, taxonomy=("background", "defect"), mask=MaskDescriptor(sha256="b" * 64, shape=(32, 32),
                           original_shape=(32, 48), payload_available=payload),
        candidates=(KnownClassEvidence(class_name="defect", class_id=1, confidence=confidence, predicted_pixels=128),) if foreground else ())


def anomaly(score=0.1, *, localized=False, boxes=True, method="top_k_mean"):
    return AnomalyEvidence(status="available", identity=identity(), reference_version="hypothetical_ref_v1", backbone_version="fixture_backbone_v1",
        score=score, localized_score=0.8 if localized else 0.1, image_shape=(32, 48), score_method=method, localized_review_trigger=localized,
        boxes=(FusionBox(x=2, y=3, w=4, h=5, score=score, label="dense_anomaly"),) if boxes else ())
