from pathlib import Path

import pytest

from backend.app.ai.fusion import FusionEngine
from backend.app.core.fusion import (AnomalyEvidence, CaptureEvidence, DefectEvidence, EvidenceIdentity, FusionPolicy,
                                     KnownClassEvidence, MaskDescriptor, SupervisedEvidence)
from tests.fusion_fixtures import anomaly, capture, policy, supervised


def test_default_policy_and_example_are_observe_only_not_calibrated_qc():
    root = Path(__file__).resolve().parents[1]
    example = FusionPolicy.model_validate_json((root / "config/fusion_example.json").read_text())
    assert example == FusionPolicy() and example.mode == "observe_only" and not example.allow_synthetic_supervised


@pytest.mark.parametrize("values", [{"mode": "active"}, {"known_threshold": 0}, {"known_threshold": 1.1}, {"known_threshold": True},
    {"known_threshold": "0.9"}, {"anomaly_high_threshold": 0.2}, {"anomaly_review_threshold": 0.7},
    {"anomaly_high_threshold": float("nan")}, {"anomaly_review_threshold": float("inf")}, {"anomaly_high_threshold": True},
    {"allow_synthetic_supervised": "true"}, {"anomaly_score_method": "probability"}, {"policy_version": " "}, {"learned_weights": [1, 2]}])
def test_invalid_or_learned_or_decision_policy_rejected(values):
    with pytest.raises(ValueError):
        FusionPolicy(**values)


@pytest.mark.parametrize("values", [{"fabric_id": " ", "capture_id": "x"}, {"fabric_id": "x", "capture_id": ""}])
def test_identity_is_explicit_and_clean(values):
    with pytest.raises(ValueError):
        EvidenceIdentity(**values)


@pytest.mark.parametrize("values", [{"status": "valid"}, {"status": "invalid", "source": "quality_gate"},
                                  {"status": "valid", "source": "quality_gate", "reasons": ["blur"]}])
def test_capture_cannot_fabricate_verified_validity(values):
    with pytest.raises(ValueError):
        CaptureEvidence(**values)


@pytest.mark.parametrize("failure", ["missing_identity", "missing_version", "missing_hash", "missing_mask", "wrong_class", "duplicate_class", "excess_pixels", "missing_with_payload", "nan"])
def test_invalid_supervised_payload_rejected(failure):
    values = supervised().model_dump(mode="json")
    if failure.startswith("missing_") and failure != "missing_with_payload":
        values[{"missing_identity": "identity", "missing_version": "model_version", "missing_hash": "checkpoint_sha256", "missing_mask": "mask"}[failure]] = None
    elif failure == "wrong_class":
        values["candidates"][0]["class_name"] = "background"
    elif failure == "duplicate_class":
        values["candidates"].append(values["candidates"][0])
    elif failure == "excess_pixels":
        values["candidates"][0]["predicted_pixels"] = 1025
    elif failure == "missing_with_payload":
        values["status"] = "missing"
    else:
        values["candidates"][0]["confidence"] = float("nan")
    with pytest.raises(ValueError):
        SupervisedEvidence.model_validate(values)


@pytest.mark.parametrize("failure", ["missing_identity", "missing_score", "missing_version", "missing_geometry", "wrong_method", "invalid_box", "failed_with_score", "negative", "nan"])
def test_invalid_anomaly_payload_rejected(failure):
    values = anomaly().model_dump(mode="json")
    if failure.startswith("missing_"):
        values[{"missing_identity": "identity", "missing_score": "score", "missing_version": "reference_version", "missing_geometry": "image_shape"}[failure]] = None
    elif failure == "wrong_method":
        values["score_method"] = "unknown"
    elif failure == "invalid_box":
        values["boxes"][0]["w"] = 100
    elif failure == "failed_with_score":
        values["status"] = "failed"
    else:
        values["score"] = -1 if failure == "negative" else float("nan")
    with pytest.raises(ValueError):
        AnomalyEvidence.model_validate(values)


@pytest.mark.parametrize("values", [{"confidence": True}, {"class_id": True}, {"predicted_pixels": 0}, {"confidence": -1}])
def test_known_class_fields_are_strict(values):
    fields = dict(class_name="defect", class_id=1, confidence=0.5, predicted_pixels=1)
    fields.update(values)
    with pytest.raises(ValueError):
        KnownClassEvidence(**fields)


@pytest.mark.parametrize("shape", [(0, 32), (True, 32), (513, 32)])
def test_mask_descriptor_requires_bounded_integer_geometry(shape):
    with pytest.raises(ValueError):
        MaskDescriptor(sha256="a" * 64, shape=shape, original_shape=(32, 48), payload_available=True)


@pytest.mark.parametrize("failure", ["good_label", "active_qc", "normal_missing", "normal_unchecked", "review_waived", "normal_above_threshold", "missing_trace", "localized_positive_trace"])
def test_output_cannot_label_uncertain_state_normal_or_good(failure):
    value = FusionEngine(policy()).fuse(supervised(), anomaly(), capture=capture())
    fields = value.model_dump(mode="json")
    if failure == "good_label":
        fields["kind"] = "GOOD"
    elif failure == "active_qc":
        fields["affects_qc"] = True
    elif failure == "normal_missing":
        fields["supervised_status"] = "missing"
    elif failure == "normal_unchecked":
        fields["capture_status"] = "unchecked"
    elif failure == "review_waived":
        fields.update(kind="NEEDS_REVIEW", review_required=False)
    elif failure == "normal_above_threshold":
        fields["anomaly_score"] = 0.8
    elif failure == "missing_trace":
        fields["trace"] = []
    else:
        fields["trace"][-1]["passed"] = True
    with pytest.raises(ValueError):
        DefectEvidence.model_validate(fields)


@pytest.mark.parametrize("name", ["background", "unsupported_label"])
def test_standalone_known_class_never_accepts_background_or_unknown_taxonomy(name):
    with pytest.raises(ValueError):
        KnownClassEvidence(class_name=name, class_id=1, confidence=0.99, predicted_pixels=10)
