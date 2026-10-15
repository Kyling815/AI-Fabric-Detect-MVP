from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from backend.app.ai.fusion import FusionEngine
from backend.app.ai.fusion_adapter import fusion_inputs_from_inspection, supervised_evidence_from_shadow
from backend.app.ai.preprocessing.quality import QualityMetrics
from backend.app.ai.quality_gate import QualityGateResult
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.segmentation_shadow import ShadowClassSummary, SupervisedShadowObservation
from backend.app.services.inspection_service import InspectionService
from tests.fusion_fixtures import identity, policy
from tests.phase1_fixtures import phase1_runtime
from tests.segmentation_shadow_fixtures import observed_fixture


@pytest.mark.parametrize("status,expected", [("unavailable", "missing"), ("timeout", "timeout"), ("error", "failed"), ("busy", "busy"), ("configuration_error", "failed")])
def test_shadow_failure_to_evidence_does_not_invent_zero_confidence(status, expected):
    result = supervised_evidence_from_shadow(SupervisedShadowObservation(status=status, reason="fixture"), identity())
    assert result.status == expected and not result.candidates and result.mask is None and result.model_version is None


def test_shadow_success_preserves_class_mask_provenance_not_background_confidence():
    observation = observed_fixture()
    result = supervised_evidence_from_shadow(observation, identity())
    assert result.status == "available" and result.model_version == observation.model_version
    assert result.candidates[0].confidence == 0.99 and result.candidates[0].predicted_pixels == 1024
    assert result.mask.sha256 == observation.mask_sha256 and result.mask.coordinate_frame == "model_grid"
    assert result.mask.original_shape == observation.original_shape and result.synthetic and not result.confidence_calibrated


def test_shadow_background_only_has_no_known_candidates():
    observation = observed_fixture()
    fields = observation.model_dump(mode="json")
    fields.update(classes=[ShadowClassSummary(name="background", predicted_pixels=1024, mean_class_probability=0.99, mean_winning_confidence=0.99).model_dump(),
                           ShadowClassSummary(name="defect", predicted_pixels=0, mean_class_probability=0.01).model_dump()],
                  mask_rle=[[0, 1024]], mask_sha256=hashlib.sha256(b"\x00" * (4 * 1024)).hexdigest())
    result = supervised_evidence_from_shadow(SupervisedShadowObservation.model_validate(fields), identity())
    assert result.status == "available" and not result.candidates


def test_shadow_model_copy_bypass_is_not_trusted_by_adapter():
    observation = observed_fixture().model_copy(update={"affects_qc": True})
    assert supervised_evidence_from_shadow(observation, identity()).status == "invalid"


def test_missing_or_disabled_shadow_is_explicit():
    assert supervised_evidence_from_shadow(None, identity()).status == "missing"
    assert supervised_evidence_from_shadow(None, identity(), disabled=True).status == "disabled"


def test_existing_dense_and_shadow_geometry_identity_and_quality_are_read_only(phase1_runtime):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    value = InspectionInput(Path("unused.png"), "fixture_fabric")
    result = service._inspect_fabric(value, context)
    quality = QualityGateResult(QualityMetrics(width=48, height=32, decode_success=True))
    before = result.evidence.features.embeddings.copy(), result.evidence.anomaly.patch_scores.copy()
    sup, ano, cap = fusion_inputs_from_inspection(value, context, result.evidence, result.candidates, quality, observed_fixture())
    assert sup.identity == ano.identity and sup.mask.original_shape == ano.image_shape == (32, 48)
    assert len(ano.identity.capture_id) == 64 and "unused.png" not in ano.model_dump_json()
    assert cap.status == "valid" and cap.source == "quality_gate"
    assert FusionEngine(policy()).fuse(sup, ano, capture=cap).kind == "KNOWN_DEFECT"
    np.testing.assert_array_equal(before[0], result.evidence.features.embeddings)
    np.testing.assert_array_equal(before[1], result.evidence.anomaly.patch_scores)


@pytest.mark.parametrize("failure", ["empty", "nan", "map_mismatch", "backbone", "feature_count"])
def test_bad_dense_evidence_not_interpreted_as_normal(phase1_runtime, failure):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    value = InspectionInput(Path("unused.png"), "fixture_fabric")
    result = service._inspect_fabric(value, context)
    evidence = result.evidence
    if failure in {"empty", "nan", "map_mismatch"}:
        changes = {"empty": {"patch_scores": np.array([])}, "nan": {"patch_scores": np.full(6, np.nan)}, "map_mismatch": {"anomaly_map": np.ones((2, 3))}}
        evidence = replace(evidence, anomaly=replace(evidence.anomaly, **changes[failure]))
    elif failure == "backbone":
        evidence = replace(evidence, extractor_info=replace(evidence.extractor_info, backbone_version="other"))
    else:
        evidence = replace(evidence, features=replace(evidence.features, embeddings=np.zeros((2, 2))))
    _, ano, _ = fusion_inputs_from_inspection(value, context, evidence, result.candidates, None, None)
    assert ano.status == "invalid" and ano.score is None


def test_quality_none_is_unchecked_not_fabricated_valid_and_invalid_is_capture_status(phase1_runtime):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    value = InspectionInput(Path("unused.png"), "fixture_fabric")
    result = service._inspect_fabric(value, context)
    _, _, cap = fusion_inputs_from_inspection(value, context, result.evidence, result.candidates, None, None)
    assert cap.status == "unchecked"
    quality = QualityGateResult(QualityMetrics(decode_success=True), reasons=("blur",))
    _, _, cap = fusion_inputs_from_inspection(value, context, result.evidence, result.candidates, quality, None)
    assert cap.status == "invalid" and cap.reasons == ("blur",)
