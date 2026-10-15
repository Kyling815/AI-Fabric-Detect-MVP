from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from backend.app.ai.bbox import MergedBox
from backend.app.ai.dense_anomaly import DenseAnomalyResult
from backend.app.ai.fabric_decision import FabricDecision
from backend.app.core.inspection_contracts import (
    DefectCandidate, DenseAnomalyEvidence, InspectionContext, InspectionInput, InspectionResult,
)
from backend.app.services.fabric_registry import FabricReferenceEntry
from tests.phase1_fixtures import dense_features, fake_extractor


def test_input_contract_is_internal_and_frozen() -> None:
    value = InspectionInput(Path("capture.png"), "fabric_001")
    assert value.image_path == Path("capture.png")
    with pytest.raises(FrozenInstanceError):
        value.fabric_id = "fabric_002"


def test_candidate_reuses_existing_geometry_without_conversion() -> None:
    assert DefectCandidate is MergedBox
    candidate = DefectCandidate(2, 3, 4, 5, 0.75, patch_count=6, label="dense_anomaly")
    assert candidate.area == 20
    assert candidate.patch_count == 6 and candidate.label == "dense_anomaly"


def test_context_keeps_exact_reference_and_runtime_configuration(tmp_path: Path) -> None:
    reference = FabricReferenceEntry(
        "fabric_001", "Fabric A", "ref_v1", 1, "ACTIVE", "shared_v1",
        tmp_path, tmp_path / "reference.json", tmp_path / "bank.npy", tmp_path / "bank.faiss", {"pixel_threshold": 0.2},
    )
    context = InspectionContext(reference, {"backbone_version": "shared_v1"}, "auto", 768)
    assert context.reference is reference
    assert context.reference.thresholds == {"pixel_threshold": 0.2}
    assert context.default_image_max_size == 768
    with pytest.raises(FrozenInstanceError):
        context.device = "cuda"


def test_result_keeps_evidence_separate_from_qc_decision() -> None:
    scores = np.zeros(6, dtype=np.float32)
    anomaly = DenseAnomalyResult(scores, 0.0, 0.0, scores.reshape(2, 3), ())
    evidence = DenseAnomalyEvidence(dense_features(), fake_extractor().info, anomaly)
    result = InspectionResult(evidence, (), FabricDecision("NEEDS_REVIEW", "TEST_POLICY"), 0.0)
    assert result.evidence is evidence
    assert result.evidence.anomaly.image_score == 0
    assert result.decision.label == "NEEDS_REVIEW"
    with pytest.raises(FrozenInstanceError):
        result.defect_area_ratio = 1.0
