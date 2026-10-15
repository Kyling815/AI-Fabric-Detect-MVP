from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from backend.app.ai.current_anomaly import CurrentAnomalyDetector
from backend.app.ai.lot_memory import FabricLotAnomalyMemory
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.core.lot_memory import LotMemoryConfig
from backend.app.services.inspection_service import InspectionService
from tests.lot_memory_fixtures import make_memory
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime


def test_global_only_hierarchy_matches_existing_dense_anomaly_evidence(phase1_runtime):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    evidence = CurrentAnomalyDetector().inspect(InspectionInput(Path("fixture.png"), "fixture_fabric"), context)
    global_memory = make_memory(fabric_id="fixture_fabric", vectors=np.load(context.reference.memory_bank_path))
    hierarchy = FabricLotAnomalyMemory(fabric_id="fixture_fabric", global_memory=global_memory, config=LotMemoryConfig())
    np.testing.assert_array_equal(hierarchy.search(evidence.features.embeddings).anomaly_distances[:, 0], evidence.anomaly.patch_scores)


def test_experimental_lot_union_search_cannot_mutate_existing_qc_or_evidence(phase1_runtime):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    value = InspectionInput(Path("fixture.png"), "fixture_fabric")
    before = service._inspect_fabric(value, context)
    embeddings = before.evidence.features.embeddings.copy()
    global_memory = make_memory(fabric_id="fixture_fabric", vectors=[[1, 0]])
    lot_memory = make_memory(scope="lot", fabric_id="fixture_fabric", vectors=[[0, 1]])
    hierarchy = FabricLotAnomalyMemory(fabric_id="fixture_fabric", lot_id="lot_001", global_memory=global_memory,
                                       lot_memory=lot_memory, config=LotMemoryConfig(enabled=True, mode="global_plus_lot"))
    experimental = hierarchy.search(before.evidence.features.embeddings)
    assert experimental.anomaly_distances.max() == 0
    after = service._inspect_fabric(value, context)
    np.testing.assert_array_equal(before.evidence.features.embeddings, embeddings)
    np.testing.assert_array_equal(after.evidence.anomaly.patch_scores, before.evidence.anomaly.patch_scores)
    assert before.decision == after.decision and before.candidates == after.candidates


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_production_api_never_calls_lot_hierarchy_and_keeps_golden_contract(phase1_runtime, case):
    phase1_runtime.use_case(case)
    with patch.object(FabricLotAnomalyMemory, "load", side_effect=AssertionError("production must not use research lot memory")):
        response = phase1_runtime.upload()
    assert response.status_code == 201
    snapshot = baseline()
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]
