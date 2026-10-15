import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select

from backend.app.ai.anomaly_branch import AnomalyDetector
from backend.app.ai.current_anomaly import CurrentAnomalyDetector, InvalidFabricReferenceError
from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.core.inspection_contracts import InspectionInput
from backend.app.db.models import Inspection
from backend.app.services.inspection_service import InspectionService
from tests.memory_experiment_fixtures import config, identity
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime


def write_metadata(runtime, values):
    path = Path(runtime.reference["metadata_path"])
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(values)
    path.write_text(json.dumps(current), encoding="utf-8")


def assert_failed_cleanly(runtime, response):
    assert response.status_code == 409, response.text
    assert not list((runtime.root / "uploads").iterdir())
    assert not list((runtime.root / "annotated").iterdir())
    with runtime.sessions() as db:
        assert db.scalar(select(func.count()).select_from(Inspection)) == 0


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_protocol_branch_keeps_golden_http_get_and_openapi_contract(phase1_runtime, case):
    assert isinstance(CurrentAnomalyDetector(), AnomalyDetector)
    phase1_runtime.use_case(case)
    service = InspectionService(None)
    assert isinstance(service.anomaly_detector, CurrentAnomalyDetector)
    response = phase1_runtime.upload()
    assert response.status_code == 201
    value = response.json()
    snapshot = baseline()
    assert normalize_response(value) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{value['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]


def test_offline_100_percent_memory_matches_dense_pipeline_patch_scores(phase1_runtime, tmp_path):
    service = InspectionService(None)
    context = service._resolve_fabric_context("fixture_fabric")
    evidence = CurrentAnomalyDetector().inspect(InspectionInput(Path("fixture.png"), "fixture_fabric"), context)
    values = np.load(context.reference.memory_bank_path, allow_pickle=False)
    original_bytes = context.reference.memory_bank_path.read_bytes()
    provenance = identity(fabric_id="fixture_fabric")
    memory = ExperimentalAnomalyMemory.build(values, identity=provenance, config=config())
    experimental = memory.search(evidence.features.embeddings)
    np.testing.assert_array_equal(experimental.anomaly_distances[:, 0], evidence.anomaly.patch_scores)
    np.testing.assert_array_equal(memory.bank, values)
    memory.save_new(tmp_path / "research")
    assert context.reference.memory_bank_path.read_bytes() == original_bytes


@pytest.mark.parametrize("key,value", [
    ("fabric_id", "fabric_b"), ("reference_id", "unrelated_ref"), ("reference_version", 99),
    ("backbone_version", "other_encoder"), ("adapter_version", "other_adapter"),
    ("schema_version", 99), ("schema_version", True), ("feature_dim", 999),
    ("faiss_metric", "l1"), ("faiss_metric", "l2"),
])
def test_declared_wrong_identity_or_version_rejected_with_409(phase1_runtime, key, value):
    write_metadata(phase1_runtime, {key: value})
    assert_failed_cleanly(phase1_runtime, phase1_runtime.upload())


def test_registry_declared_wrong_fabric_fails_before_upload_and_ai(phase1_runtime):
    phase1_runtime.reference["fabric_id"] = "fabric_b"
    phase1_runtime.save_registry()
    assert_failed_cleanly(phase1_runtime, phase1_runtime.upload())
    phase1_runtime.factory.assert_not_called()


def test_l2_index_cannot_masquerade_as_current_cosine_even_without_metadata_metric(phase1_runtime):
    FaissIndexStore("l2").save(np.array([[1, 0]], dtype=np.float32), Path(phase1_runtime.reference["faiss_index_path"]))
    assert_failed_cleanly(phase1_runtime, phase1_runtime.upload())


@pytest.mark.parametrize("key", ["faiss_index_path", "memory_bank_path"])
def test_registry_and_metadata_artifact_paths_cannot_disagree(phase1_runtime, key):
    if key == "faiss_index_path":
        path = phase1_runtime.root / "another.faiss"
        FaissIndexStore("cosine").save(np.array([[1, 0]], dtype=np.float32), path)
    else:
        path = phase1_runtime.root / "another.npy"
        np.save(path, np.array([[1, 0]], dtype=np.float32))
    write_metadata(phase1_runtime, {key: str(path)})
    assert_failed_cleanly(phase1_runtime, phase1_runtime.upload())


def test_direct_component_wrong_requested_fabric_is_a_domain_error(phase1_runtime):
    context = InspectionService(None)._resolve_fabric_context("fixture_fabric")
    with pytest.raises(InvalidFabricReferenceError, match="fabric"):
        CurrentAnomalyDetector().inspect(InspectionInput(Path("fixture.png"), "fabric_b"), context)


def test_explicit_research_artifact_is_never_accepted_by_production_route(phase1_runtime, tmp_path):
    path = tmp_path / "research"
    ExperimentalAnomalyMemory.build(np.array([[1, 0]], dtype=np.float32),
                                    identity=identity(fabric_id="fixture_fabric"), config=config()).save_new(path)
    for key in ("metadata_path", "memory_bank_path", "faiss_index_path"):
        filename = {"metadata_path": "reference.json", "memory_bank_path": "memory_bank.npy", "faiss_index_path": "memory_bank.faiss"}[key]
        phase1_runtime.reference[key] = str(path / filename)
    phase1_runtime.save_registry()
    response = phase1_runtime.upload()
    assert_failed_cleanly(phase1_runtime, response)
    assert "not calibrated or approved" in response.json()["detail"]


def test_complete_valid_legacy_provenance_keeps_exact_response(phase1_runtime):
    write_metadata(phase1_runtime, {
        "schema_version": 1, "fabric_id": "fixture_fabric", "reference_id": "fixture_ref_v1",
        "reference_version": 1, "backbone_version": "shared_resnet18_imagenet1k_v1",
        "adapter_version": None, "feature_dim": 2, "faiss_metric": "cosine", "memory_bank_path": "memory_bank.npy",
    })
    response = phase1_runtime.upload()
    assert response.status_code == 201
    assert normalize_response(response.json()) == baseline()["common"]
