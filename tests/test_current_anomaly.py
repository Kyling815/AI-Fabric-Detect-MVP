from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backend.app.ai import current_anomaly
from backend.app.ai.current_anomaly import CurrentAnomalyDetector, InvalidFabricReferenceError
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.core.inspection_contracts import DenseAnomalyEvidence, InspectionContext, InspectionInput
from backend.app.services.fabric_registry import FabricRegistry
from tests.phase1_fixtures import phase1_runtime


def context_for(runtime) -> InspectionContext:
    registry = FabricRegistry(runtime.registry_path)
    return InspectionContext(registry.get_active_reference("fixture_fabric"), registry.shared_backbone, "auto", 768)


def test_callable_component_uses_original_extractor_and_real_faiss(phase1_runtime) -> None:
    context = context_for(phase1_runtime)
    path = phase1_runtime.root / "input.png"
    path.write_bytes(phase1_runtime.png)
    evidence = CurrentAnomalyDetector().inspect(InspectionInput(path, "fixture_fabric"), context)
    assert isinstance(evidence, DenseAnomalyEvidence)
    phase1_runtime.factory.assert_called_once_with("shared_resnet18_imagenet1k_v1", "auto", False, "")
    phase1_runtime.extractor.extract_path.assert_called_once_with(path, image_max_size=24)
    np.testing.assert_array_equal(evidence.anomaly.patch_scores, [0, 0, 0, 0, 0, 1])
    assert evidence.anomaly.image_score == 1.0
    assert evidence.extractor_info is phase1_runtime.extractor.info


def test_missing_resize_setting_keeps_settings_fallback(phase1_runtime) -> None:
    context = replace(context_for(phase1_runtime), backbone={"backbone_version": "shared_resnet18_imagenet1k_v1"}, default_image_max_size=640)
    CurrentAnomalyDetector().inspect(InspectionInput(Path("input.png"), "fixture_fabric"), context)
    phase1_runtime.extractor.extract_path.assert_called_once_with(Path("input.png"), image_max_size=640)
    assert phase1_runtime.factory.call_args.args[2] is True


@pytest.mark.parametrize("failure", ["missing_metadata", "corrupt_metadata", "missing_index_key", "wrong_dimension"])
def test_reference_failures_are_domain_errors_without_http(phase1_runtime, failure: str) -> None:
    context = context_for(phase1_runtime)
    if failure == "missing_metadata":
        context.reference.metadata_path.unlink()
    elif failure == "corrupt_metadata":
        context.reference.metadata_path.write_text("{invalid", encoding="utf-8")
    elif failure == "missing_index_key":
        context.reference.metadata_path.write_text("{}", encoding="utf-8")
    else:
        FaissIndexStore().save(np.array([[1, 0, 0]], dtype=np.float32), context.reference.faiss_index_path)
    with pytest.raises(InvalidFabricReferenceError) as error:
        CurrentAnomalyDetector().inspect(InspectionInput(Path("input.png"), "fixture_fabric"), context)
    assert isinstance(error.value.__cause__, (OSError, KeyError, ValueError))
    assert not hasattr(error.value, "status_code")


def test_encoder_failure_is_not_reclassified_as_reference_failure(phase1_runtime) -> None:
    failure = ValueError("encoder failed")
    phase1_runtime.factory.side_effect = failure
    with pytest.raises(ValueError) as error:
        CurrentAnomalyDetector().inspect(InspectionInput(Path("input.png"), "fixture_fabric"), context_for(phase1_runtime))
    assert error.value is failure


def test_native_faiss_failure_preserves_baseline_exception_scope(phase1_runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    failure = RuntimeError("native index failure")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(current_anomaly, "DenseReferenceDetector", fail)
    with pytest.raises(RuntimeError) as error:
        CurrentAnomalyDetector().inspect(InspectionInput(Path("input.png"), "fixture_fabric"), context_for(phase1_runtime))
    assert error.value is failure
