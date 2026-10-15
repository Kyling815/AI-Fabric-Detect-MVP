from dataclasses import replace
import json

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory
from backend.app.ai.pattern_memory import PatternAwareAnomalyMemory
from backend.app.core.memory_experiment import MemoryExperimentConfig
from backend.app.core.pattern_pooling import FeatureGeometry
from tests.pattern_pooling_fixtures import feature_result, identity_for


@pytest.mark.parametrize("enabled,mode", [(False, "period_aware"), (True, "period_aware"), (True, "fixed_multiscale")])
def test_matching_geometry_faiss_save_reload_reproduces_scores(tmp_path, enabled, mode):
    value = feature_result(enabled, mode)
    identity, config = identity_for(value), MemoryExperimentConfig(enabled=True)
    memory = PatternAwareAnomalyMemory.build([value], identity=identity, config=config)
    before = memory.search(value)
    path = tmp_path / "new_memory"
    manifest = memory.save_new(path)
    restored = PatternAwareAnomalyMemory.load(path, expected_geometry=value.geometry, expected_identity=identity, config=config)
    np.testing.assert_array_equal(before.anomaly_distances, restored.search(value).anomaly_distances)
    assert manifest.geometry_signature == value.geometry.signature


@pytest.mark.parametrize("reference,query", [(False, True), (True, False)])
def test_same_dimension_baseline_and_adaptive_geometries_are_incompatible(reference, query):
    value, query = feature_result(reference), feature_result(query)
    assert value.features.embeddings.shape[1] == query.features.embeddings.shape[1]
    memory = PatternAwareAnomalyMemory.build([value], identity=identity_for(value), config=MemoryExperimentConfig(enabled=True))
    with pytest.raises(ValueError, match="geometry mismatch"):
        memory.search(query)


def test_fixed_and_adaptive_geometry_cannot_mix_even_if_fallback_values_match():
    fixed, adaptive = feature_result(True, "fixed_multiscale"), feature_result(True)
    with pytest.raises(ValueError, match="different representation"):
        PatternAwareAnomalyMemory.build([fixed, adaptive], identity=identity_for(fixed), config=MemoryExperimentConfig(enabled=True))


def test_missing_geometry_manifest_rejects_legacy_equal_dimension_bank(tmp_path):
    value = feature_result(True)
    config, identity = MemoryExperimentConfig(enabled=True), identity_for(value)
    ExperimentalAnomalyMemory.build(value.features.embeddings, identity=identity, config=config).save_new(tmp_path / "legacy")
    with pytest.raises(FileNotFoundError):
        PatternAwareAnomalyMemory.load(tmp_path / "legacy", expected_geometry=value.geometry, expected_identity=identity, config=config)


@pytest.mark.parametrize("field, value", [("model_weights_sha256", "1" * 64), ("image_max_size", 512), ("feature_dim", 3)])
def test_weight_preprocessing_or_dimension_changes_require_matching_bank(tmp_path, field, value):
    features = feature_result(True)
    config, identity = MemoryExperimentConfig(enabled=True), identity_for(features)
    memory = PatternAwareAnomalyMemory.build([features], identity=identity, config=config)
    memory.save_new(tmp_path / "memory")
    changed = features.geometry.model_copy(update={field: value})
    with pytest.raises(ValueError, match="incompatible"):
        PatternAwareAnomalyMemory.load(tmp_path / "memory", expected_geometry=changed, expected_identity=identity, config=config)


def test_geometry_fingerprint_tampering_fails(tmp_path):
    value = feature_result(True)
    config, identity = MemoryExperimentConfig(enabled=True), identity_for(value)
    PatternAwareAnomalyMemory.build([value], identity=identity, config=config).save_new(tmp_path / "memory")
    path = tmp_path / "memory" / "pattern_geometry.json"
    metadata = json.loads(path.read_text())
    metadata["geometry_signature"] = "0" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="signature mismatch"):
        PatternAwareAnomalyMemory.load(tmp_path / "memory", expected_geometry=value.geometry, expected_identity=identity, config=config)


def test_new_bank_requires_enabled_experiment_and_geometry_pinned_identity():
    value = feature_result(True)
    with pytest.raises(ValueError, match="disabled"):
        PatternAwareAnomalyMemory.build([value], identity=identity_for(value), config=MemoryExperimentConfig())
    bad_identity = identity_for(value).model_copy(update={"encoder_version": "legacy_same_dim"})
    with pytest.raises(ValueError, match="exact feature geometry"):
        PatternAwareAnomalyMemory.build([value], identity=bad_identity, config=MemoryExperimentConfig(enabled=True))


@pytest.mark.parametrize("version", [2, True, "1"])
def test_unknown_geometry_schema_rejected(version):
    value = feature_result().geometry.model_dump()
    value["schema_version"] = version
    with pytest.raises(ValidationError):
        FeatureGeometry.model_validate(value)


def test_empty_reference_features_rejected():
    with pytest.raises(ValueError, match="required"):
        PatternAwareAnomalyMemory.build([], identity=identity_for(feature_result()), config=MemoryExperimentConfig(enabled=True))


@pytest.mark.parametrize("field", ["backbone_version", "architecture"])
def test_blank_geometry_identifiers_are_rejected(field):
    value = feature_result().geometry.model_dump()
    value[field] = " "
    with pytest.raises(ValidationError, match="nonblank"):
        FeatureGeometry.model_validate(value)
