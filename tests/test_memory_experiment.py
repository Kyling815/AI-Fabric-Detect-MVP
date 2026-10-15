import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.ai.coreset import CoresetSelector, SeededRandomCoresetSelector
from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory, file_sha256, validate_research_output
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity
from tests.memory_experiment_fixtures import bank, config, identity


@pytest.mark.parametrize("metric", ["current", "cosine", "l2"])
def test_identical_memory_and_reload_reproduce_nearest_neighbors_and_distances(tmp_path, metric):
    values = bank()
    options, provenance = config(metric), identity()
    memory = ExperimentalAnomalyMemory.build(values, identity=provenance, config=options)
    first = memory.search(values[:6], k=3)
    repeated = memory.search(values[:6], k=3)
    np.testing.assert_array_equal(first.indices[:, 0], np.arange(6))
    np.testing.assert_allclose(first.anomaly_distances[:, 0], 0, atol=3e-7)
    np.testing.assert_array_equal(first.indices, repeated.indices)
    np.testing.assert_array_equal(first.anomaly_distances, repeated.anomaly_distances)
    path = tmp_path / "experiment"
    metadata = memory.save_new(path)
    loaded = ExperimentalAnomalyMemory.load(path, expected_identity=provenance, config=options)
    result = loaded.search(values[:6], k=3)
    np.testing.assert_array_equal(result.indices, first.indices)
    np.testing.assert_array_equal(result.faiss_scores, first.faiss_scores)
    np.testing.assert_array_equal(result.source_indices, first.source_indices)
    assert metadata.status == "EXPERIMENT_ONLY" and metadata.threshold_calibration == "NOT_CALIBRATED"
    assert metadata.threshold_profile_version is None
    assert metadata.distance_metric == ("cosine" if metric == "current" else metric)


@pytest.mark.parametrize("ratio", [1.0, 0.25, 0.10, 0.05])
def test_research_ratios_are_deterministic_unique_subsets_with_original_order(ratio):
    values = bank()
    selector = SeededRandomCoresetSelector()
    assert isinstance(selector, CoresetSelector)
    selected = selector.select(values, ratio=ratio, seed=42)
    repeated = selector.select(values, ratio=ratio, seed=42)
    assert len(selected.indices) == int(100 * ratio)
    assert selected.actual_ratio == ratio and selected.requested_ratio == ratio
    assert np.all(np.diff(selected.indices) > 0)
    np.testing.assert_array_equal(selected.indices, repeated.indices)
    np.testing.assert_array_equal(selected.embeddings, values[selected.indices])
    selected.embeddings[0, 0] = 999
    assert values[0, 0] != 999


def test_current_metric_100_percent_is_exact_historical_faiss_baseline():
    values = bank()
    query = bank()[10:20]
    store = FaissIndexStore("cosine")
    baseline = store.search(store.build(values), query, k=3)
    memory = ExperimentalAnomalyMemory.build(values, identity=identity(), config=config())
    result = memory.search(query, k=3)
    np.testing.assert_array_equal(memory.bank, values)
    np.testing.assert_array_equal(memory.selected_indices, np.arange(100))
    np.testing.assert_array_equal(result.faiss_scores, baseline.scores)
    np.testing.assert_array_equal(result.indices, baseline.indices)
    np.testing.assert_array_equal(result.anomaly_distances, np.maximum(0, 1 - baseline.scores))


def test_explicit_cosine_normalizes_both_memory_and_query_without_mutation():
    values = np.array([[3, 4], [-10, 0]], dtype=np.float32)
    queries = np.array([[30, 40], [-2, 0]], dtype=np.float32)
    original = values.copy()
    memory = ExperimentalAnomalyMemory.build(values, identity=identity(), config=config("cosine"))
    result = memory.search(queries)
    np.testing.assert_allclose(memory.bank, [[0.6, 0.8], [-1, 0]], atol=1e-7)
    np.testing.assert_allclose(result.faiss_scores[:, 0], 1, atol=1e-7)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(queries, [[30, 40], [-2, 0]])


def test_l2_is_squared_euclidean_not_cosine_or_manhattan():
    memory = ExperimentalAnomalyMemory.build(np.array([[0, 0], [8, 8]], dtype=np.float32),
                                             identity=identity(), config=config("l2"))
    result = memory.search(np.array([[3, 4]], dtype=np.float32))
    assert result.score_definition == "squared_l2"
    assert result.anomaly_distances[0, 0] == 25
    assert result.indices[0, 0] == 0


def test_ties_are_repeatable_for_identical_memory_not_claimed_unique():
    values = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    memory = ExperimentalAnomalyMemory.build(values, identity=identity(), config=config())
    first = memory.search(values[:1], k=2)
    second = memory.search(values[:1], k=2)
    np.testing.assert_array_equal(first.indices, second.indices)
    assert set(first.indices[0]) == {0, 1}


def test_small_bank_rounds_ratio_up_and_records_actual_ratio(tmp_path):
    memory = ExperimentalAnomalyMemory.build(np.eye(3, dtype=np.float32), identity=identity(), config=config(ratio=0.05))
    metadata = memory.save_new(tmp_path / "small")
    assert metadata.memory_bank_count == 1 and metadata.actual_coreset_ratio == pytest.approx(1 / 3)
    assert metadata.coreset_ratio == 0.05


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_coreset_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        SeededRandomCoresetSelector().select(bank(), ratio=ratio, seed=42)
    with pytest.raises(ValidationError):
        config(ratio=ratio)


@pytest.mark.parametrize("metric", ["l1", "manhattan", "ip", "L2", "unknown"])
def test_no_silent_metric_alias_or_l1_fallback(metric):
    with pytest.raises(ValidationError):
        config(metric)


@pytest.mark.parametrize("values", [np.empty((0, 2)), np.zeros(5), np.zeros((3, 0)),
                                   np.array([[np.nan, 1]]), np.array([[np.inf, 1]]),
                                   np.ones((2, 2), dtype=complex), np.array([["a"]])])
def test_bad_embedding_matrices_fail_before_search(values):
    with pytest.raises(ValueError):
        ExperimentalAnomalyMemory.build(values, identity=identity(), config=config())


@pytest.mark.parametrize("metric", ["current", "cosine"])
def test_cosine_zero_vectors_fail(metric):
    with pytest.raises(ValueError, match="zero"):
        ExperimentalAnomalyMemory.build(np.zeros((2, 2)), identity=identity(), config=config(metric))


def test_current_metric_refuses_unnormalized_vectors():
    with pytest.raises(ValueError, match="already L2-normalized"):
        ExperimentalAnomalyMemory.build(np.array([[3, 4]]), identity=identity(), config=config())


def test_disabled_experiment_is_not_built_or_loaded(tmp_path):
    with pytest.raises(ValueError, match="disabled"):
        ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=MemoryExperimentConfig())
    with pytest.raises(ValueError, match="disabled"):
        ExperimentalAnomalyMemory.load(tmp_path / "missing", expected_identity=identity(), config=MemoryExperimentConfig())


@pytest.mark.parametrize("field, value", [("fabric_id", "fabric_b"), ("encoder_version", "other_encoder"),
                                        ("adapter_version", "other_adapter"), ("reference_version", "v999"),
                                        ("lot_id", "other_lot"), ("threshold_profile_version", "different_profile")])
def test_reload_refuses_unrelated_identity_and_versions(tmp_path, field, value):
    provenance = identity()
    path = tmp_path / "experiment"
    ExperimentalAnomalyMemory.build(bank(), identity=provenance, config=config()).save_new(path)
    requested = provenance.model_copy(update={field: value})
    with pytest.raises(ValueError, match="provenance/version/fabric"):
        ExperimentalAnomalyMemory.load(path, expected_identity=requested, config=config())


@pytest.mark.parametrize("file", ["memory_bank.npy", "memory_bank.faiss", "selected_indices.npy"])
def test_corrupt_or_swapped_artifact_detected_by_hash(tmp_path, file):
    provenance, options = identity(), config()
    path = tmp_path / "experiment"
    ExperimentalAnomalyMemory.build(bank(), identity=provenance, config=options).save_new(path)
    (path / file).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ExperimentalAnomalyMemory.load(path, expected_identity=provenance, config=options)


@pytest.mark.parametrize("field, value", [("schema_version", 99), ("schema_version", True), ("distance_metric", "l1"),
                                        ("vector_policy", "raw"), ("score_definition", "squared_l2"),
                                        ("memory_bank_count", 999), ("actual_coreset_ratio", 0.5),
                                        ("faiss_index_path", "../other.faiss")])
def test_incompatible_or_inconsistent_metadata_rejected(tmp_path, field, value):
    path = tmp_path / "experiment"
    ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config()).save_new(path)
    content = json.loads((path / "reference.json").read_text())
    content[field] = value
    (path / "reference.json").write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError):
        ExperimentalAnomalyMemory.load(path, expected_identity=identity(), config=config())


def test_faiss_metric_swap_detected_even_if_file_hash_is_updated(tmp_path):
    path = tmp_path / "experiment"
    memory = ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config())
    memory.save_new(path)
    FaissIndexStore("l2").save(memory.bank, path / "memory_bank.faiss")
    content = json.loads((path / "reference.json").read_text())
    content["faiss_index_sha256"] = file_sha256(path / "memory_bank.faiss")
    (path / "reference.json").write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="FAISS metric"):
        ExperimentalAnomalyMemory.load(path, expected_identity=identity(), config=config())


def test_same_shape_index_with_wrong_vectors_is_not_accepted(tmp_path):
    path = tmp_path / "experiment"
    memory = ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config())
    memory.save_new(path)
    FaissIndexStore("cosine").save(-memory.bank, path / "memory_bank.faiss")
    content = json.loads((path / "reference.json").read_text())
    content["faiss_index_sha256"] = file_sha256(path / "memory_bank.faiss")
    (path / "reference.json").write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="stored vector"):
        ExperimentalAnomalyMemory.load(path, expected_identity=identity(), config=config())


def test_refuses_existing_output_and_preserves_bytes(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    marker = path / "marker.txt"
    marker.write_bytes(b"user work")
    memory = ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config())
    with pytest.raises(FileExistsError):
        memory.save_new(path)
    assert marker.read_bytes() == b"user work" and list(path.iterdir()) == [marker]


@pytest.mark.parametrize("relative", ["Data/new", "storage/references/new", "storage/models/new",
                                      "storage/metadata/new", "storage/indexes/new", "storage", "storage/experiments"])
def test_output_guard_blocks_production_and_dataset_roots(relative):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="Research output"):
        validate_research_output(root / relative)


def test_untruthful_injected_selector_is_rejected():
    selector = Mock(spec=CoresetSelector)
    selected = SeededRandomCoresetSelector().select(bank(), ratio=0.25, seed=3)
    selector.select.return_value = selected
    with pytest.raises(ValueError, match="seed"):
        ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config(ratio=0.25), selector=selector)


def test_source_good_identity_and_split_must_be_consistent():
    data = identity().model_dump(mode="json")
    data["source_good_images"][0]["fabric_id"] = "fabric_b"
    with pytest.raises(ValidationError, match="fabric"):
        MemoryIdentity.model_validate(data)
    data = identity().model_dump(mode="json")
    data["source_good_images"][0]["split"] = "locked_test"
    with pytest.raises(ValidationError):
        MemoryIdentity.model_validate(data)
    data = identity().model_dump(mode="json")
    data["lot_id"] = "lot_unknown"
    with pytest.raises(ValidationError, match="lot"):
        MemoryIdentity.model_validate(data)


@pytest.mark.parametrize("k", [0, -1, 101, True, 1.5])
def test_invalid_neighbor_count_rejected(k):
    memory = ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config())
    with pytest.raises(ValueError, match="k must"):
        memory.search(bank()[:1], k=k)


def test_query_dimension_and_large_l2_values_fail_safely():
    memory = ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config())
    with pytest.raises(ValueError, match="dimension"):
        memory.search(np.array([[1, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="overflow"):
        ExperimentalAnomalyMemory.build(np.array([[1e30, 0]]), identity=identity(), config=config("l2"))


@pytest.mark.parametrize("override", [{"enabled": "true"}, {"random_seed": True}, {"random_seed": -1},
                                     {"selector": "k_center"}, {"coreset_ratio": "0.25"}])
def test_invalid_opt_in_seed_or_unimplemented_selector_rejected(override):
    with pytest.raises(ValidationError):
        MemoryExperimentConfig.model_validate(override)


@pytest.mark.parametrize("requested", [config("l2"), config(ratio=0.25), config(random_seed=99)])
def test_reload_refuses_different_requested_configuration(tmp_path, requested):
    path = tmp_path / "experiment"
    ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config()).save_new(path)
    with pytest.raises(ValueError, match="configuration"):
        ExperimentalAnomalyMemory.load(path, expected_identity=identity(), config=requested)


def test_compatible_selector_injection_is_used():
    selector = Mock(spec=CoresetSelector)
    selector.select.side_effect = SeededRandomCoresetSelector().select
    memory = ExperimentalAnomalyMemory.build(bank(), identity=identity(), config=config(ratio=0.25), selector=selector)
    selector.select.assert_called_once()
    assert len(memory.bank) == 25
