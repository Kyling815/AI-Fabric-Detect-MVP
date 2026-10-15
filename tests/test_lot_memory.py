from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.ai.lot_memory import FabricLotAnomalyMemory, MissingLotMemoryError
from backend.app.core.lot_memory import FabricLotMemoryManifest, LotMemoryConfig, MemoryArtifactSpec
from tests.lot_memory_fixtures import make_memory


@pytest.mark.parametrize("metric", ["current", "cosine", "l2"])
@pytest.mark.parametrize("mode", ["global_only", "lot_only", "global_plus_lot"])
def test_selected_scopes_and_union_distances(metric, mode):
    global_memory, lot_memory = make_memory(metric=metric), make_memory(scope="lot", metric=metric)
    query = np.array([[0.8, 0.6], [1, 0], [-1, 0]], dtype=np.float32)
    hierarchy = FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id="lot_001", global_memory=global_memory,
                                       lot_memory=lot_memory, config=LotMemoryConfig(enabled=True, mode=mode))
    result = hierarchy.search(query)
    global_dist = global_memory.search(query).anomaly_distances
    lot_dist = lot_memory.search(query).anomaly_distances
    expected = global_dist if mode == "global_only" else lot_dist if mode == "lot_only" else np.minimum(global_dist, lot_dist)
    np.testing.assert_allclose(result.anomaly_distances, expected, atol=1e-7, rtol=0)
    assert result.effective_mode == mode
    if mode == "global_only":
        assert set(result.memory_scopes[:, 0]) == {"global"}
    elif mode == "lot_only":
        assert set(result.memory_scopes[:, 0]) == {"lot"}
    else:
        assert set(result.memory_scopes[:, 0]) == {"global", "lot"}
    for scope, index, original, version in zip(result.memory_scopes[:, 0], result.indices[:, 0],
                                               result.source_indices[:, 0], result.reference_versions[:, 0]):
        memory = global_memory if scope == "global" else lot_memory
        assert memory.selected_indices[index] == original and memory.identity.reference_version == version


def test_disabled_flag_ignores_unrelated_lot_and_is_exact_global_baseline():
    global_memory = make_memory()
    unrelated = make_memory(scope="lot", fabric_id="fabric_other", lot_id="another_lot")
    query = np.array([[0.8, 0.6]], dtype=np.float32)
    hierarchy = FabricLotAnomalyMemory(fabric_id="fabric_001", global_memory=global_memory, lot_memory=unrelated,
                                       lot_id="invalid/unused", config=LotMemoryConfig(mode="global_plus_lot"))
    np.testing.assert_array_equal(hierarchy.search(query).faiss_scores, global_memory.search(query).faiss_scores)
    assert hierarchy.lot_memory is None and hierarchy.effective_mode == "global_only"


@pytest.mark.parametrize("mode,policy", [("global_plus_lot", "error"), ("lot_only", "error"), ("lot_only", "global_only")])
def test_missing_lot_is_rejected_unless_explicit_combined_fallback(mode, policy):
    with pytest.raises(MissingLotMemoryError):
        FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id="lot_001", global_memory=make_memory(),
                               config=LotMemoryConfig(enabled=True, mode=mode, missing_lot_policy=policy))


def test_explicit_missing_lot_fallback_reports_reason_and_matches_global():
    memory = make_memory()
    hierarchy = FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id="lot_001", global_memory=memory,
                                       config=LotMemoryConfig(enabled=True, mode="global_plus_lot", missing_lot_policy="global_only"))
    result = hierarchy.search(np.array([[1, 0]], dtype=np.float32))
    assert result.effective_mode == "global_only" and result.fallback_reason == "missing_lot_memory"
    assert result.anomaly_distances[0, 0] == 0


@pytest.mark.parametrize("lot_id", [None, "", " ", "../lot", "lot/a", "lot\\a", "lot\n1", "a" * 129, 3])
def test_invalid_current_lot_never_silently_falls_back(lot_id):
    with pytest.raises(ValueError, match="Invalid lot_id"):
        FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id=lot_id, global_memory=make_memory(),
                               config=LotMemoryConfig(enabled=True, mode="global_plus_lot", missing_lot_policy="global_only"))


@pytest.mark.parametrize("wrong_scope", ["global", "lot"])
def test_unrelated_fabric_memory_never_mixed(wrong_scope):
    global_memory = make_memory(fabric_id="other" if wrong_scope == "global" else "fabric_001")
    lot_memory = make_memory(scope="lot", fabric_id="other" if wrong_scope == "lot" else "fabric_001")
    with pytest.raises(ValueError, match="Unrelated fabric"):
        FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id="lot_001", global_memory=global_memory, lot_memory=lot_memory,
                               config=LotMemoryConfig(enabled=True, mode="global_plus_lot"))


@pytest.mark.parametrize("override", [{"lot_id": "wrong_lot"}, {"encoder_version": "other_encoder"},
                                     {"adapter_version": "other_adapter"}, {"metric": "cosine"},
                                     {"vectors": [[1, 0, 0]]}])
def test_incompatible_scope_feature_space_or_policy_rejected(override):
    with pytest.raises(ValueError, match="mismatch|scope"):
        FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id="lot_001", global_memory=make_memory(),
                               lot_memory=make_memory(scope="lot", **override),
                               config=LotMemoryConfig(enabled=True, mode="global_plus_lot"))


def test_global_memory_must_not_be_lot_scoped():
    with pytest.raises(ValueError, match="scope"):
        FabricLotAnomalyMemory(fabric_id="fabric_001", global_memory=make_memory(scope="lot"), config=LotMemoryConfig())


def artifact_manifest(tmp_path, *, enabled=True, missing_lot_policy="error"):
    specs = {}
    for scope in ("global", "lot"):
        memory = make_memory(scope=scope)
        memory.save_new(tmp_path / scope)
        specs[scope] = MemoryArtifactSpec(path=scope, identity=memory.identity, config=memory.config)
    return FabricLotMemoryManifest(fabric_id="fabric_001", lot_id="lot_001",
                                    global_memory=specs["global"], lot_memory=specs["lot"],
                                    config=LotMemoryConfig(enabled=enabled, mode="global_plus_lot", missing_lot_policy=missing_lot_policy))


def test_hierarchy_manifest_roundtrip_reload_is_exact(tmp_path):
    manifest = artifact_manifest(tmp_path)
    reloaded = FabricLotMemoryManifest.model_validate_json(manifest.model_dump_json())
    first, second = FabricLotAnomalyMemory.load(manifest, base_dir=tmp_path), FabricLotAnomalyMemory.load(reloaded, base_dir=tmp_path)
    query = np.array([[0.8, 0.6], [1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(first.search(query, k=3).anomaly_distances, second.search(query, k=3).anomaly_distances)
    np.testing.assert_array_equal(first.search(query, k=3).memory_scopes, second.search(query, k=3).memory_scopes)
    with pytest.raises(ValidationError):
        manifest.fabric_id = "other"
    with pytest.raises(FrozenInstanceError):
        first.search(query).effective_mode = "lot_only"


def test_disabled_load_does_not_read_missing_or_wrong_lot_artifact(tmp_path):
    manifest = artifact_manifest(tmp_path, enabled=False)
    wrong_identity = manifest.lot_memory.identity.model_copy(update={"fabric_id": "other"})
    manifest = manifest.model_copy(update={"lot_memory": manifest.lot_memory.model_copy(update={"path": "missing", "identity": wrong_identity})})
    memory = FabricLotAnomalyMemory.load(manifest, base_dir=tmp_path)
    assert memory.lot_memory is None and memory.effective_mode == "global_only"


@pytest.mark.parametrize("policy", ["error", "global_only"])
def test_missing_lot_directory_uses_only_declared_policy(tmp_path, policy):
    manifest = artifact_manifest(tmp_path, missing_lot_policy=policy)
    manifest = manifest.model_copy(update={"lot_memory": manifest.lot_memory.model_copy(update={"path": "absent"})})
    if policy == "error":
        with pytest.raises(MissingLotMemoryError):
            FabricLotAnomalyMemory.load(manifest, base_dir=tmp_path)
    else:
        assert FabricLotAnomalyMemory.load(manifest, base_dir=tmp_path).fallback_reason == "missing_lot_memory"


@pytest.mark.parametrize("failure", ["wrong_version", "wrong_schema", "corrupt_index", "missing_manifest", "wrong_fabric_identity"])
def test_bad_lot_artifact_never_falls_back_to_mask_corruption(tmp_path, failure):
    manifest = artifact_manifest(tmp_path, missing_lot_policy="global_only")
    if failure == "wrong_version":
        changed = manifest.lot_memory.identity.model_copy(update={"reference_version": "v999"})
        manifest = manifest.model_copy(update={"lot_memory": manifest.lot_memory.model_copy(update={"identity": changed})})
    elif failure == "wrong_fabric_identity":
        changed = manifest.lot_memory.identity.model_copy(update={"fabric_id": "other"})
        manifest = manifest.model_copy(update={"lot_memory": manifest.lot_memory.model_copy(update={"identity": changed, "path": "absent"})})
    elif failure == "corrupt_index":
        (tmp_path / "lot" / "memory_bank.faiss").write_bytes(b"invalid")
    elif failure == "missing_manifest":
        (tmp_path / "lot" / "reference.json").unlink()
    else:
        path = tmp_path / "lot" / "reference.json"
        content = json.loads(path.read_text())
        content["schema_version"] = 999
        path.write_text(json.dumps(content))
    with pytest.raises((ValueError, FileNotFoundError)):
        FabricLotAnomalyMemory.load(manifest, base_dir=tmp_path)


@pytest.mark.parametrize("bad_config", [{"enabled": "true"}, {"mode": "auto"}, {"missing_lot_policy": "ignore"}])
def test_invalid_configuration_rejected(bad_config):
    with pytest.raises(ValidationError):
        LotMemoryConfig.model_validate(bad_config)


@pytest.mark.parametrize("version", [2, True, "1"])
def test_hierarchy_version_mismatch_rejected(version):
    with pytest.raises(ValidationError):
        FabricLotMemoryManifest.model_validate({"schema_version": version, "fabric_id": "fabric_001"})


@pytest.mark.parametrize("k", [0, -1, 5, True, 1.5])
def test_union_search_invalid_neighbor_count_rejected(k):
    hierarchy = FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id="lot_001", global_memory=make_memory(),
                                       lot_memory=make_memory(scope="lot"), config=LotMemoryConfig(enabled=True, mode="global_plus_lot"))
    with pytest.raises(ValueError):
        hierarchy.search(np.array([[1, 0]], dtype=np.float32), k=k)


def test_example_default_off_and_query_dimension_check():
    root = Path(__file__).resolve().parents[1]
    config = LotMemoryConfig.model_validate_json((root / "config/lot_memory_example.json").read_text())
    assert not config.enabled
    hierarchy = FabricLotAnomalyMemory(fabric_id="fabric_001", global_memory=make_memory(), config=config)
    with pytest.raises(ValueError, match="dimension"):
        hierarchy.search(np.array([[1, 0, 0]], dtype=np.float32))


def test_lot_only_load_does_not_require_or_read_global_artifact(tmp_path):
    manifest = artifact_manifest(tmp_path)
    manifest = manifest.model_copy(update={"config": LotMemoryConfig(enabled=True, mode="lot_only"),
                                           "global_memory": manifest.global_memory.model_copy(update={"path": "absent"})})
    memory = FabricLotAnomalyMemory.load(manifest, base_dir=tmp_path)
    assert memory.global_memory is None and memory.effective_mode == "lot_only"


@pytest.mark.parametrize("enabled,mode", [(False, "lot_only"), (True, "global_plus_lot")])
def test_missing_global_never_auto_activates_lot_memory(enabled, mode):
    with pytest.raises(ValueError, match="Global Good memory"):
        FabricLotAnomalyMemory(fabric_id="fabric_001", lot_id="lot_001", lot_memory=make_memory(scope="lot"),
                               config=LotMemoryConfig(enabled=enabled, mode=mode))
