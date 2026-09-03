import json
from pathlib import Path

import pytest

from backend.app.services.fabric_registry import FabricRegistry, FabricRegistryError


def _registry(tmp_path: Path, *, backbone: str = "shared_v1") -> FabricRegistry:
    artifact = tmp_path / "fabric_001" / "v1"
    artifact.mkdir(parents=True)
    for name in ("reference.json", "memory_bank.npy", "memory_bank.faiss"):
        (artifact / name).write_bytes(b"ok")
    payload = {
        "schema_version": 1,
        "shared_backbone": {"backbone_version": "shared_v1"},
        "fabrics": {
            "fabric_001": {
                "display_name": "Fabric A",
                "status": "ACTIVE",
                "active_reference": "fabric_001_ref_v1",
                "references": [
                    {
                        "reference_id": "fabric_001_ref_v1",
                        "reference_version": 1,
                        "status": "ACTIVE",
                        "backbone_version": backbone,
                        "artifact_dir": str(artifact),
                        "metadata_path": str(artifact / "reference.json"),
                        "memory_bank_path": str(artifact / "memory_bank.npy"),
                        "faiss_index_path": str(artifact / "memory_bank.faiss"),
                        "thresholds": {
                            "image_review_threshold": 0.2,
                            "image_bad_threshold": 0.4,
                            "pixel_threshold": 0.3,
                            "minimum_defect_area": 4,
                        },
                    }
                ],
            }
        },
    }
    path = tmp_path / "fabric_registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return FabricRegistry(path)


def test_fabric_registry_routes_only_to_selected_fabric(tmp_path: Path) -> None:
    entry = _registry(tmp_path).get_active_reference("fabric_001")
    assert entry.fabric_id == "fabric_001"
    assert entry.reference_id == "fabric_001_ref_v1"
    assert entry.display_name == "Fabric A"


def test_fabric_registry_rejects_unknown_fabric(tmp_path: Path) -> None:
    with pytest.raises(FabricRegistryError, match="Unknown fabric_id"):
        _registry(tmp_path).get_active_reference("fabric_999")


def test_fabric_registry_rejects_backbone_mismatch(tmp_path: Path) -> None:
    with pytest.raises(FabricRegistryError, match="version mismatch"):
        _registry(tmp_path, backbone="shared_v2").get_active_reference("fabric_001")


def test_fabric_registry_rejects_inactive_reference(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = json.loads(registry.registry_path.read_text(encoding="utf-8"))
    payload["fabrics"]["fabric_001"]["status"] = "INACTIVE"
    registry.registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FabricRegistryError, match="Fabric is not active"):
        FabricRegistry(registry.registry_path).get_active_reference("fabric_001")


def test_fabric_registry_rejects_missing_artifact(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    entry = registry.get_reference("fabric_001", "fabric_001_ref_v1")
    entry.faiss_index_path.unlink()

    with pytest.raises(FabricRegistryError, match="artifact is missing"):
        registry.get_active_reference("fabric_001")


def test_selected_fabric_never_falls_back_to_another_bank(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = json.loads(registry.registry_path.read_text(encoding="utf-8"))
    source = payload["fabrics"]["fabric_001"]
    artifact = tmp_path / "fabric_002" / "v1"
    artifact.mkdir(parents=True)
    for name in ("reference.json", "memory_bank.npy", "memory_bank.faiss"):
        (artifact / name).write_bytes(b"fabric-b")
    fabric_b = json.loads(json.dumps(source))
    fabric_b["display_name"] = "Fabric B"
    fabric_b["active_reference"] = "fabric_002_ref_v1"
    reference = fabric_b["references"][0]
    reference.update(
        {
            "reference_id": "fabric_002_ref_v1",
            "artifact_dir": str(artifact),
            "metadata_path": str(artifact / "reference.json"),
            "memory_bank_path": str(artifact / "memory_bank.npy"),
            "faiss_index_path": str(artifact / "memory_bank.faiss"),
        }
    )
    payload["fabrics"]["fabric_002"] = fabric_b
    registry.registry_path.write_text(json.dumps(payload), encoding="utf-8")

    selected = FabricRegistry(registry.registry_path).get_active_reference("fabric_002")
    assert selected.reference_id == "fabric_002_ref_v1"
    assert selected.faiss_index_path == artifact / "memory_bank.faiss"
    assert "fabric_001" not in str(selected.faiss_index_path)
