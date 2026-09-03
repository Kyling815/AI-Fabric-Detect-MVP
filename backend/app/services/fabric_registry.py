import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


class FabricRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class FabricReferenceEntry:
    fabric_id: str
    display_name: str
    reference_id: str
    reference_version: int
    status: str
    backbone_version: str
    artifact_dir: Path
    metadata_path: Path
    memory_bank_path: Path
    faiss_index_path: Path
    thresholds: dict[str, Any]


class FabricRegistry:
    def __init__(self, registry_path: Path | None = None) -> None:
        settings = get_settings()
        self.registry_path = registry_path or resolve_project_path(settings.fabric_registry_path)
        self._raw = self._load()

    def list_fabrics(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        fabrics = []
        for fabric_id, record in sorted((self._raw.get("fabrics") or {}).items()):
            if active_only and str(record.get("status", "")).upper() != "ACTIVE":
                continue
            fabrics.append({"fabric_id": fabric_id, **record})
        return fabrics

    def get_fabric(self, fabric_id: str) -> dict[str, Any]:
        record = (self._raw.get("fabrics") or {}).get(fabric_id)
        if record is None:
            raise FabricRegistryError(f"Unknown fabric_id: {fabric_id}")
        return {"fabric_id": fabric_id, **record}

    def get_active_reference(self, fabric_id: str) -> FabricReferenceEntry:
        fabric = self.get_fabric(fabric_id)
        if str(fabric.get("status", "")).upper() != "ACTIVE":
            raise FabricRegistryError(f"Fabric is not active: {fabric_id}")
        active_reference = fabric.get("active_reference")
        if not active_reference:
            raise FabricRegistryError(f"Fabric has no active reference: {fabric_id}")
        references = fabric.get("references") or []
        record = next((item for item in references if item.get("reference_id") == active_reference), None)
        if record is None:
            raise FabricRegistryError(
                f"Active reference {active_reference!r} is not registered for {fabric_id}"
            )
        entry = self._entry(fabric_id, fabric, record)
        if str(entry.status).upper() != "ACTIVE":
            raise FabricRegistryError(f"Reference is not active: {entry.reference_id}")
        missing = [
            path
            for path in (entry.metadata_path, entry.memory_bank_path, entry.faiss_index_path)
            if not path.exists()
        ]
        if missing:
            raise FabricRegistryError(
                "Active reference artifact is missing: " + ", ".join(str(path) for path in missing)
            )
        expected_backbone = str(self._raw.get("shared_backbone", {}).get("backbone_version", ""))
        if expected_backbone and entry.backbone_version != expected_backbone:
            raise FabricRegistryError(
                "Reference/backbone version mismatch: "
                f"reference={entry.backbone_version}, runtime={expected_backbone}"
            )
        required_thresholds = {
            "image_review_threshold",
            "image_bad_threshold",
            "pixel_threshold",
            "minimum_defect_area",
        }
        if not required_thresholds.issubset(entry.thresholds):
            missing_names = sorted(required_thresholds - set(entry.thresholds))
            raise FabricRegistryError(
                f"Reference thresholds are incomplete for {entry.reference_id}: {missing_names}"
            )
        return entry

    def get_reference(self, fabric_id: str, reference_id: str) -> FabricReferenceEntry:
        fabric = self.get_fabric(fabric_id)
        record = next(
            (item for item in fabric.get("references") or [] if item.get("reference_id") == reference_id),
            None,
        )
        if record is None:
            raise FabricRegistryError(f"Reference {reference_id!r} is not registered for {fabric_id}")
        return self._entry(fabric_id, fabric, record)

    @property
    def shared_backbone(self) -> dict[str, Any]:
        return dict(self._raw.get("shared_backbone") or {})

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            raise FabricRegistryError(f"Fabric registry not found: {self.registry_path}")
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FabricRegistryError(f"Fabric registry is invalid: {exc}") from exc
        if not isinstance(payload, dict):
            raise FabricRegistryError("Fabric registry must be a JSON object.")
        return payload

    def _entry(
        self,
        fabric_id: str,
        fabric: dict[str, Any],
        record: dict[str, Any],
    ) -> FabricReferenceEntry:
        return FabricReferenceEntry(
            fabric_id=fabric_id,
            display_name=str(fabric.get("display_name") or fabric_id),
            reference_id=str(record["reference_id"]),
            reference_version=int(record["reference_version"]),
            status=str(record.get("status", "CANDIDATE")),
            backbone_version=str(record["backbone_version"]),
            artifact_dir=_stored_path(record["artifact_dir"]),
            metadata_path=_stored_path(record["metadata_path"]),
            memory_bank_path=_stored_path(record["memory_bank_path"]),
            faiss_index_path=_stored_path(record["faiss_index_path"]),
            thresholds=dict(record.get("thresholds") or {}),
        )


def _stored_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else resolve_project_path(value)
