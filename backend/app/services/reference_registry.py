import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


@dataclass(frozen=True)
class ReferenceRegistryEntry:
    reference_id: str
    reference_version: str
    dataset_name: str
    fabric_type: str | None
    source_layer: str
    metadata_path: Path
    embeddings_path: Path
    faiss_index_path: Path
    faiss_metric: str
    embedding_dim: int
    patch_count: int
    processed_image_count: int
    created_at: str
    thresholds: dict[str, Any]
    preprocessing_version: str = "v1"
    configuration_hash: str | None = None


class ReferenceRegistry:
    def __init__(self, registry_path: Path | None = None) -> None:
        settings = get_settings()
        self.registry_path = registry_path or resolve_project_path(settings.reference_registry_path)
        self._raw = self._load()

    @property
    def active_reference_version(self) -> str | None:
        return self._raw.get("active_reference_version")

    def list_entries(self) -> list[ReferenceRegistryEntry]:
        return [self._entry_from_record(record) for record in self._raw.get("references", [])]

    def list_latest_entries(self) -> list[ReferenceRegistryEntry]:
        """Return only the newest artifact for each stable reference id."""
        latest_by_id: dict[str, ReferenceRegistryEntry] = {}
        for entry in self.list_entries():
            latest_by_id[entry.reference_id] = entry
        return list(latest_by_id.values())

    def get(self, reference_id: str | None = None) -> ReferenceRegistryEntry:
        selected = reference_id or self.default_reference_id()
        # Registry entries are chronological; a stable reference_id should
        # resolve to its most recently rebuilt version.
        for entry in reversed(self.list_entries()):
            if selected in {entry.reference_id, entry.reference_version}:
                return entry
        raise KeyError(f"Reference id not found in registry: {selected}")

    def default_reference_id(self) -> str:
        settings = get_settings()
        if settings.default_reference_id:
            configured = settings.default_reference_id
            if any(
                configured in {entry.reference_id, entry.reference_version}
                for entry in self.list_entries()
            ):
                return configured
        active = self._raw.get("active_reference_version")
        if active:
            return str(active)
        references = self._raw.get("references", [])
        if references:
            return str(references[-1].get("reference_id") or references[-1]["reference_version"])
        raise KeyError("Reference registry contains no references.")

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"schema_version": 1, "references": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _entry_from_record(self, record: dict[str, Any]) -> ReferenceRegistryEntry:
        reference_version = str(record["reference_version"])
        return ReferenceRegistryEntry(
            reference_id=str(record.get("reference_id") or reference_version),
            reference_version=reference_version,
            dataset_name=str(record.get("dataset_name", "")),
            fabric_type=record.get("fabric_type"),
            source_layer=str(record.get("source_layer", "unknown")),
            metadata_path=_resolve_stored_path(record["metadata_path"]),
            embeddings_path=_resolve_stored_path(record["embeddings_path"]),
            faiss_index_path=_resolve_stored_path(record["faiss_index_path"]),
            faiss_metric=str(record.get("faiss_metric", "cosine")),
            embedding_dim=int(record.get("embedding_dim", 0)),
            patch_count=int(record.get("patch_count", 0)),
            processed_image_count=int(record.get("processed_image_count", 0)),
            created_at=str(record.get("created_at", "")),
            thresholds=dict(record.get("thresholds") or {}),
            preprocessing_version=str(record.get("preprocessing_version", "v1")),
            configuration_hash=record.get("configuration_hash"),
        )


def _resolve_stored_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else resolve_project_path(value)
