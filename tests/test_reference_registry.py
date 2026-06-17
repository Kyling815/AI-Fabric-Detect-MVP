import json
from pathlib import Path

from backend.app.services.reference_registry import ReferenceRegistry


def test_reference_registry_loads_reference_by_id(tmp_path: Path) -> None:
    metadata_path = tmp_path / "reference.json"
    embeddings_path = tmp_path / "embeddings.npy"
    index_path = tmp_path / "faiss.index"
    registry_path = tmp_path / "reference_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "active_reference_version": "version_1",
                "references": [
                    {
                        "reference_id": "dme_plain_v1",
                        "reference_version": "version_1",
                        "dataset_name": "dme_20250129",
                        "fabric_type": None,
                        "source_layer": "raw_dataset",
                        "metadata_path": str(metadata_path),
                        "embeddings_path": str(embeddings_path),
                        "faiss_index_path": str(index_path),
                        "faiss_metric": "cosine",
                        "embedding_dim": 128,
                        "patch_count": 10,
                        "processed_image_count": 2,
                        "created_at": "2026-06-12T00:00:00Z",
                        "thresholds": {"anomaly_threshold": 0.075},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    entry = ReferenceRegistry(registry_path=registry_path).get("dme_plain_v1")

    assert entry.reference_id == "dme_plain_v1"
    assert entry.reference_version == "version_1"
    assert entry.embedding_dim == 128
    assert entry.thresholds["anomaly_threshold"] == 0.075
