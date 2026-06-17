import json

import numpy as np

from backend.app.ai.anomaly_detector import FaissAnomalyDetector
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.ai.patching import PatchBox


def test_faiss_anomaly_detector_scores_suspicious_patches(tmp_path) -> None:
    reference_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    index_path = tmp_path / "reference.index"
    FaissIndexStore(metric="cosine").save(reference_embeddings, index_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "reference_version": "unit_reference",
                "faiss_metric": "cosine",
                "faiss_index_path": str(index_path),
            }
        ),
        encoding="utf-8",
    )
    query_embeddings = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    boxes = [
        PatchBox(x=0, y=0, w=32, h=32),
        PatchBox(x=32, y=0, w=32, h=32),
    ]

    result = FaissAnomalyDetector(metadata_path=metadata_path, threshold=0.5).score(query_embeddings, boxes)

    assert result.status == "scored"
    assert result.reference_version == "unit_reference"
    assert len(result.patch_scores) == 2
    assert result.patch_scores[0].suspicious is False
    assert result.patch_scores[1].suspicious is True
    assert result.suspicious_patch_count == 1
    assert result.max_anomaly_score == 1.0
