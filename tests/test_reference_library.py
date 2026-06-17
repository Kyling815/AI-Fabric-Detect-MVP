import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.ai.reference_library import ReferenceBuildConfig, ReferenceLibraryBuilder


def test_reference_library_builder_saves_embeddings_and_metadata(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    image_path = normal_dir / "fabric.png"
    Image.new("RGB", (64, 64), color=(230, 230, 230)).save(image_path)

    builder = ReferenceLibraryBuilder(
        indexes_dir=tmp_path / "indexes",
        metadata_dir=tmp_path / "metadata",
    )
    artifact = builder.build(
        ReferenceBuildConfig(
            dataset_name="unit",
            fabric_type=None,
            normal_path=normal_dir,
            reference_version="unit_reference_v1",
            model_name="resnet18",
            device="cpu",
            use_pretrained_weights=False,
            image_max_size=64,
            model_input_size=64,
            patch_size=32,
            patch_stride=32,
            batch_size=2,
            max_images=1,
        ),
        image_paths=[image_path],
    )

    embeddings = np.load(artifact.embeddings_path)
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))

    assert artifact.processed_image_count == 1
    assert artifact.skipped_image_count == 0
    assert artifact.patch_count == 4
    assert artifact.faiss_index_path.exists()
    assert artifact.faiss_metric == "cosine"
    assert embeddings.shape == (4, 512)
    assert metadata["reference_version"] == "unit_reference_v1"
    assert metadata["faiss_index_path"] == str(artifact.faiss_index_path)
    assert metadata["faiss_vector_count"] == 4
    assert metadata["images"][0]["patch_count"] == 4
    assert metadata["images"][0]["patches"][0]["embedding_offset"] == 0

    index = FaissIndexStore(metric="cosine").load(artifact.faiss_index_path)
    result = FaissIndexStore(metric="cosine").search(index, embeddings[:1], k=1)
    assert result.indices[0, 0] == 0
