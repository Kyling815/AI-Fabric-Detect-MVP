import numpy as np

from backend.app.ai.faiss_index import FaissIndexStore


def test_faiss_index_save_load_and_search(tmp_path) -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    index_path = tmp_path / "reference.index"
    store = FaissIndexStore(metric="cosine")

    info = store.save(embeddings, index_path)
    index = store.load(index_path)
    result = store.search(index, np.array([[0.0, 1.0, 0.0]], dtype=np.float32), k=2)

    assert info.vector_count == 3
    assert info.embedding_dim == 3
    assert info.metric == "cosine"
    assert index_path.exists()
    assert result.indices.shape == (1, 2)
    assert result.indices[0, 0] == 1
    assert result.scores[0, 0] == 1.0


def test_faiss_index_save_from_npy(tmp_path) -> None:
    embeddings = np.eye(4, dtype=np.float32)
    embeddings_path = tmp_path / "embeddings.npy"
    index_path = tmp_path / "reference.index"
    np.save(embeddings_path, embeddings)

    info = FaissIndexStore(metric="cosine").save_from_npy(embeddings_path, index_path)

    assert info.vector_count == 4
    assert index_path.exists()
