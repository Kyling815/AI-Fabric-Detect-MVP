from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

# Windows wheels for PyTorch/torchvision and faiss-cpu can load different OpenMP
# runtimes in one process. Without this, local MVP requests that extract CNN
# embeddings and query FAISS can terminate before Python raises an exception.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    faiss = None
    FAISS_IMPORT_ERROR = exc
else:
    FAISS_IMPORT_ERROR = None


@dataclass(frozen=True)
class FaissSearchResult:
    scores: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class FaissIndexInfo:
    path: Path
    metric: str
    vector_count: int
    embedding_dim: int


class FaissIndexStore:
    def __init__(self, metric: str = "cosine") -> None:
        if faiss is None:
            raise RuntimeError("faiss-cpu is required for FAISS index operations.") from FAISS_IMPORT_ERROR
        if metric not in {"cosine", "l2"}:
            raise ValueError("Metric must be 'cosine' or 'l2'.")
        self.metric = metric

    def build(self, embeddings: np.ndarray) -> object:
        matrix = self._as_float32_matrix(embeddings)
        embedding_dim = matrix.shape[1]
        if self.metric == "cosine":
            index = faiss.IndexFlatIP(embedding_dim)
        else:
            index = faiss.IndexFlatL2(embedding_dim)
        if matrix.shape[0] > 0:
            index.add(matrix)
        return index

    def save(self, embeddings: np.ndarray, path: Path) -> FaissIndexInfo:
        index = self.build(embeddings)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        matrix = self._as_float32_matrix(embeddings)
        return FaissIndexInfo(
            path=path,
            metric=self.metric,
            vector_count=int(matrix.shape[0]),
            embedding_dim=int(matrix.shape[1]),
        )

    def load(self, path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        return faiss.read_index(str(path))

    def search(self, index: object, query_embeddings: np.ndarray, k: int = 1) -> FaissSearchResult:
        if k <= 0:
            raise ValueError("k must be positive.")
        if index.ntotal == 0:
            raise ValueError("Cannot search an empty FAISS index.")
        queries = self._as_float32_matrix(query_embeddings)
        scores, indices = index.search(queries, k)
        return FaissSearchResult(scores=scores, indices=indices)

    def save_from_npy(self, embeddings_path: Path, index_path: Path) -> FaissIndexInfo:
        embeddings = np.load(embeddings_path)
        return self.save(embeddings, index_path)

    def _as_float32_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if matrix.shape[1] <= 0:
            raise ValueError("Embeddings must have a positive embedding dimension.")
        return np.ascontiguousarray(matrix)
