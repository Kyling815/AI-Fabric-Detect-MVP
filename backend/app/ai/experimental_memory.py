"""Opt-in offline FAISS memory. No QC, thresholds, registry writes or activation."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from backend.app.ai.coreset import CoresetSelector, SeededRandomCoresetSelector
from backend.app.ai.faiss_index import FaissIndexStore, faiss
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryExperimentMetadata, MemoryIdentity


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_research_output(directory: Path) -> Path:
    target = directory.resolve()
    storage = PROJECT_ROOT / "storage"
    research = storage / "experiments"
    data = PROJECT_ROOT / "Data"
    if (target in {PROJECT_ROOT, storage, research, data} or data in target.parents
            or (storage in target.parents and research not in target.parents)):
        raise ValueError("Research output must be a new directory outside Data/production storage (use storage/experiments/<name>)")
    return target


def _real_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix)
    if (values.ndim != 2 or min(values.shape) <= 0 or np.iscomplexobj(values)
            or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()):
        raise ValueError("Embeddings must be a finite nonempty real 2D matrix")
    if np.max(np.abs(values)) > np.finfo(np.float32).max:
        raise ValueError("Embeddings exceed float32 range")
    return np.array(values, dtype=np.float32, order="C", copy=True)


def _vectors(matrix: np.ndarray, policy: str) -> np.ndarray:
    values = _real_matrix(matrix)
    if policy == "raw":
        # Squared distances must also fit float32 FAISS scores.
        if np.max(np.abs(values)) > np.sqrt(np.finfo(np.float32).max / (4 * values.shape[1])):
            raise ValueError("L2 vectors risk overflow in squared distance")
        return values
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Cosine is undefined for zero/near-zero vectors")
    if policy == "preserve_unit":
        if not np.allclose(norms, 1, atol=1e-5, rtol=0):
            raise ValueError("Current metric requires already L2-normalized embeddings")
        return values  # Do not renormalize the historical baseline.
    return np.ascontiguousarray(values.astype(np.float64) / norms[:, None], dtype=np.float32)


@dataclass(frozen=True)
class MemorySearchResult:
    faiss_scores: np.ndarray
    anomaly_distances: np.ndarray
    indices: np.ndarray
    source_indices: np.ndarray
    distance_metric: str
    score_definition: str


class ExperimentalAnomalyMemory:
    def __init__(self, *, identity: MemoryIdentity, config: MemoryExperimentConfig,
                 bank: np.ndarray, selected_indices: np.ndarray, full_count: int,
                 source_embeddings_sha256: str):
        if not config.enabled:
            raise ValueError("Memory experiment is disabled; explicit opt-in required")
        self.identity = identity
        self.config = config
        self.metric = "cosine" if config.distance_metric == "current" else config.distance_metric
        self.policy = {"current": "preserve_unit", "cosine": "normalize", "l2": "raw"}[config.distance_metric]
        self.score_definition = "one_minus_cosine" if self.metric == "cosine" else "squared_l2"
        self.bank = _real_matrix(bank)
        self.selected_indices = np.asarray(selected_indices, dtype=np.int64).copy()
        self.full_count = full_count
        self.source_embeddings_sha256 = source_embeddings_sha256
        self.store = FaissIndexStore(self.metric)
        self.index = self.store.build(self.bank)
        self.bank.flags.writeable = False
        self.selected_indices.flags.writeable = False

    @classmethod
    def build(cls, embeddings: np.ndarray, *, identity: MemoryIdentity, config: MemoryExperimentConfig,
              selector: CoresetSelector | None = None) -> "ExperimentalAnomalyMemory":
        if not config.enabled:
            raise ValueError("Memory experiment is disabled; explicit opt-in required")
        original = _real_matrix(embeddings)
        policy = {"current": "preserve_unit", "cosine": "normalize", "l2": "raw"}[config.distance_metric]
        prepared = _vectors(original, policy)
        selected = (selector if selector is not None else SeededRandomCoresetSelector()).select(
            prepared, ratio=config.coreset_ratio, seed=config.random_seed,
        )
        # Never serialize misleading provenance for an injected selector.
        if selected.selector_name != config.selector or selected.seed != config.random_seed:
            raise ValueError("Selector identity/seed disagrees with experimental configuration")
        expected = SeededRandomCoresetSelector().select(prepared, ratio=config.coreset_ratio, seed=config.random_seed)
        if (not np.array_equal(selected.indices, expected.indices)
                or not np.array_equal(selected.embeddings, expected.embeddings)):
            raise ValueError("Selector result is not reproducible under the declared method")
        return cls(identity=identity, config=config, bank=selected.embeddings,
                   selected_indices=selected.indices, full_count=original.shape[0],
                   source_embeddings_sha256=hashlib.sha256(original.tobytes()).hexdigest())

    def search(self, query_embeddings: np.ndarray, *, k: int = 1) -> MemorySearchResult:
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self.index.ntotal:
            raise ValueError("k must be an integer between 1 and memory size")
        queries = _vectors(query_embeddings, self.policy)
        if queries.shape[1] != self.bank.shape[1]:
            raise ValueError("Query embedding dimension does not match memory")
        nearest = self.store.search(self.index, queries, k=k)
        distances = (np.maximum(0.0, 1.0 - nearest.scores) if self.metric == "cosine"
                     else np.maximum(0.0, nearest.scores)).astype(np.float32)
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite FAISS distances")
        return MemorySearchResult(nearest.scores, distances, nearest.indices,
                                  self.selected_indices[nearest.indices], self.metric, self.score_definition)

    def save_new(self, directory: Path) -> MemoryExperimentMetadata:
        target = validate_research_output(directory)
        # Exclusive directory allocation: never overwrite even an empty existing directory.
        target.mkdir(parents=True, exist_ok=False)
        bank_path, index_path, selected_path = target / "memory_bank.npy", target / "memory_bank.faiss", target / "selected_indices.npy"
        np.save(bank_path, self.bank, allow_pickle=False)
        np.save(selected_path, self.selected_indices, allow_pickle=False)
        self.store.save(self.bank, index_path)
        metadata = MemoryExperimentMetadata(
            **self.identity.model_dump(), distance_metric=self.metric, requested_metric=self.config.distance_metric,
            score_definition=self.score_definition, vector_policy=self.policy,
            coreset_ratio=self.config.coreset_ratio, actual_coreset_ratio=len(self.bank) / self.full_count,
            selector=self.config.selector, random_seed=self.config.random_seed,
            full_feature_count=self.full_count, memory_bank_count=len(self.bank), feature_dim=self.bank.shape[1],
            source_embeddings_sha256=self.source_embeddings_sha256,
            memory_bank_sha256=file_sha256(bank_path), faiss_index_sha256=file_sha256(index_path),
            selected_indices_sha256=file_sha256(selected_path),
        )
        # Manifest last: incomplete artifacts remain unloadable, never registered or activated.
        with (target / "reference.json").open("x", encoding="utf-8") as stream:
            stream.write(metadata.model_dump_json(indent=2) + "\n")
        return metadata

    @classmethod
    def load(cls, directory: Path, *, expected_identity: MemoryIdentity,
             config: MemoryExperimentConfig) -> "ExperimentalAnomalyMemory":
        if not config.enabled:
            raise ValueError("Memory experiment is disabled; explicit opt-in required")
        metadata = MemoryExperimentMetadata.model_validate_json((directory / "reference.json").read_text(encoding="utf-8"))
        actual_identity = MemoryIdentity.model_validate({key: getattr(metadata, key) for key in MemoryIdentity.model_fields})
        if actual_identity != expected_identity:
            raise ValueError("Memory provenance/version/fabric does not match requested identity")
        expected_config = {
            "requested_metric": config.distance_metric, "coreset_ratio": config.coreset_ratio,
            "selector": config.selector, "random_seed": config.random_seed,
        }
        if any(getattr(metadata, key) != value for key, value in expected_config.items()):
            raise ValueError("Memory configuration does not match requested experiment")
        paths = {
            "memory_bank": directory / metadata.memory_bank_path,
            "faiss_index": directory / metadata.faiss_index_path,
            "selected_indices": directory / metadata.selected_indices_path,
        }
        for name, path in paths.items():
            if file_sha256(path) != getattr(metadata, name + "_sha256"):
                raise ValueError(f"{name} artifact checksum mismatch")
        bank = _real_matrix(np.load(paths["memory_bank"], allow_pickle=False))
        if bank.shape != (metadata.memory_bank_count, metadata.feature_dim):
            raise ValueError("Memory bank shape disagrees with metadata")
        selected = np.load(paths["selected_indices"], allow_pickle=False)
        if (selected.dtype.kind not in "iu" or selected.shape != (len(bank),)
                or np.any(selected < 0) or np.any(selected >= metadata.full_feature_count)
                or np.any(np.diff(selected.astype(np.int64)) <= 0)):
            raise ValueError("Invalid source selection indices")
        expected_indices = (np.arange(metadata.full_feature_count, dtype=np.int64) if config.coreset_ratio == 1
                            else np.sort(np.random.default_rng(config.random_seed).choice(
                                metadata.full_feature_count, len(bank), replace=False)))
        if not np.array_equal(selected, expected_indices):
            raise ValueError("Selection indices disagree with declared seed/ratio")
        if metadata.distance_metric == "cosine":
            _vectors(bank, "preserve_unit")  # Validate saved unit vectors; do not renormalize on reload.
        else:
            _vectors(bank, "raw")
        store = FaissIndexStore(metadata.distance_metric)
        index = store.load(paths["faiss_index"])
        metric_type = faiss.METRIC_INNER_PRODUCT if metadata.distance_metric == "cosine" else faiss.METRIC_L2
        if (index.metric_type != metric_type or index.d != bank.shape[1] or index.ntotal != len(bank)
                or not isinstance(index, (faiss.IndexFlatIP, faiss.IndexFlatL2))):
            raise ValueError("FAISS metric/type/dimension/count incompatible with metadata")
        for offset in range(len(bank)):
            if not np.array_equal(index.reconstruct(offset), bank[offset]):
                raise ValueError("FAISS stored vector disagrees with memory bank")
        result = cls(identity=expected_identity, config=config, bank=bank, selected_indices=selected,
                     full_count=metadata.full_feature_count, source_embeddings_sha256=metadata.source_embeddings_sha256)
        result.index = index
        return result
