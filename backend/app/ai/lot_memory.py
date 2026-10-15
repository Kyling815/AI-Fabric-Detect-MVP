"""Read-only fabric/lot memory union for experiments, never production QC."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory, _vectors
from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.core.lot_memory import FabricLotMemoryManifest, LotMemoryConfig, validate_memory_identifier


class MissingLotMemoryError(ValueError):
    pass


@dataclass(frozen=True)
class FabricLotSearchResult:
    anomaly_distances: np.ndarray
    faiss_scores: np.ndarray
    memory_scopes: np.ndarray
    indices: np.ndarray  # Local bank index; pair with memory_scopes, not a global offset.
    source_indices: np.ndarray
    reference_versions: np.ndarray
    distance_metric: str
    score_definition: str
    effective_mode: str
    fallback_reason: str | None


class FabricLotAnomalyMemory:
    def __init__(self, *, fabric_id: str, config: LotMemoryConfig,
                 global_memory: ExperimentalAnomalyMemory | None = None,
                 lot_memory: ExperimentalAnomalyMemory | None = None, lot_id: str | None = None):
        self.fabric_id = validate_memory_identifier(fabric_id, "fabric_id")
        self.config = config
        self.lot_id = lot_id
        self.global_memory = None
        self.lot_memory = None
        self.fallback_reason = None
        mode = config.mode if config.enabled else "global_only"
        if mode in {"global_only", "global_plus_lot"}:
            if global_memory is None:
                raise ValueError("Global Good memory is required")
            self._validate_scope(global_memory, lot_id=None)
            self.global_memory = global_memory
        if mode in {"lot_only", "global_plus_lot"}:
            validate_memory_identifier(lot_id, "lot_id")
            if lot_memory is None:
                if mode == "global_plus_lot" and config.missing_lot_policy == "global_only":
                    mode = "global_only"
                    self.fallback_reason = "missing_lot_memory"
                else:
                    raise MissingLotMemoryError("Selected lot has no approved research Good memory")
            else:
                self._validate_scope(lot_memory, lot_id=lot_id)
                self.lot_memory = lot_memory
        self.effective_mode = mode
        memories = [(scope, memory) for scope, memory in (("global", self.global_memory), ("lot", self.lot_memory))
                    if memory is not None]
        first = memories[0][1]
        for _, memory in memories[1:]:
            for name in ("encoder_version", "adapter_version"):
                if getattr(memory.identity, name) != getattr(first.identity, name):
                    raise ValueError(f"Global/lot {name} mismatch")
            if (memory.metric, memory.policy, memory.bank.shape[1]) != (first.metric, first.policy, first.bank.shape[1]):
                raise ValueError("Global/lot metric, vector policy or feature dimension mismatch")
        self.metric, self.policy, self.score_definition = first.metric, first.policy, first.score_definition
        self.store = FaissIndexStore(self.metric)
        # Reuse the exact index for one memory. Union concatenates stored vectors;
        # no new coreset selection, re-normalization, weighting or score fusion.
        self.index = (first.index if len(memories) == 1
                      else self.store.build(np.concatenate([memory.bank for _, memory in memories], axis=0)))
        self._scopes = np.concatenate([np.full(len(memory.bank), scope) for scope, memory in memories])
        self._local_indices = np.concatenate([np.arange(len(memory.bank)) for _, memory in memories])
        self._source_indices = np.concatenate([memory.selected_indices for _, memory in memories])
        self._versions = np.concatenate([np.full(len(memory.bank), memory.identity.reference_version) for _, memory in memories])

    def _validate_scope(self, memory: ExperimentalAnomalyMemory, *, lot_id: str | None) -> None:
        if not memory.config.enabled:
            raise ValueError("Child memory experiment is not enabled")
        if memory.identity.fabric_id != self.fabric_id:
            raise ValueError("Unrelated fabric memory must never be mixed")
        if memory.identity.lot_id != lot_id:
            raise ValueError("Memory lot scope does not match requested global/lot identity")

    @classmethod
    def load(cls, manifest: FabricLotMemoryManifest, *, base_dir: Path = Path(".")) -> "FabricLotAnomalyMemory":
        mode = manifest.config.mode if manifest.config.enabled else "global_only"
        global_memory, lot_memory = None, None

        def load_spec(spec, *, expected_lot):
            if spec.identity.fabric_id != manifest.fabric_id or spec.identity.lot_id != expected_lot:
                raise ValueError("Artifact identity disagrees with requested fabric/lot scope")
            path = Path(spec.path)
            path = path if path.is_absolute() else base_dir / path
            return ExperimentalAnomalyMemory.load(path, expected_identity=spec.identity, config=spec.config)

        if mode in {"global_only", "global_plus_lot"}:
            if manifest.global_memory is None:
                raise ValueError("Global memory artifact is required")
            global_memory = load_spec(manifest.global_memory, expected_lot=None)
        if mode in {"lot_only", "global_plus_lot"}:
            validate_memory_identifier(manifest.lot_id, "lot_id")
            spec = manifest.lot_memory
            if spec is not None:
                # Validate identity before considering a missing-path fallback.
                if spec.identity.fabric_id != manifest.fabric_id or spec.identity.lot_id != manifest.lot_id:
                    raise ValueError("Artifact identity disagrees with requested fabric/lot scope")
                path = Path(spec.path)
                path = path if path.is_absolute() else base_dir / path
                if path.exists():
                    # Corrupt/incomplete/version-mismatched artifacts never fall back silently.
                    lot_memory = load_spec(spec, expected_lot=manifest.lot_id)
        return cls(fabric_id=manifest.fabric_id, lot_id=manifest.lot_id, config=manifest.config,
                   global_memory=global_memory, lot_memory=lot_memory)

    def search(self, query_embeddings: np.ndarray, *, k: int = 1) -> FabricLotSearchResult:
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= self.index.ntotal:
            raise ValueError("k must be between 1 and combined memory count")
        query = _vectors(query_embeddings, self.policy)
        if query.shape[1] != self.index.d:
            raise ValueError("Query dimension does not match fabric/lot memory")
        nearest = self.store.search(self.index, query, k=k)
        distances = (np.maximum(0, 1 - nearest.scores) if self.metric == "cosine"
                     else np.maximum(0, nearest.scores)).astype(np.float32)
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite union distances")
        rows = nearest.indices
        return FabricLotSearchResult(distances, nearest.scores, self._scopes[rows], self._local_indices[rows],
                                     self._source_indices[rows], self._versions[rows], self.metric,
                                     self.score_definition, self.effective_mode, self.fallback_reason)
