"""Seeded random subset baseline, not a geometric k-center approximation."""

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CoresetSelection:
    embeddings: np.ndarray
    indices: np.ndarray
    requested_ratio: float
    actual_ratio: float
    selector_name: str
    seed: int


@runtime_checkable
class CoresetSelector(Protocol):
    def select(self, matrix: np.ndarray, *, ratio: float, seed: int) -> CoresetSelection:
        ...


class SeededRandomCoresetSelector:
    def select(self, matrix: np.ndarray, *, ratio: float, seed: int) -> CoresetSelection:
        values = np.asarray(matrix)
        if (values.ndim != 2 or min(values.shape) <= 0 or np.iscomplexobj(values)
                or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()):
            raise ValueError("Coreset requires a finite nonempty real matrix")
        if isinstance(ratio, bool) or not np.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("Coreset ratio must be in (0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer in [0, 2**32-1]")
        count = max(1, math.ceil(values.shape[0] * ratio))
        indices = (np.arange(values.shape[0], dtype=np.int64) if count == values.shape[0]
                   else np.sort(np.random.default_rng(seed).choice(values.shape[0], count, replace=False)))
        return CoresetSelection(values[indices].copy(), indices, float(ratio), count / values.shape[0],
                                "seeded_random", int(seed))
