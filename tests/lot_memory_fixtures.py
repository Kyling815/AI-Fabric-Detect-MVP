import hashlib

import numpy as np

from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory
from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity, SourceGoodImage


def make_memory(*, scope="global", fabric_id="fabric_001", lot_id="lot_001", metric="current",
                vectors=None, encoder_version="encoder_v1", adapter_version="adapter_v4", version=None):
    source_lot = None if scope == "global" else lot_id
    source = SourceGoodImage(image_id=f"fixture_{scope}_{fabric_id}_{lot_id}", path=f"{scope}_source.png",
                             sha256=hashlib.sha256(f"{scope}-{fabric_id}-{lot_id}".encode()).hexdigest(),
                             fabric_id=fabric_id, lot_id=source_lot)
    identity = MemoryIdentity(reference_version=version or f"{scope}_v1", encoder_version=encoder_version,
                              adapter_version=adapter_version, fabric_id=fabric_id, lot_id=source_lot,
                              threshold_profile_version=None, source_good_images=(source,))
    if vectors is None:
        vectors = [[1, 0], [0, -1]] if scope == "global" else [[0.8, 0.6], [-1, 0]]
    return ExperimentalAnomalyMemory.build(np.array(vectors, dtype=np.float32), identity=identity,
                                           config=MemoryExperimentConfig(enabled=True, distance_metric=metric))
