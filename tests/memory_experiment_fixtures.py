import hashlib

import numpy as np

from backend.app.core.memory_experiment import MemoryExperimentConfig, MemoryIdentity, SourceGoodImage


def identity(*, fabric_id="fabric_a", **overrides):
    source = SourceGoodImage(image_id="synthetic_good_1", path="synthetic-fixture.png",
                             sha256=hashlib.sha256(b"synthetic fixture only").hexdigest(),
                             fabric_id=fabric_id, lot_id=None)
    return MemoryIdentity(reference_version="research_v1", encoder_version="resnet18_layer2_layer3_v1",
                          adapter_version="adapter_v4", fabric_id=fabric_id, lot_id=None,
                          threshold_profile_version=None, source_good_images=(source,), **overrides)


def config(metric="current", ratio=1.0, **overrides):
    return MemoryExperimentConfig(enabled=True, distance_metric=metric, coreset_ratio=ratio, **overrides)


def bank():
    values = np.random.default_rng(19).normal(size=(100, 8)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)
