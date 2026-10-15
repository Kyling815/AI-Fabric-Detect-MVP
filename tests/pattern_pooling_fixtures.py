import hashlib

from backend.app.ai.pattern_period import PatternPeriodResult
from backend.app.ai.pattern_pooling import PatternFeatureResult, PoolingPlan
from backend.app.core.memory_experiment import MemoryIdentity, SourceGoodImage
from backend.app.core.pattern import PatternEstimatorConfig
from backend.app.core.pattern_pooling import FeatureGeometry, PatternPoolingConfig
from tests.phase1_fixtures import dense_features


def periodic(px=24, py=24):
    return PatternPeriodResult(px, py, 0.99, True, 0.99, 0.99)


def feature_result(enabled=False, mode="period_aware"):
    config = PatternPoolingConfig(enabled=enabled, mode=mode)
    geometry = FeatureGeometry(algorithm="pattern_pooling_mean_v1" if enabled else "dense_baseline_v1",
                               backbone_version="synthetic_encoder_v1", architecture="synthetic_dense", feature_dim=2,
                               model_weights_sha256="0" * 64, adapter_weights_sha256=None, adapter_version=None,
                               use_pretrained_weights=False, image_max_size=768,
                               pooling=config if enabled else None,
                               period_estimator=PatternEstimatorConfig() if enabled and mode == "period_aware" else None)
    features = dense_features()
    return PatternFeatureResult(features, geometry, PoolingPlan("baseline" if not enabled else "period_aware",
                                                               ((3, 3),), ((3, 3),), None), None)


def identity_for(value):
    source = SourceGoodImage(image_id="synthetic_good", path="synthetic_good.png",
                             sha256=hashlib.sha256(b"synthetic good fixture only").hexdigest(),
                             fabric_id="fabric_001", lot_id=None)
    return MemoryIdentity(reference_version="pattern_research_v1", encoder_version=value.geometry.encoder_version,
                          adapter_version=value.geometry.adapter_version, fabric_id="fabric_001", lot_id=None,
                          threshold_profile_version=None, source_good_images=(source,))
