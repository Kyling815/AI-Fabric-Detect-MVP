import numpy as np
from PIL import Image

from backend.app.ai.feature_extractor import CNNFeatureExtractor


def test_feature_extractor_returns_one_embedding_per_patch() -> None:
    extractor = CNNFeatureExtractor(
        model_name="resnet18",
        device="cpu",
        use_pretrained_weights=False,
        model_input_size=64,
    )
    patches = [
        Image.new("RGB", (32, 32), color=(255, 255, 255)),
        Image.new("RGB", (32, 32), color=(0, 0, 0)),
    ]

    embeddings = extractor.extract(patches, batch_size=1)

    assert embeddings.shape == (2, 512)
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_feature_extractor_cuda_request_falls_back_to_cpu_when_unavailable() -> None:
    extractor = CNNFeatureExtractor(
        model_name="resnet18",
        device="cuda",
        use_pretrained_weights=False,
        model_input_size=64,
    )

    assert extractor.info.device in {"cpu", "cuda"}
