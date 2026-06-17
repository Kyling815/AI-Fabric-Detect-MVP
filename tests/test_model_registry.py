from backend.app.ai.model_registry import get_model_spec, pretrained_weight_filename


def test_model_registry_reports_supported_model_metadata() -> None:
    spec = get_model_spec("resnet18")

    assert spec.name == "resnet18"
    assert spec.embedding_dim == 512
    assert spec.weights_url.startswith("https://")
    assert pretrained_weight_filename("resnet18").endswith(".pth")


def test_model_registry_accepts_dash_alias() -> None:
    spec = get_model_spec("mobilenet-v3-small")

    assert spec.name == "mobilenet_v3_small"
    assert spec.embedding_dim == 576
