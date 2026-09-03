from backend.app.ai.dense_feature_extractor import (
    is_adapted_backbone_version,
    is_supported_backbone_version,
)


def test_versioned_company_adapters_are_supported() -> None:
    assert is_adapted_backbone_version(
        "shared_resnet18_imagenet1k_dense_adapter_company_v1"
    )
    assert is_adapted_backbone_version(
        "shared_resnet18_imagenet1k_dense_adapter_company_v4"
    )
    assert is_supported_backbone_version(
        "shared_resnet18_imagenet1k_dense_adapter_company_v4"
    )


def test_malformed_or_unrelated_backbone_versions_are_rejected() -> None:
    assert not is_adapted_backbone_version(
        "shared_resnet18_imagenet1k_dense_adapter_company_v"
    )
    assert not is_adapted_backbone_version(
        "shared_resnet18_imagenet1k_dense_adapter_company_v0"
    )
    assert not is_supported_backbone_version("arbitrary_backbone_v4")
