"""Temporary independent random CNN/checkpoint/ONNX; no training or activation."""

import hashlib
from pathlib import Path

import pytest

from backend.app.ai.segmentation_model import ResNet18UNet
from backend.app.ai.segmentation_onnx import OnnxSegmentationRunner, export_segmentation_onnx
from backend.app.ai.segmentation_trainer import save_segmentation_checkpoint, training_reproducibility
from backend.app.core.segmentation_onnx import OnnxExportConfig
from backend.app.core.segmentation_training import SegmentationBaselineConfig
from tests.segmentation_training_fixtures import small_config


@pytest.fixture(scope="module")
def onnx_bundle(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    root = tmp_path_factory.mktemp("phase16_onnx")
    source = root / "synthetic_onnx_fixture_v1"
    source.mkdir()
    checkpoint = source / "checkpoint.pt"
    values = small_config().model_dump(mode="json")
    values["pipeline"]["taxonomy"]["classes"] = ["background", "scratch", "stain"]
    baseline = SegmentationBaselineConfig.model_validate(values)
    with training_reproducibility(16, 1):
        model = ResNet18UNet(baseline.pipeline.taxonomy, pretrained=False, decoder_channels=baseline.decoder_channels)
        save_segmentation_checkpoint(checkpoint, model, config=baseline, epoch=1, validation_loss=1,
                                     dataset_fingerprint="1" * 64, synthetic=True)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = OnnxExportConfig(input_shape=(1, 3, 35, 49))
    output = root / "export_v1"
    arguments = dict(expected_sha256=checkpoint_hash, expected_model_version=source.name,
                     trusted_local=True, allow_synthetic=True, config=config)
    result = export_segmentation_onnx(checkpoint, output_dir=output, **arguments)
    return {"root": root, "source": checkpoint, "output": output, "config": config,
            "arguments": arguments, "result": result, "baseline": baseline}


def runner_for(bundle):
    return OnnxSegmentationRunner(bundle["output"], expected_model_sha256=bundle["result"]["model_sha256"],
        expected_manifest_sha256=bundle["result"]["manifest_sha256"], trusted_local=True)
