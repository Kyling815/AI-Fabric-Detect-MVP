import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from backend.app.ai.segmentation_dataset import SegmentationDataset, load_segmentation_sample
from backend.app.ai.segmentation_model import ResNet18UNet
from backend.app.ai.segmentation_split import grouped_segmentation_split
from backend.app.ai.segmentation_trainer import (baseline_loss, build_optimizer, evaluate_segmentation,
                                                load_segmentation_checkpoint, resolve_training_device,
                                                save_segmentation_checkpoint, train_segmentation_baseline, training_reproducibility)
from backend.app.core.segmentation_data import SegmentationManifest, SegmentationTaxonomy
from backend.app.core.segmentation_training import SegmentationBaselineConfig
from scripts.check_segmentation_pipeline import create_demo_manifest
from tests.segmentation_training_fixtures import deterministic_cpu, overfit_batch, small_config


def test_optimizer_parameter_groups_are_disjoint_and_learning_rates_are_configured():
    config = small_config()
    model = ResNet18UNet(config.pipeline.taxonomy, pretrained=False, decoder_channels=config.decoder_channels)
    optimizer = build_optimizer(model, config)
    encoder, decoder = optimizer.param_groups
    assert encoder["lr"] == config.pipeline.training.encoder_lr
    assert decoder["lr"] == config.pipeline.training.decoder_lr
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))


def test_checkpoint_roundtrip_preserves_probabilities_mask_confidence_and_optimizer(tmp_path):
    config = small_config()
    model = ResNet18UNet(config.pipeline.taxonomy, pretrained=False, decoder_channels=config.decoder_channels)
    image, _ = overfit_batch()
    before = model.predict(image)
    path = tmp_path / "new_checkpoint.pt"
    save_segmentation_checkpoint(path, model, config=config, epoch=1, validation_loss=1.2,
                                 dataset_fingerprint="a" * 64, synthetic=True, optimizer=build_optimizer(model, config))
    loaded, metadata = load_segmentation_checkpoint(path, trusted_local=True, expected_taxonomy=config.pipeline.taxonomy)
    after = loaded.predict(image)
    assert torch.equal(before.probabilities, after.probabilities)
    assert torch.equal(before.mask, after.mask) and torch.equal(before.confidence, after.confidence)
    assert metadata["synthetic"] and not metadata["production_approved"] and metadata["initialization"] == "random_explicit"
    payload = torch.load(path, weights_only=True)
    assert len(payload["optimizer_state"]["param_groups"]) == 2
    assert loaded.encoder_weights_sha256 is None
    with pytest.raises(ValueError, match="trusted"):
        load_segmentation_checkpoint(path)
    with pytest.raises(ValueError, match="taxonomy"):
        load_segmentation_checkpoint(path, trusted_local=True, expected_taxonomy=SegmentationTaxonomy(classes=("background", "scratch")))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_segmentation_checkpoint(path, model, config=config, epoch=2, validation_loss=1,
                                     dataset_fingerprint="a" * 64, synthetic=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["architecture", "preprocessing", "approval", "nonfinite", "taxonomy", "normalization", "initialization", "epoch", "fingerprint", "state_shape"])
def test_corrupted_checkpoint_contracts_rejected(tmp_path, failure):
    config = small_config()
    model = ResNet18UNet(config.pipeline.taxonomy, pretrained=False, decoder_channels=config.decoder_channels)
    good = tmp_path / "good.pt"
    save_segmentation_checkpoint(good, model, config=config, epoch=1, validation_loss=1, dataset_fingerprint="a" * 64, synthetic=True)
    payload = torch.load(good, weights_only=True)
    if failure == "architecture":
        payload["metadata"]["architecture_version"] = "unknown"
    elif failure == "preprocessing":
        payload["metadata"]["preprocessing"] = "unknown"
    elif failure == "approval":
        payload["metadata"]["production_approved"] = True
    elif failure == "nonfinite":
        payload["model_state"]["head.weight"].fill_(float("nan"))
    elif failure == "taxonomy":
        payload["metadata"]["taxonomy"]["classes"] = ["background", "scratch"]
    elif failure == "normalization":
        payload["model_state"]["rgb_mean"].zero_()
    elif failure == "initialization":
        payload["metadata"]["initialization"] = "imagenet1k_v1"
    elif failure == "epoch":
        payload["metadata"]["epoch"] = True
    elif failure == "fingerprint":
        payload["metadata"]["dataset_fingerprint"] = "invalid"
    else:
        payload["model_state"]["head.weight"] = torch.zeros(3, 4, 1, 1)
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError):
        load_segmentation_checkpoint(broken, trusted_local=True)


def test_cpu_auto_fallback_and_explicit_cuda_failure():
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_training_device("auto").type == "cpu"
        assert resolve_training_device("cpu").type == "cpu"
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_training_device("cuda")


def test_evaluation_determinism_and_metric_arithmetic(tmp_path):
    manifest, config = create_demo_manifest(tmp_path), small_config()
    model = ResNet18UNet(config.pipeline.taxonomy, pretrained=False, decoder_channels=config.decoder_channels)
    dataset = SegmentationDataset(manifest, partition="validation", output_size=(32, 32), allow_synthetic=True)
    loader = DataLoader(dataset, batch_size=3)
    first = evaluate_segmentation(model, loader, baseline_loss(config), device=torch.device("cpu"))
    second = evaluate_segmentation(model, loader, baseline_loss(config), device=torch.device("cpu"))
    assert first == second and model.training
    matrix = np.asarray(first["metrics"]["confusion_matrix"])
    assert first["samples"] == 3 and matrix.sum() == 3072
    assert first["metrics"]["pixel_accuracy"] == matrix.trace() / matrix.sum()


def test_tiny_training_checkpoint_reload_and_no_test_or_locked_pixels(tmp_path):
    manifest, config = create_demo_manifest(tmp_path), small_config()
    split = grouped_segmentation_split(manifest, config.pipeline.split)
    read_ids = []
    def monitored(sample, *args, **kwargs):
        read_ids.append(sample.sample_id)
        assert split.assignments[sample.sample_id] in {"train", "validation"}
        return load_segmentation_sample(sample, *args, **kwargs)
    before = {Path(sample.image_path): Path(sample.image_path).read_bytes() for sample in manifest.samples}
    output = tmp_path / "new_candidate"
    with patch("backend.app.ai.segmentation_trainer.load_segmentation_sample", side_effect=monitored), patch("backend.app.ai.segmentation_dataset.load_segmentation_sample", side_effect=monitored):
        report = train_segmentation_baseline(manifest, config, output_dir=output, allow_synthetic=True)
    assert report["epochs_completed"] == 2 and not report["amp_enabled"]
    assert not report["test_partition_evaluated"] and not report["production_approved"] and report["trained_model"]
    assert report["status"] == "SYNTHETIC_SOFTWARE_CHECK_ONLY" and report["contains_synthetic_train_or_validation"]
    assert len(set(read_ids)) == 6
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest.model_dump(mode="json")
    assert report["model_version"] == output.name
    assert before == {path: path.read_bytes() for path in before}
    model, metadata = load_segmentation_checkpoint(output / report["best_checkpoint"], trusted_local=True)
    assert metadata["epoch"] == report["best_epoch"] and not model.training
    assert sorted(path.name for path in output.glob("*.pt"))


def test_early_stopping_keeps_initial_best_checkpoint_without_overwrite(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    config = small_config(training_updates={"epochs": 5, "early_stopping_patience": 1})
    validation = {"loss": 1.0, "samples": 3, "metrics": {"note": "mocked only for early-stop control test"}}
    with patch("backend.app.ai.segmentation_trainer.evaluate_segmentation", return_value=validation):
        report = train_segmentation_baseline(manifest, config, output_dir=tmp_path / "candidate", allow_synthetic=True)
    assert report["early_stopped"] and report["epochs_completed"] == 2 and report["best_epoch"] == 1
    assert [path.name for path in (tmp_path / "candidate").glob("*.pt")] == ["best_epoch_0001.pt"]


@pytest.mark.parametrize("failure", ["disabled", "config_synthetic_off", "caller_synthetic_off", "taxonomy", "existing_directory", "no_validation", "missing_train_class"])
def test_training_gates_fail_before_artifact_creation(tmp_path, failure):
    manifest, config = create_demo_manifest(tmp_path), small_config()
    output, allow = tmp_path / "candidate", True
    if failure == "disabled":
        config = small_config(training_updates={"enabled": False})
    elif failure == "config_synthetic_off":
        config = small_config(training_updates={"allow_synthetic": False})
    elif failure == "caller_synthetic_off":
        allow = False
    elif failure == "taxonomy":
        manifest = manifest.model_copy(update={"taxonomy": SegmentationTaxonomy(classes=("background", "scratch"))})
    elif failure == "existing_directory":
        output.mkdir()
        (output / "user_file.txt").write_text("preserve", encoding="utf-8")
    elif failure == "no_validation":
        values = config.model_dump(mode="json")
        values["pipeline"]["split"].update(train_ratio=0.8, validation_ratio=0, test_ratio=0.2)
        config = SegmentationBaselineConfig.model_validate(values)
    else:
        taxonomy = SegmentationTaxonomy(classes=("background", "defect", "scratch"))
        manifest = manifest.model_copy(update={"taxonomy": taxonomy})
        values = config.model_dump(mode="json")
        values["pipeline"]["taxonomy"] = taxonomy.model_dump(mode="json")
        config = SegmentationBaselineConfig.model_validate(values)
    with pytest.raises((ValueError, FileExistsError)):
        train_segmentation_baseline(manifest, config, output_dir=output, allow_synthetic=allow)
    if failure == "existing_directory":
        assert (output / "user_file.txt").read_text() == "preserve" and len(list(output.iterdir())) == 1
    else:
        assert not output.exists()


def test_real_unverified_annotations_are_not_accepted_as_training_ready(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    values = manifest.model_dump(mode="json")
    for sample in values["samples"]:
        sample.update(origin="real", annotation_source="existing_binary")
    manifest = SegmentationManifest.model_validate(values)
    with pytest.raises(ValueError, match="human_verified"):
        train_segmentation_baseline(manifest, small_config(), output_dir=tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_protected_storage_output_rejected_before_writes(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="protected"):
        train_segmentation_baseline(manifest, small_config(), output_dir=root / "storage/references/phase8_forbidden", allow_synthetic=True)


def test_checksum_failure_stops_preflight_without_output_directory(tmp_path):
    manifest, config = create_demo_manifest(tmp_path), small_config()
    split = grouped_segmentation_split(manifest, config.pipeline.split)
    selected = next(sample for sample in manifest.samples if split.assignments[sample.sample_id] == "train")
    Path(selected.image_path).write_bytes(b"fixture tampered after manifest creation")
    with pytest.raises(ValueError, match="checksum"):
        train_segmentation_baseline(manifest, config, output_dir=tmp_path / "candidate", allow_synthetic=True)
    assert not (tmp_path / "candidate").exists()


def test_source_only_override_cannot_weaken_trainer_roll_gate(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    values = small_config().model_dump(mode="json")
    values["pipeline"]["split"]["require_roll_id"] = False
    config = SegmentationBaselineConfig.model_validate(values)
    with pytest.raises(ValueError, match="roll-disjoint"):
        train_segmentation_baseline(manifest, config, output_dir=tmp_path / "candidate", allow_synthetic=True)
    assert not (tmp_path / "candidate").exists()


def test_rng_backend_and_thread_settings_restore_on_failure():
    before_rng, before_threads = torch.get_rng_state().clone(), torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(RuntimeError):
        with training_reproducibility(123, 1):
            torch.rand(10)
            raise RuntimeError("fixture")
    assert torch.equal(before_rng, torch.get_rng_state()) and before_threads == torch.get_num_threads()
    assert before_deterministic == torch.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("values", [{"device": "mps"}, {"decoder_channels": [4, 4]}, {"decoder_channels": [True] * 5},
                                  {"ce_weight": 0, "dice_weight": 0}, {"min_delta": -1}, {"cpu_threads": 0}])
def test_invalid_baseline_config_rejected(values):
    with pytest.raises(ValueError):
        small_config(**values)
