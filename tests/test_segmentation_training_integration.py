import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from backend.app.ai.segmentation_model import ResNet18UNet
from backend.app.ai.segmentation_trainer import train_segmentation_baseline
from backend.app.core.segmentation_training import SegmentationBaselineConfig
from scripts.check_segmentation_pipeline import create_demo_manifest
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime
from tests.segmentation_training_fixtures import deterministic_cpu, small_config


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_production_api_never_calls_supervised_model_or_training(phase1_runtime, case):
    phase1_runtime.use_case(case)
    with patch.object(ResNet18UNet, "forward", side_effect=AssertionError("Phase 9 not authorized")), patch("backend.app.ai.segmentation_trainer.train_segmentation_baseline", side_effect=AssertionError("No API training")):
        response = phase1_runtime.upload()
    snapshot = baseline()
    assert response.status_code == 201
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]


def test_cli_demo_trains_independent_pretrained_model_with_temporary_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/train_segmentation_baseline.py"), "--demo"],
                            cwd=tmp_path, capture_output=True, text=True, check=True, timeout=60)
    report = json.loads(result.stdout)
    assert report["trained_model"] and report["contains_synthetic_train_or_validation"]
    assert report["status"] == "SYNTHETIC_SOFTWARE_CHECK_ONLY" and report["encoder_initialization"] == "imagenet1k_v1"
    assert report["device"] == "cpu" and not report["amp_enabled"] and not report["artifacts_persisted"]
    assert not report["production_approved"] and not report["production_accuracy_evaluated"] and not report["test_partition_evaluated"]
    assert report["prediction_contract"]["probability_shape"] == [1, 2, 32, 32]
    assert report["prediction_contract"]["deterministic_evaluation"]
    assert list(tmp_path.iterdir()) == []


def test_cli_default_disabled_and_explicit_synthetic_training_candidate(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = create_demo_manifest(tmp_path)
    manifest_path, config_path = tmp_path / "manifest.json", tmp_path / "training.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    command = [sys.executable, str(root / "scripts/train_segmentation_baseline.py"), "--manifest", str(manifest_path), "--output-dir", str(tmp_path / "candidate")]
    failed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert failed.returncode != 0 and "Training is disabled" in failed.stderr
    assert not (tmp_path / "candidate").exists()
    config_path.write_text(small_config(training_updates={"epochs": 1}).model_dump_json(), encoding="utf-8")
    failed = subprocess.run([*command, "--config", str(config_path)], capture_output=True, text=True, timeout=60)
    assert failed.returncode != 0 and "BOTH" in failed.stderr and not (tmp_path / "candidate").exists()
    success = subprocess.run([*command, "--config", str(config_path), "--allow-synthetic"], capture_output=True, text=True, check=True, timeout=60)
    report = json.loads(success.stdout)
    assert report["epochs_completed"] == 1 and report["encoder_initialization"] == "random_explicit"
    assert not report["production_approved"]
    before = (tmp_path / "candidate/report.json").read_bytes()
    repeated = subprocess.run([*command, "--config", str(config_path), "--allow-synthetic"], capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0 and "refusing to overwrite" in repeated.stderr
    assert (tmp_path / "candidate/report.json").read_bytes() == before


def test_training_reproducibility_between_two_new_version_directories(tmp_path):
    manifest, config = create_demo_manifest(tmp_path), small_config(training_updates={"epochs": 1})
    first = train_segmentation_baseline(manifest, config, output_dir=tmp_path / "v1", allow_synthetic=True)
    second = train_segmentation_baseline(manifest, config, output_dir=tmp_path / "v2", allow_synthetic=True)
    assert first["history"] == second["history"] and first["validation"] == second["validation"]
    for directory in ("v1", "v2"):
        payload = torch.load(tmp_path / directory / first["best_checkpoint"], weights_only=True)
        if directory == "v1":
            state = payload["model_state"]
        else:
            assert all(torch.equal(value, payload["model_state"][key]) for key, value in state.items())


def test_focal_training_variant_is_explicit_and_finite(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    config = small_config(training_updates={"epochs": 1, "focal_optional": True})
    report = train_segmentation_baseline(manifest, config, output_dir=tmp_path / "focal_v1", allow_synthetic=True)
    assert report["validation"]["loss"] > 0
    assert json.loads((tmp_path / "focal_v1/config.json").read_text())["pipeline"]["training"]["focal_optional"]


def test_nonfinite_training_fails_without_checkpoint_or_approval(tmp_path):
    manifest, config = create_demo_manifest(tmp_path), small_config(training_updates={"epochs": 1})
    with patch("backend.app.ai.segmentation_loss.CrossEntropyDiceLoss.forward", return_value=torch.tensor(float("nan"))):
        with pytest.raises(ValueError, match="Nonfinite training loss"):
            train_segmentation_baseline(manifest, config, output_dir=tmp_path / "failed_v1", allow_synthetic=True)
    assert not list((tmp_path / "failed_v1").glob("*.pt"))
    assert not (tmp_path / "failed_v1/report.json").exists()


def test_example_config_is_disabled_pretrained_independent_and_matches_defaults():
    root = Path(__file__).resolve().parents[1]
    config = SegmentationBaselineConfig.model_validate_json((root / "config/segmentation_baseline_example.json").read_text())
    assert config == SegmentationBaselineConfig()
    assert config.pretrained_encoder and not config.pipeline.training.enabled
    assert not config.pipeline.training.share_anomaly_encoder and not config.pipeline.training.allow_synthetic
