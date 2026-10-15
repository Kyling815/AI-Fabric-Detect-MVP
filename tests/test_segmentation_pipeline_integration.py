import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from backend.app.ai.segmentation_dataset import SegmentationDataset
from backend.app.core.segmentation_data import SegmentationAugmentationConfig
from scripts.check_segmentation_pipeline import create_demo_manifest
from tests.phase1_fixtures import baseline, normalize_response, phase1_runtime


def test_loader_batch_shapes_dtypes_and_deterministic_sample_epoch(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    augmentation = SegmentationAugmentationConfig(enabled=True)
    dataset = SegmentationDataset(manifest, partition="train", output_size=(16, 24), allow_synthetic=True, augmentation=augmentation)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    assert batch["image"].shape == (2, 3, 16, 24) and batch["image"].dtype == torch.float32
    assert batch["mask"].shape == (2, 16, 24) and batch["mask"].dtype == torch.int64
    assert torch.isfinite(batch["image"]).all() and 0 <= batch["image"].min() <= batch["image"].max() <= 1
    dataset.set_epoch(2)
    assert torch.equal(dataset[0]["image"], dataset[0]["image"])
    fresh = SegmentationDataset(manifest, partition="train", output_size=(16, 24), allow_synthetic=True, augmentation=augmentation)
    fresh.set_epoch(2)
    assert torch.equal(dataset[0]["image"], fresh[0]["image"])


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_evaluation_never_augments_or_changes_with_epoch(tmp_path, partition):
    manifest = create_demo_manifest(tmp_path)
    dataset = SegmentationDataset(manifest, partition=partition, output_size=(16, 24), allow_synthetic=True,
                                  augmentation=SegmentationAugmentationConfig(enabled=True))
    before = dataset[0]
    dataset.set_epoch(100)
    assert torch.equal(before["image"], dataset[0]["image"])
    assert torch.equal(before["mask"], dataset[0]["mask"])


@pytest.mark.parametrize("failure", ["synthetic_default", "historical_partition", "invalid_size", "invalid_seed"])
def test_dataset_rejects_unsafe_or_invalid_configuration(tmp_path, failure):
    manifest = create_demo_manifest(tmp_path)
    values = {"partition": "train", "allow_synthetic": True}
    values.update({"synthetic_default": {"allow_synthetic": False}, "historical_partition": {"partition": "historical_locked"},
                   "invalid_size": {"output_size": (0, 12)}, "invalid_seed": {"seed": True}}[failure])
    with pytest.raises(ValueError):
        SegmentationDataset(manifest, **values)


@pytest.mark.parametrize("case", list(baseline()["cases"]))
def test_production_inspect_isolated_from_segmentation_pipeline(phase1_runtime, case):
    phase1_runtime.use_case(case)
    with patch.object(SegmentationDataset, "__init__", side_effect=AssertionError("production must not load supervised dataset")):
        response = phase1_runtime.upload()
    assert response.status_code == 201
    snapshot = baseline()
    assert normalize_response(response.json()) == {**snapshot["common"], **snapshot["cases"][case]["post_overrides"]}
    saved = phase1_runtime.client.get(f"/api/inspections/{response.json()['inspection_id']}")
    assert normalize_response(saved.json()) == {**snapshot["common"], **snapshot["cases"][case]["get_overrides"]}
    schema = phase1_runtime.client.get("/openapi.json").json()
    assert schema["paths"]["/api/inspect"] == snapshot["openapi_inspect"]
    assert schema["components"]["schemas"]["InspectionCreateResponse"] == snapshot["response_schema"]


@pytest.mark.parametrize("command", ["--demo", "--schema"])
def test_cli_demo_or_schema_creates_no_permanent_artifacts(tmp_path, command):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/check_segmentation_pipeline.py"), command],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout)
    if command == "--demo":
        assert value["trained_model"] is False and value["production_accuracy_evaluated"] is False
    else:
        assert value["title"] == "SegmentationManifest"
    assert list(tmp_path.iterdir()) == []


def test_cli_explicit_manifest_preflight_is_read_only_and_requires_synthetic_opt_in(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = create_demo_manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    command = [sys.executable, str(root / "scripts/check_segmentation_pipeline.py"), "--manifest", str(path)]
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and "Synthetic samples" in failed.stderr
    checked = subprocess.run([*command, "--allow-synthetic"], capture_output=True, text=True, check=True)
    report = json.loads(checked.stdout)
    assert report["checked_samples"] == 9 and not report["trained_model"] and not report["writes_performed"]
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}
