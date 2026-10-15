import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.ai.lot_evaluation import binary_auroc, evaluate_fabric_lot
from backend.app.ai.dense_anomaly import aggregate_image_score
from backend.app.core.lot_evaluation import LotEvaluationManifest, ResearchAggregationConfig
from backend.app.core.lot_memory import LotMemoryConfig
from scripts.evaluate_fabric_lot_memory import create_synthetic_evaluation_fixture


@pytest.mark.parametrize("labels,scores,expected", [
    (["normal", "defect"], [0, 1], 1), (["normal", "defect"], [1, 0], 0),
    (["normal", "defect"], [1, 1], 0.5), (["normal", "normal", "defect", "defect"], [0, 1, 1, 2], 0.875),
    (["normal"], [0], None), (["defect"], [1], None),
])
def test_auc_known_values_and_single_class_fallback(labels, scores, expected):
    assert binary_auroc(labels, scores) == expected


@pytest.mark.parametrize("labels,scores", [(["invalid"], [0]), (["normal"], []), (["normal"], [float("nan")])])
def test_auc_invalid_input_rejected(labels, scores):
    with pytest.raises(ValueError):
        binary_auroc(labels, scores)


def test_comparison_reports_synthetic_only_without_production_accuracy_claim(tmp_path):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    result = evaluate_fabric_lot(hierarchy, dataset)
    assert result["comparison_status"] == "COMPLETED_RESEARCH_ONLY"
    assert result["fabric_id"] == "fabric_001" and result["lot_id"] == "demo_lot"
    assert result["evaluation_kind"] == "synthetic_or_mixed_research"
    assert result["real_evaluation_performed"] is False
    assert result["fresh_locked_test"] is False and result["qc_performed"] is False
    assert result["improvement_status"] == "UNVERIFIED_PRODUCTION_BENEFIT"
    assert result["fabric_only_auroc"] == result["fabric_plus_lot_auroc"] == 1
    assert result["auroc_difference"] == 0
    assert len(result["per_image"]) == 4
    assert result["per_image"][0]["fabric_plus_lot_score"] < result["per_image"][0]["fabric_only_score"]


def test_independent_evaluation_single_class_does_not_invent_auc(tmp_path):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    dataset = dataset.model_copy(update={"samples": dataset.samples[:2]})
    result = evaluate_fabric_lot(hierarchy, dataset)
    assert result["fabric_only_auroc"] is None and result["fabric_plus_lot_auroc"] is None
    assert result["auroc_difference"] is None


def test_disabled_comparison_requires_explicit_opt_in(tmp_path):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    hierarchy = hierarchy.model_copy(update={"config": LotMemoryConfig(mode="global_plus_lot")})
    with pytest.raises(ValueError, match="explicit enabled"):
        evaluate_fabric_lot(hierarchy, dataset)


@pytest.mark.parametrize("field,value", [("fabric_id", "other_fabric"), ("lot_id", "other_lot"),
                                        ("encoder_version", "other_encoder"), ("adapter_version", "other_adapter")])
def test_evaluation_scope_or_embedding_version_mismatch_fails(tmp_path, field, value):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    changed = dataset.model_copy(update={field: value})
    with pytest.raises(ValueError, match="match"):
        evaluate_fabric_lot(hierarchy, changed)


@pytest.mark.parametrize("overlap", ["image_id", "image_hash", "image_path", "parent_hash"])
def test_evaluation_rejects_reference_overlap_and_known_descendants(tmp_path, overlap):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    source = hierarchy.lot_memory.identity.source_good_images[0]
    field, value = {"image_id": ("image_id", source.image_id), "image_hash": ("image_sha256", source.sha256),
                    "image_path": ("image_path", source.path), "parent_hash": ("parent_source_sha256", source.sha256)}[overlap]
    first = dataset.samples[0].model_copy(update={field: value})
    changed = dataset.model_copy(update={"samples": (first, *dataset.samples[1:])})
    with pytest.raises(ValueError, match="overlap|descendant"):
        evaluate_fabric_lot(hierarchy, changed)


@pytest.mark.parametrize("field", ["image_sha256", "embeddings_sha256"])
def test_query_hash_failure_is_fatal(tmp_path, field):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    first = dataset.samples[0].model_copy(update={field: "0" * 64})
    changed = dataset.model_copy(update={"samples": (first, *dataset.samples[1:])})
    with pytest.raises(ValueError, match="checksum"):
        evaluate_fabric_lot(hierarchy, changed)


def test_bank_vectors_cannot_be_mislabeled_as_independent_queries(tmp_path):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    first = dataset.samples[0].model_copy(update={"embeddings_path": str(tmp_path / "lot" / "memory_bank.npy")})
    with pytest.raises(ValueError, match="Memory artifact vectors"):
        evaluate_fabric_lot(hierarchy, dataset.model_copy(update={"samples": (first, *dataset.samples[1:])}))


def test_missing_lot_fallback_is_not_reported_as_completed_comparison(tmp_path):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    config = LotMemoryConfig(enabled=True, mode="global_plus_lot", missing_lot_policy="global_only")
    hierarchy = hierarchy.model_copy(update={"config": config, "lot_memory": None})
    result = evaluate_fabric_lot(hierarchy, dataset)
    assert result["comparison_status"] == "NOT_COMPARED_MISSING_LOT_MEMORY"
    assert result["effective_mode"] == "global_only"
    assert result["fabric_plus_lot_auroc"] is None and result["auroc_difference"] is None


@pytest.mark.parametrize("failure", ["locked_split", "duplicate_id", "duplicate_hash", "wrong_sample_fabric", "wrong_sample_lot"])
def test_bad_dataset_manifest_is_rejected(tmp_path, failure):
    _, dataset = create_synthetic_evaluation_fixture(tmp_path)
    values = dataset.model_dump(mode="json")
    if failure == "locked_split":
        values["split"] = "locked_test"
    elif failure == "duplicate_id":
        values["samples"][1]["image_id"] = values["samples"][0]["image_id"]
    elif failure == "duplicate_hash":
        values["samples"][1]["image_sha256"] = values["samples"][0]["image_sha256"]
    else:
        values["samples"][0]["fabric_id" if failure == "wrong_sample_fabric" else "lot_id"] = "other"
    with pytest.raises(ValidationError):
        LotEvaluationManifest.model_validate(values)


@pytest.mark.parametrize("method", ["max", "top_k_mean", "top_k_median_contrast"])
def test_comparison_reuses_explicit_aggregation_policy(tmp_path, method):
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    config = ResearchAggregationConfig(image_score_method=method, image_score_top_fraction=0.2)
    result = evaluate_fabric_lot(hierarchy, dataset.model_copy(update={"aggregation": config}))
    assert result["aggregation"]["image_score_method"] == method
    assert result["qc_performed"] is False


def test_cli_demo_is_synthetic_and_does_not_leave_user_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts/evaluate_fabric_lot_memory.py"), "--demo"],
                               cwd=tmp_path, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert result["real_evaluation_performed"] is False
    assert result["fabric_only_auroc"] == result["fabric_plus_lot_auroc"]
    assert list(tmp_path.iterdir()) == []


def test_cli_default_plan_then_explicit_comparison_is_read_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    hierarchy, dataset = create_synthetic_evaluation_fixture(tmp_path)
    hierarchy_path, dataset_path = tmp_path / "hierarchy.json", tmp_path / "evaluation.json"
    hierarchy_path.write_text(hierarchy.model_dump_json(indent=2), encoding="utf-8")
    dataset_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    command = [sys.executable, str(root / "scripts/evaluate_fabric_lot_memory.py"),
               "--hierarchy", str(hierarchy_path), "--dataset", str(dataset_path)]
    plan = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(plan.stdout)["status"] == "PLAN_ONLY"
    compared = subprocess.run([*command, "--enable-lot-memory"], capture_output=True, text=True, check=True)
    assert json.loads(compared.stdout)["comparison_status"] == "COMPLETED_RESEARCH_ONLY"
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_lower_patch_distances_do_not_imply_lower_contrast_image_score():
    global_scores = np.array([0.2, 0.8], dtype=np.float32)
    union_scores = np.array([0.0, 0.7], dtype=np.float32)
    assert np.all(union_scores <= global_scores)
    assert aggregate_image_score(union_scores, method="max_median_contrast") > aggregate_image_score(global_scores, method="max_median_contrast")
