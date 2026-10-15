import json
from pathlib import Path

import pytest

from backend.app.core.segmentation_shadow import SupervisedShadowConfig, SupervisedShadowObservation
from tests.segmentation_shadow_fixtures import observed_fixture, shadow_config


def test_safe_defaults_and_example_match():
    root = Path(__file__).resolve().parents[1]
    example = SupervisedShadowConfig.model_validate_json((root / "config/supervised_shadow_example.json").read_text())
    assert example == SupervisedShadowConfig()
    assert not example.trusted_local and not example.allow_synthetic and not example.checkpoint_path


@pytest.mark.parametrize("values", [{"mode": "active"}, {"timeout_seconds": 0}, {"timeout_seconds": True}, {"timeout_seconds": float("nan")},
                                  {"timeout_seconds": 31}, {"trusted_local": "true"}, {"cpu_threads": 0}, {"cpu_threads": True},
                                  {"checkpoint_sha256": "invalid"}, {"checkpoint_path": "new.pt"}, {"allowed_fabric_ids": ["a", "a"]},
                                  {"allowed_fabric_ids": [" a"]}, {"max_mask_runs": 8193}, {"affects_qc": True}])
def test_invalid_or_decision_changing_config_rejected(values):
    with pytest.raises(ValueError):
        SupervisedShadowConfig(**values)


def test_observation_is_internal_bounded_and_audit_log_omits_mask_runs():
    value = observed_fixture()
    assert value.mask_rle == ((1, 1024),) and not value.affects_qc and not value.confidence_calibrated
    log = value.audit_payload()
    assert "mask_rle" not in log and log["mask_sha256"] == value.mask_sha256
    assert len(json.dumps(log)) < 4096
    assert SupervisedShadowObservation.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize("failure", ["failed_with_evidence", "wrong_version", "bad_counts", "missing_hash", "wrong_class_order",
                                   "wrong_runs", "missing_runs", "calibrated", "affects_qc", "bad_geometry", "bad_probability"])
def test_malformed_or_unsafe_observations_rejected(failure):
    values = observed_fixture().model_dump(mode="json")
    if failure == "failed_with_evidence":
        values["status"] = "timeout"
    elif failure == "wrong_version":
        values["model_version"] = "other"
    elif failure == "bad_counts":
        values["classes"][1]["predicted_pixels"] = 100
    elif failure == "missing_hash":
        values["mask_sha256"] = None
    elif failure == "wrong_class_order":
        values["taxonomy"] = ["defect", "background"]
    elif failure == "wrong_runs":
        values["mask_rle"] = [[0, 1024]]
    elif failure == "missing_runs":
        values["mask_rle"] = None
    elif failure in {"calibrated", "affects_qc"}:
        values["confidence_calibrated" if failure == "calibrated" else "affects_qc"] = True
    elif failure == "bad_geometry":
        values["mask_shape"] = [0, 32]
    else:
        values["classes"][0]["mean_class_probability"] = 0.5
    with pytest.raises(ValueError):
        SupervisedShadowObservation.model_validate(values)


def test_absent_model_does_not_fabricate_good_or_defect_evidence():
    value = SupervisedShadowObservation(status="unavailable", reason="checkpoint_missing")
    assert value.model_version is None and value.classes == () and value.mask_shape is None


def test_network_checkpoint_paths_not_accepted_as_trusted_local():
    with pytest.raises(ValueError, match="network"):
        shadow_config(checkpoint_path="//untrusted-server/share/model.pt")


def test_mask_run_checksum_must_match_reported_mask():
    values = observed_fixture().model_dump(mode="json")
    values["mask_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="checksum"):
        SupervisedShadowObservation.model_validate(values)
