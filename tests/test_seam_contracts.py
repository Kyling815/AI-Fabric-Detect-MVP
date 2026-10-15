from datetime import datetime
from pathlib import Path

import pytest

from backend.app.core.seam import SeamMergeResult, SeamObservation, SeamPolicy
from backend.app.services.seam_merger import SeamMerger
from tests.seam_fixtures import calibration, duplicate_pair, observation, seam_policy


@pytest.mark.parametrize("changes", [{"minimum_seam_iou": 0}, {"minimum_seam_iou": 1.1}, {"minimum_seam_iou": True},
    {"minimum_seam_iou": "0.5"}, {"maximum_center_distance_mm": 0}, {"maximum_center_distance_mm": float("inf")},
    {"maximum_time_delta_ms": -1}, {"maximum_time_delta_ms": float("nan")}, {"max_observations": 257},
    {"max_observations": True}, {"adjacent_camera_pairs": [("a", "a")]},
    {"adjacent_camera_pairs": [("a", "b"), ("b", "a")]}, {"adjacent_camera_pairs": [(" ", "b")]},
    {"adjacent_camera_pairs": [("a", "b", "c")]}, {"policy_version": " "}, {"learned_nms": True}])
def test_policy_rejects_unsafe_thresholds_or_ambiguous_adjacency(changes):
    with pytest.raises(ValueError):
        SeamPolicy(**changes)


def test_example_policy_is_starting_rules_not_automatic_adjacency():
    root = Path(__file__).resolve().parents[1]
    parsed = SeamPolicy.model_validate_json((root / "config/seam_merging_example.json").read_text(encoding="utf-8"))
    assert parsed == seam_policy() and not SeamPolicy().adjacent_camera_pairs


@pytest.mark.parametrize("failure", ["naive_clock", "numeric_clock", "no_capture_group", "no_model", "anomaly_without_reference", "model_grid"])
def test_observation_needs_explicit_scope_time_provenance_and_pixel_frame(failure):
    fields = observation().model_dump()
    if failure == "naive_clock":
        fields["timestamp"] = datetime(2026, 10, 15)
    elif failure == "numeric_clock":
        fields["timestamp"] = 12345
    elif failure == "no_capture_group":
        fields["capture_group_id"] = " "
    elif failure == "no_model":
        fields["model_version"] = ""
    elif failure == "anomaly_without_reference":
        fields["source_kind"] = "anomaly"
    else:
        fields["candidate"]["coordinate_frame"] = "model_grid"
    with pytest.raises(ValueError):
        SeamObservation.model_validate(fields)


@pytest.mark.parametrize("failure", ["lost_input", "duplicated_member", "no_calibration", "fake_envelope", "wrong_label", "no_pairs",
    "duplicate_pairs", "ineligible_pair", "qc_activation", "numeric_qc", "wrong_mapping_lineage", "unknown_group_member"])
def test_output_cannot_hide_lineage_or_claim_unsupported_merge(failure):
    fields = SeamMerger(calibration(), seam_policy()).merge(duplicate_pair()).model_dump()
    if failure == "lost_input":
        fields["groups"] = []
    elif failure == "duplicated_member":
        fields["groups"][0]["member_ids"] = tuple(fields["groups"][0]["member_ids"]) * 2
    elif failure == "no_calibration":
        fields["calibration"] = None
    elif failure == "fake_envelope":
        fields["groups"][0]["minimum"]["x_mm"] = 0
    elif failure == "wrong_label":
        fields["groups"][0]["label"] = "scratch"
    elif failure == "no_pairs":
        fields["pairs"] = []
    elif failure == "duplicate_pairs":
        fields["pairs"] = fields["pairs"] * 2
    elif failure == "ineligible_pair":
        fields["pairs"][0]["eligible"] = False
    elif failure in {"qc_activation", "numeric_qc"}:
        fields["affects_qc"] = True if failure == "qc_activation" else 0
    elif failure == "wrong_mapping_lineage":
        fields["mappings"][0]["frame"]["frame_id"] = "other"
    else:
        fields["groups"][0]["member_ids"] = ("unknown",)
    with pytest.raises(ValueError):
        SeamMergeResult.model_validate(fields)
