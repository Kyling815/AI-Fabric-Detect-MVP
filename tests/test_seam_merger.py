from datetime import timedelta, timezone
from itertools import permutations

import pytest

from backend.app.core.physical_coordinates import EncoderReading, SimulatedLinePosition
from backend.app.core.seam import SeamPolicy
from backend.app.services.seam_merger import SeamMerger
from tests.seam_fixtures import calibration, duplicate_pair, observation, seam_policy


def merge(items=None, **policy_updates):
    return SeamMerger(calibration(), seam_policy(**policy_updates)).merge(duplicate_pair() if items is None else items)


def test_same_defect_adjacent_cameras_retains_both_observations_and_scores():
    items = [observation(score=0.94), observation(camera="lane_right", score=0.99)]
    result = merge(items)
    assert len(result.groups) == 1 and result.groups[0].merged and not result.affects_qc
    assert result.groups[0].minimum.x_mm == 21 and result.groups[0].maximum.x_mm == 23
    assert result.groups[0].minimum.y_mm == 114 and result.groups[0].maximum.y_mm == 115
    assert result.pairs[0].eligible and result.pairs[0].seam_iou == 1 and result.pairs[0].center_distance_mm == 0
    assert sorted(item.candidate.score for item in result.observations) == [0.94, 0.99]
    assert "confidence" not in result.groups[0].model_dump()


@pytest.mark.parametrize("right", [observation(camera="lane_right", y=12), observation(camera="lane_right", x=6),
    observation(camera="lane_right", x=4)])
def test_nearby_separate_or_touching_defects_are_not_merged(right):
    result = merge([observation(), right])
    assert len(result.groups) == 2 and not any(group.merged for group in result.groups)
    assert result.pairs[0].reason == "GEOMETRY_NOT_DUPLICATE"


def test_conflicting_class_outputs_are_retained_for_review_not_nms_suppressed():
    result = merge([observation(), observation(camera="lane_right", label="scratch")])
    assert len(result.groups) == 2 and all(group.review_required for group in result.groups)
    assert {group.label for group in result.groups} == {"defect", "scratch"}
    assert result.pairs[0].reason == "CLASS_OR_BRANCH_CONFLICT"


def test_edge_of_fov_clips_match_test_to_seam_but_keeps_full_source_envelope():
    result = merge([observation(x=46, w=2), observation(camera="lane_right", x=6, w=4)])
    assert result.groups[0].merged and result.pairs[0].seam_iou == 1
    assert (result.groups[0].minimum.x_mm, result.groups[0].maximum.x_mm) == (23, 25)
    assert result.groups[0].geometry_kind == "bbox_envelope_not_mask_union"


@pytest.mark.parametrize("profile,reason", [(None, "CALIBRATION_UNAVAILABLE"), ({}, "INVALID_CALIBRATION")])
def test_no_valid_calibration_never_claims_physical_merge(profile, reason):
    result = SeamMerger(profile, seam_policy()).merge(duplicate_pair())
    assert result.status == "unavailable" and reason in result.issues
    assert len(result.observations) == len(result.groups) == 2
    assert all(not group.merged and group.minimum is None and group.review_required for group in result.groups)


def test_default_policy_does_not_guess_camera_adjacency():
    result = SeamMerger(calibration()).merge(duplicate_pair())
    assert len(result.groups) == 2 and result.pairs[0].reason == "CAMERAS_NOT_DECLARED_ADJACENT"


@pytest.mark.parametrize("failure", ["fabric", "lot", "roll", "capture_group", "frame_index"])
def test_different_capture_scopes_never_merge(failure):
    frame_updates, updates = {}, {}
    if failure == "capture_group":
        updates["capture_group_id"] = "another_capture"
    elif failure == "frame_index":
        frame_updates.update(frame_index=2, position=EncoderReading(encoder_id="simulated_encoder", frame_index=2, count=1100, simulated=True))
    else:
        frame_updates[failure + "_id"] = "other"
    result = merge([observation(), observation(camera="lane_right", frame_updates=frame_updates, **updates)])
    assert len(result.groups) == 2 and result.pairs[0].reason == "CAPTURE_SCOPE_MISMATCH"


@pytest.mark.parametrize("field", ["lot_id", "roll_id"])
def test_missing_production_identity_is_review_not_duplicate(field):
    result = merge([observation(frame_updates={field: None}), observation(camera="lane_right")])
    assert len(result.groups) == 2 and any(group.review_required for group in result.groups)
    assert result.pairs[0].reason == "PRODUCTION_IDENTITY_UNVERIFIED"


@pytest.mark.parametrize("failure", ["time", "encoder_count", "encoder_id", "position_kind", "simulated_flag"])
def test_unverified_synchronization_does_not_merge(failure):
    frame_updates, updates = {}, {}
    if failure == "time":
        updates["timestamp"] = observation().timestamp + timedelta(microseconds=1)
    elif failure == "position_kind":
        frame_updates["position"] = SimulatedLinePosition(frame_index=1, line_index=40)
    else:
        fields = observation().frame.position.model_dump()
        fields[{"encoder_count": "count", "encoder_id": "encoder_id", "simulated_flag": "simulated"}[failure]] = {
            "encoder_count": 1101, "encoder_id": "other", "simulated_flag": False}[failure]
        frame_updates["position"] = EncoderReading.model_validate(fields)
    result = merge([observation(), observation(camera="lane_right", frame_updates=frame_updates, **updates)])
    assert len(result.groups) == 2
    assert result.pairs[0].reason in {"CAPTURE_SYNCHRONIZATION_UNVERIFIED", "PHYSICAL_MAPPING_UNAVAILABLE"}


@pytest.mark.parametrize("delta_us,expected", [(999, True), (1000, True), (1001, False)])
def test_explicit_timestamp_tolerance_has_inclusive_boundary(delta_us, expected):
    result = merge([observation(), observation(camera="lane_right", timestamp=observation().timestamp + timedelta(microseconds=delta_us))],
                   maximum_time_delta_ms=1.0)
    assert any(group.merged for group in result.groups) is expected


def test_equivalent_timezones_represent_same_capture_instant():
    right = observation(camera="lane_right", timestamp=observation().timestamp.astimezone(timezone(timedelta(hours=7))))
    assert merge([observation(), right]).groups[0].merged


@pytest.mark.parametrize("field,value", [("model_version", "other_model"), ("model_sha256", "b" * 64), ("reference_version", "other_ref")])
def test_model_or_reference_mismatch_is_not_silently_combined(field, value):
    result = merge([observation(), observation(camera="lane_right", **{field: value})])
    assert len(result.groups) == 2 and result.pairs[0].reason == "MODEL_OR_REFERENCE_MISMATCH"
    assert all(group.review_required for group in result.groups)


def test_supervised_and_anomaly_sources_never_average_scores_or_classes():
    result = merge([observation(), observation(camera="lane_right", source_kind="anomaly", reference_version="ref_v1", score=2.5)])
    assert len(result.groups) == 2 and result.pairs[0].reason == "CLASS_OR_BRANCH_CONFLICT"
    assert sorted(item.candidate.score for item in result.observations) == [0.94, 2.5]


def test_unknown_anomaly_duplicate_can_group_but_never_becomes_known_or_qc_good():
    items = [observation(camera=camera, source_kind="anomaly", reference_version="ref_v1", label="dense_anomaly", score=2.5)
             for camera in ("lane_left", "lane_right")]
    result = merge(items)
    assert result.groups[0].merged and result.groups[0].label == "dense_anomaly" and result.affects_qc is False


@pytest.mark.parametrize("iou,limit,expected", [(0.6, 0.5, True), (0.60001, 0.5, False), (0.5, 0.4999, False)])
def test_iou_and_center_distance_thresholds_are_both_required(iou, limit, expected):
    result = merge([observation(), observation(camera="lane_right", x=3)], minimum_seam_iou=iou, maximum_center_distance_mm=limit)
    assert any(group.merged for group in result.groups) is expected


def test_candidate_outside_shared_fov_is_retained_separately():
    result = merge([observation(x=0), observation(camera="lane_right", x=0)])
    assert len(result.groups) == 2 and result.pairs[0].reason == "OUTSIDE_SHARED_FOV"


@pytest.mark.parametrize("origin", [24.0, 25.0])
def test_touching_or_disjoint_calibrated_fovs_are_not_overlap(origin):
    fields = calibration().model_dump(mode="json")
    fields["cameras"][1]["x_origin_mm"] = origin
    result = SeamMerger(fields, seam_policy()).merge(duplicate_pair())
    assert len(result.groups) == 2 and result.pairs[0].reason == "NO_CALIBRATED_SHARED_FOV"


def test_mirrored_camera_calibration_maps_before_matching():
    fields = calibration().model_dump(mode="json")
    fields["cameras"][1].update(x_origin_mm=44, x_direction=-1)
    result = SeamMerger(fields, seam_policy()).merge([observation(), observation(camera="lane_right", x=42)])
    assert result.groups[0].merged and result.pairs[0].seam_iou == 1


def test_one_to_many_matching_is_not_resolved_by_greedy_score_selection():
    result = merge([*duplicate_pair(), observation(camera="lane_right", candidate_id="another", score=0.95)])
    assert len(result.groups) == 3 and not any(group.merged for group in result.groups)
    assert all(group.review_required and "AMBIGUOUS_MATCH_SET" in group.reasons for group in result.groups)


def test_class_conflict_also_blocks_suppression_of_other_close_candidates():
    result = merge([*duplicate_pair(), observation(camera="lane_right", label="scratch", candidate_id="conflict")])
    assert len(result.groups) == 3 and all(group.review_required for group in result.groups)


@pytest.mark.parametrize("complete", [False, True])
def test_three_camera_chain_needs_complete_link_not_transitive_closeness(complete):
    fields = calibration().model_dump(mode="json")
    fields["cameras"].append({**fields["cameras"][1], "camera_id": "third", "x_origin_mm": 22})
    pairs = [("lane_left", "lane_right"), ("lane_right", "third")]
    if complete:
        pairs.append(("lane_left", "third"))
    items = [observation(x=46, w=2), observation(camera="lane_right", x=6, w=2), observation(camera="third", x=2, w=2)]
    result = SeamMerger(fields, seam_policy(adjacent_camera_pairs=tuple(pairs))).merge(items)
    assert len(result.groups) == (1 if complete else 3)
    assert sum(group.merged for group in result.groups) == int(complete)
    if not complete:
        assert all(group.review_required for group in result.groups)


def test_two_separate_defect_pairs_remain_two_groups_and_input_permutations_are_stable():
    items = [observation(camera=camera, y=y, h=2, candidate_id=str(y)) for camera in ("lane_left", "lane_right") for y in (4, 12)]
    expected = merge(items).model_dump_json()
    assert len(merge(items).groups) == 2
    for reordered in permutations(items):
        assert merge(list(reordered)).model_dump_json() == expected


@pytest.mark.parametrize("failure", ["same_camera", "wrong_size", "out_of_bounds", "unknown_camera"])
def test_bad_geometry_or_same_camera_never_suppresses_raw_candidates(failure):
    right = observation(camera="lane_right")
    if failure == "same_camera":
        right = observation(candidate_id="another")
    elif failure == "wrong_size":
        right = observation(camera="lane_right", frame_updates={"image_width": 24})
    elif failure == "out_of_bounds":
        right = observation(camera="lane_right", x=47)
    else:
        right = observation(camera="unknown")
    result = merge([observation(), right])
    assert len(result.observations) == len(result.groups) == 2 and not any(group.merged for group in result.groups)


def test_adjacency_with_uncalibrated_lane_is_explicitly_unavailable():
    result = merge(adjacent_camera_pairs=(("lane_left", "unknown"),))
    assert result.status == "unavailable" and "ADJACENCY_CAMERA_NOT_CALIBRATED" in result.issues
    assert len(result.groups) == 2 and not any(group.merged for group in result.groups)


@pytest.mark.parametrize("failure", ["duplicates", "too_many", "generator", "raw_dict", "forged_observation", "forged_policy"])
def test_invalid_batch_is_explicit_error_not_truncation_or_partial_filter(failure):
    engine, items = SeamMerger(calibration(), seam_policy(max_observations=2)), duplicate_pair()
    if failure == "duplicates":
        items = [items[0], items[0]]
    elif failure == "too_many":
        items.append(observation(candidate_id="third"))
    elif failure == "generator":
        items = iter(items)
    elif failure == "raw_dict":
        items[0] = items[0].model_dump()
    elif failure == "forged_observation":
        items[0] = items[0].model_copy(update={"timestamp": None})
    else:
        engine.policy = engine.policy.model_copy(update={"minimum_seam_iou": -1})
    with pytest.raises(ValueError):
        engine.merge(items)


def test_inputs_and_calibration_remain_unchanged_and_empty_batch_is_valid():
    profile, policy, items = calibration(), seam_policy(), duplicate_pair()
    snapshots = (profile.model_dump_json(), policy.model_dump_json(), [item.model_dump_json() for item in items])
    engine = SeamMerger(profile, policy)
    engine.merge(items)
    assert snapshots == (profile.model_dump_json(), policy.model_dump_json(), [item.model_dump_json() for item in items])
    assert not engine.merge([]).groups and not SeamMerger(None).merge([]).groups


def test_numeric_area_overflow_is_review_not_an_infinite_iou_merge():
    fields = calibration().model_dump(mode="json")
    for index, camera in enumerate(fields["cameras"]):
        camera.update(x_mm_per_pixel=1e150, y_mm_per_pixel=1e160, x_origin_mm=4e151 if index else 0)
    result = SeamMerger(fields, seam_policy()).merge(duplicate_pair())
    assert len(result.groups) == 2 and all(group.review_required and not group.merged for group in result.groups)
    assert result.pairs[0].reason == "NUMERIC_GEOMETRY_UNAVAILABLE"
