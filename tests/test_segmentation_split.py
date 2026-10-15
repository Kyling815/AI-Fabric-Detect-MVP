import pytest
from pydantic import ValidationError

from backend.app.ai.segmentation_split import grouped_segmentation_split
from backend.app.core.segmentation_data import GroupedSplitConfig
from scripts.check_segmentation_pipeline import create_demo_manifest


def assert_disjoint(manifest, assignments, field, scoped=False):
    owners = {}
    for sample in manifest.samples:
        value = getattr(sample, field)
        key = (sample.fabric_id, value) if scoped else value
        if value is not None:
            if key in owners:
                assert owners[key] == assignments[sample.sample_id]
            owners[key] = assignments[sample.sample_id]


def test_grouped_split_is_source_roll_lot_camera_disjoint_and_order_invariant(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    config = GroupedSplitConfig(group_by_lot=True, group_by_camera=True)
    first = grouped_segmentation_split(manifest, config)
    second = grouped_segmentation_split(manifest.model_copy(update={"samples": tuple(reversed(manifest.samples))}), config)
    assert first.assignments == second.assignments
    for field in ("source_id", "roll_id", "lot_id", "camera_id"):
        assert_disjoint(manifest, first.assignments, field, scoped=field in {"roll_id", "lot_id"})
    assert all(first.sample_counts[split] > 0 for split in ("train", "validation", "test"))


def test_seed_is_deterministic_and_can_change_whole_group_assignment(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    one = grouped_segmentation_split(manifest, GroupedSplitConfig(seed=1)).assignments
    assert one == grouped_segmentation_split(manifest, GroupedSplitConfig(seed=1)).assignments
    possibilities = {tuple(sorted(grouped_segmentation_split(manifest, GroupedSplitConfig(seed=seed)).assignments.items())) for seed in range(10)}
    assert len(possibilities) > 1


@pytest.mark.parametrize("connection", ["ancestor_id", "source_hash", "lot", "camera"])
def test_transitive_group_connections_are_not_split(tmp_path, connection):
    manifest = create_demo_manifest(tmp_path)
    rows = list(manifest.samples)
    if connection == "ancestor_id":
        rows[0] = rows[0].model_copy(update={"ancestor_source_ids": (rows[3].source_id,)})
    elif connection == "source_hash":
        rows[0] = rows[0].model_copy(update={"source_sha256": rows[3].image_sha256})
    else:
        field = "lot_id" if connection == "lot" else "camera_id"
        rows[0] = rows[0].model_copy(update={field: getattr(rows[3], field)})
    updated = manifest.model_copy(update={"samples": tuple(rows)})
    config = GroupedSplitConfig(train_ratio=0.5, validation_ratio=0, test_ratio=0.5,
                                group_by_lot=connection == "lot", group_by_camera=connection == "camera")
    result = grouped_segmentation_split(updated, config)
    assert result.assignments[rows[0].sample_id] == result.assignments[rows[3].sample_id]


def test_ancestor_of_historical_locked_data_is_quarantined_not_relabelled_fresh(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    rows = list(manifest.samples)
    rows[0] = rows[0].model_copy(update={"split": "historical_locked"})
    result = grouped_segmentation_split(manifest.model_copy(update={"samples": tuple(rows)}),
                                       GroupedSplitConfig(train_ratio=0.5, validation_ratio=0, test_ratio=0.5))
    assert result.assignments[rows[0].sample_id] == "historical_locked"
    assert result.assignments[rows[1].sample_id] == result.assignments[rows[2].sample_id] == "excluded"


@pytest.mark.parametrize("conflict", ["different_pins", "protected_pin"])
def test_preassigned_leakage_fails_instead_of_weakening_split(tmp_path, conflict):
    manifest = create_demo_manifest(tmp_path)
    rows = list(manifest.samples)
    rows[0] = rows[0].model_copy(update={"split": "train"})
    rows[1] = rows[1].model_copy(update={"split": "validation" if conflict == "different_pins" else "historical_locked"})
    with pytest.raises(ValueError, match="leaks|Protected"):
        grouped_segmentation_split(manifest.model_copy(update={"samples": tuple(rows)}), GroupedSplitConfig())


@pytest.mark.parametrize("field,config", [("roll_id", GroupedSplitConfig()), ("lot_id", GroupedSplitConfig(group_by_lot=True)),
                                        ("camera_id", GroupedSplitConfig(group_by_camera=True))])
def test_missing_required_capture_metadata_blocks_split(tmp_path, field, config):
    manifest = create_demo_manifest(tmp_path)
    rows = list(manifest.samples); rows[0] = rows[0].model_copy(update={field: None})
    with pytest.raises(ValueError, match="required|requires"):
        grouped_segmentation_split(manifest.model_copy(update={"samples": tuple(rows)}), config)


def test_single_camera_group_cannot_be_randomly_split_to_meet_ratios(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    rows = tuple(row.model_copy(update={"camera_id": "one_camera"}) for row in manifest.samples)
    with pytest.raises(ValueError, match="Not enough independent"):
        grouped_segmentation_split(manifest.model_copy(update={"samples": rows}), GroupedSplitConfig(group_by_camera=True))


def test_source_only_override_is_explicit_and_reports_missing_roll_warning(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    rows = tuple(row.model_copy(update={"roll_id": None}) for row in manifest.samples)
    result = grouped_segmentation_split(manifest.model_copy(update={"samples": rows}), GroupedSplitConfig(require_roll_id=False))
    assert any("unverified" in warning for warning in result.warnings)


@pytest.mark.parametrize("config", [{"train_ratio": 0.8}, {"seed": True}, {"group_by_lot": "true"}, {"train_ratio": float("nan")}])
def test_invalid_split_config_rejected(config):
    with pytest.raises(ValidationError):
        GroupedSplitConfig.model_validate(config)


def test_duplicate_image_content_connects_different_source_ids_and_rolls(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    rows = list(manifest.samples)
    rows[0] = rows[0].model_copy(update={"image_sha256": rows[3].image_sha256})
    result = grouped_segmentation_split(manifest.model_copy(update={"samples": tuple(rows)}),
                                       GroupedSplitConfig(train_ratio=0.5, validation_ratio=0, test_ratio=0.5))
    assert result.assignments[rows[0].sample_id] == result.assignments[rows[3].sample_id]


def test_existing_safe_pins_are_preserved(tmp_path):
    manifest = create_demo_manifest(tmp_path)
    rows = list(manifest.samples)
    rows[0] = rows[0].model_copy(update={"split": "train"})
    result = grouped_segmentation_split(manifest.model_copy(update={"samples": tuple(rows)}), GroupedSplitConfig())
    assert all(result.assignments[row.sample_id] == "train" for row in rows[:3])


def test_bool_ratio_is_not_silently_used_as_number():
    with pytest.raises(ValidationError):
        GroupedSplitConfig(train_ratio=True, validation_ratio=0, test_ratio=0)
