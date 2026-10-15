from datetime import datetime

import pytest

from backend.app.core.line_scan import ChunkBundle, LineScanConfig, LineScanChunk, ScanLineBundle, StreamSnapshot
from backend.app.services.line_scan_runtime import ChunkBuilder
from backend.app.services.line_scan_source import SimulatedLineScanSource
from tests.line_scan_fixtures import scan_config


@pytest.mark.parametrize("field,value", [("camera_ids", ["a", "b", "c"]), ("camera_ids", ["a", "b", "c", "d", "e"]),
    ("camera_ids", ["a", "a", "c", "d"]), ("camera_ids", [" ", "b", "c", "d"]),
    ("image_width", True), ("image_width", 0), ("image_width", 4097), ("chunk_height", -1), ("chunk_height", "8"),
    ("total_lines", -1), ("total_lines", 1000001), ("queue_capacity", 0), ("queue_capacity", 129),
    ("encoder_start_count", 1.5), ("counts_per_line", 0), ("counts_per_line", -1), ("line_period_us", 0),
    ("start_timestamp", datetime(2026, 10, 15)), ("start_timestamp", 12345), ("lot_id", " "), ("roll_id", ""),
    ("fabric_id", " f "), ("encoder_id", ""), ("encoder_start_count", 2 ** 63 - 1),
    ("start_timestamp", "9999-12-31T23:59:59.999999Z")])
def test_invalid_config_cannot_guess_units_ids_clock_or_bounds(field, value):
    with pytest.raises(ValueError):
        scan_config(**{field: value})


def test_payload_memory_budget_rejected_before_source_allocation():
    with pytest.raises(ValueError, match="budget"):
        scan_config(image_width=4096, chunk_height=2048, queue_capacity=128)


def test_timezone_normalization_and_strict_extra_fields():
    config = scan_config()
    assert config.start_timestamp.isoformat() == "2026-10-15T02:00:00+00:00"
    with pytest.raises(ValueError):
        LineScanConfig.model_validate({**config.model_dump(), "drop_oldest": True})


@pytest.mark.parametrize("failure", ["bad_rgb", "duplicate_lane", "wrong_encoder_index", "not_simulated", "numeric_simulated", "naive_clock"])
def test_scan_bundle_cannot_claim_fake_alignment(failure):
    fields = SimulatedLineScanSource(scan_config()).read(0).model_dump()
    if failure == "bad_rgb":
        fields["rows"][0]["rgb_bytes"] = b"bad"
    elif failure == "duplicate_lane":
        fields["rows"][1]["camera_id"] = fields["rows"][0]["camera_id"]
    elif failure == "wrong_encoder_index":
        fields["encoder"]["frame_index"] = 1
    elif failure in {"not_simulated", "numeric_simulated"}:
        fields["simulated"] = False if failure == "not_simulated" else 1
    else:
        fields["timestamp"] = datetime(2026, 10, 15)
    with pytest.raises(ValueError):
        ScanLineBundle.model_validate(fields)


def full_bundle():
    config = scan_config(chunk_height=1)
    return ChunkBuilder(config).push(SimulatedLineScanSource(config).read(0))


@pytest.mark.parametrize("failure", ["payload", "hash", "encoder_range", "clock_range", "partial", "partial_without_reason", "not_simulated"])
def test_chunk_bytes_and_interval_must_match(failure):
    fields = full_bundle().chunks[0].model_dump()
    if failure == "payload":
        fields["rgb_bytes"] = b"bad"
    elif failure == "hash":
        fields["rgb_sha256"] = "0" * 64
    elif failure == "encoder_range":
        fields["encoder_end_exclusive"] += 1
    elif failure == "clock_range":
        fields["timestamp_end_exclusive"] = fields["timestamp"]
    elif failure == "partial":
        fields["partial"] = True
    elif failure == "partial_without_reason":
        fields.update(nominal_height=2, partial=True)
    else:
        fields["simulated"] = False
    with pytest.raises(ValueError):
        LineScanChunk.model_validate(fields)


@pytest.mark.parametrize("field,value", [("lot_id", "other"), ("encoder_start", 999), ("camera_id", "lane_1")])
def test_four_lane_chunk_bundle_cannot_desynchronize(field, value):
    fields = full_bundle().model_dump()
    fields["chunks"][1][field] = value
    with pytest.raises(ValueError):
        ChunkBundle.model_validate(fields)


@pytest.mark.parametrize("failure", ["lost_tick", "lost_bundle", "unaccounted_pending", "terminal_with_buffer", "over_capacity"])
def test_snapshot_cannot_hide_a_drop(failure):
    from backend.app.services.line_scan_runtime import SimulatedStreamRuntime
    fields = SimulatedStreamRuntime(scan_config()).snapshot().model_dump()
    if failure == "lost_tick":
        fields["accepted_line_ticks"] = 1
    elif failure == "lost_bundle":
        fields["enqueued_bundles"] = 1
    elif failure == "unaccounted_pending":
        fields["created_bundles"] = 1
    elif failure == "terminal_with_buffer":
        fields.update(state="stopped", buffered_line_ticks=1, accepted_line_ticks=1)
    else:
        fields.update(queued_bundles=2, created_bundles=2, enqueued_bundles=2)
    with pytest.raises(ValueError):
        StreamSnapshot.model_validate(fields)
