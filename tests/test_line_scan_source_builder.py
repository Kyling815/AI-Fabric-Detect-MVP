from datetime import timedelta
import hashlib

import pytest

from backend.app.services.line_scan_runtime import ChunkBuilder
from backend.app.services.line_scan_source import SimulatedEncoder, SimulatedLineScanSource
from tests.line_scan_fixtures import scan_config


@pytest.mark.parametrize("index", [0, 1, 18])
def test_four_lanes_share_virtual_timestamp_and_encoder_tick(index):
    config = scan_config()
    source = SimulatedLineScanSource(config)
    line = source.read(index)
    assert line == source.read(index)
    assert line.timestamp == config.start_timestamp + timedelta(microseconds=index * config.line_period_us)
    assert line.encoder.count == 1000 + index * 2 and line.encoder.frame_index == index
    assert tuple(row.camera_id for row in line.rows) == config.camera_ids
    assert len({row.rgb_bytes for row in line.rows}) == 4
    assert all(len(row.rgb_bytes) == config.image_width * 3 for row in line.rows)


@pytest.mark.parametrize("index", [-1, True, "1", 1.5, 19])
def test_source_rejects_bad_index_and_explicitly_signals_eof(index):
    with pytest.raises(StopIteration if index == 19 else ValueError):
        SimulatedLineScanSource(scan_config()).read(index)


@pytest.mark.parametrize("origin", [-1000, 0, 1000])
def test_simulated_encoder_reuses_explicit_phase12_count_contract(origin):
    encoder = SimulatedEncoder(scan_config(encoder_start_count=origin))
    assert encoder.read(5).count == origin + 10 and encoder.read(5).simulated


@pytest.mark.parametrize("height,total", [(1, 0), (1, 1), (2, 5), (8, 19), (32, 64)])
def test_builder_preserves_all_rows_order_geometry_and_ranges(height, total):
    config = scan_config(chunk_height=height, total_lines=total)
    source, builder, output = SimulatedLineScanSource(config), ChunkBuilder(config), []
    for index in range(total):
        complete = builder.push(source.read(index))
        if complete is not None:
            output.append(complete)
    tail = builder.finish("source_exhausted")
    if tail:
        output.append(tail)
    assert builder.next_line == total and builder.buffered_line_ticks == 0 and builder.closed
    assert sum(item.line_count for item in output) == total
    for lane in range(4):
        rows = b"".join(source.read(i).rows[lane].rgb_bytes for i in range(total))
        chunks = [bundle.chunks[lane] for bundle in output]
        assert b"".join(chunk.rgb_bytes for chunk in chunks) == rows
        for index, chunk in enumerate(chunks):
            assert chunk.start_line == index * height and chunk.chunk_index == index
            assert chunk.rgb_sha256 == hashlib.sha256(chunk.rgb_bytes).hexdigest()
            assert chunk.encoder_start == config.encoder_start_count + chunk.start_line * config.counts_per_line
            assert chunk.timestamp == config.timestamp_at(chunk.start_line)
            assert chunk.to_frame_coordinates().position.count == chunk.encoder_start
            assert chunk.to_frame_coordinates().image_height == chunk.image_height


@pytest.mark.parametrize("failure", ["skip", "duplicate", "fabric", "lot", "roll", "simulation", "time", "encoder_id",
    "encoder_count", "lane_order", "width", "payload", "hardware_encoder"])
def test_builder_rejects_entire_invalid_tick_before_mutating_any_lane(failure):
    config = scan_config(chunk_height=2)
    source, builder = SimulatedLineScanSource(config), ChunkBuilder(config)
    builder.push(source.read(0))
    line = source.read(1)
    if failure in {"skip", "duplicate"}:
        line = source.read(2 if failure == "skip" else 0)
    elif failure in {"fabric", "lot", "roll", "simulation"}:
        key = {"fabric": "fabric_id", "lot": "lot_id", "roll": "roll_id", "simulation": "simulation_id"}[failure]
        line = line.model_copy(update={key: "other"})
    elif failure == "time":
        line = line.model_copy(update={"timestamp": config.timestamp_at(0)})
    elif failure.startswith("encoder_") or failure == "hardware_encoder":
        update = {"encoder_id": {"encoder_id": "other"}, "encoder_count": {"count": 999}, "hardware_encoder": {"simulated": False}}[failure]
        line = line.model_copy(update={"encoder": line.encoder.model_copy(update=update)})
    elif failure == "lane_order":
        line = line.model_copy(update={"rows": tuple(reversed(line.rows))})
    else:
        first = line.rows[0].model_copy(update={"image_width": 1, "rgb_bytes": b"abc"} if failure == "width" else {"rgb_bytes": b"bad"})
        line = line.model_copy(update={"rows": (first, *line.rows[1:])})
    with pytest.raises(ValueError):
        builder.push(line)
    assert builder.next_line == 1 and builder.buffered_line_ticks == 1 and builder.chunk_index == 0
    assert builder.push(source.read(1)).line_count == 2


@pytest.mark.parametrize("reason", ["user_stop", "source_exhausted", "source_or_builder_error"])
def test_partial_flush_is_explicit_once_and_builder_cannot_restart(reason):
    config = scan_config()
    builder = ChunkBuilder(config)
    builder.push(SimulatedLineScanSource(config).read(0))
    final = builder.finish(reason)
    assert final.line_count == 1 and all(chunk.partial and chunk.termination_reason == reason for chunk in final.chunks)
    assert builder.finish(reason) is None
    with pytest.raises(ValueError):
        builder.push(SimulatedLineScanSource(config).read(1))


@pytest.mark.parametrize("reason", ["drop", "", [], None])
def test_invalid_flush_reason_does_not_discard_buffer(reason):
    builder = ChunkBuilder(scan_config())
    builder.push(SimulatedLineScanSource(scan_config()).read(0))
    with pytest.raises(ValueError):
        builder.finish(reason)
    assert not builder.closed and builder.buffered_line_ticks == 1


@pytest.mark.parametrize("constructor", [SimulatedEncoder, SimulatedLineScanSource, ChunkBuilder])
def test_model_copy_cannot_bypass_constructor_config_validation(constructor):
    config = scan_config().model_copy(update={"queue_capacity": 0})
    with pytest.raises(ValueError):
        constructor(config)
