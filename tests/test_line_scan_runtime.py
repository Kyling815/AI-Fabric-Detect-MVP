from queue import Empty
from unittest.mock import Mock

import pytest

from backend.app.services.line_scan_runtime import SimulatedStreamRuntime
from backend.app.services.line_scan_source import SimulatedLineScanSource
from tests.line_scan_fixtures import run_to_end, scan_config


@pytest.mark.parametrize("height,total,capacity", [(1, 0, 1), (1, 1, 1), (2, 5, 1), (8, 19, 1), (8, 19, 3), (32, 65, 2)])
def test_ordered_lossless_four_lane_delivery(height, total, capacity):
    runtime = SimulatedStreamRuntime(scan_config(chunk_height=height, total_lines=total, queue_capacity=capacity))
    output = run_to_end(runtime)
    snapshot = runtime.snapshot()
    assert snapshot.state == "stopped" and snapshot.stop_reason == "source_exhausted"
    assert snapshot.accepted_line_ticks == snapshot.delivered_line_ticks == total
    assert snapshot.queued_bundles == snapshot.pending_line_ticks == snapshot.buffered_line_ticks == 0
    assert len(output) == snapshot.created_bundles == snapshot.delivered_bundles
    for lane in range(4):
        chunks = [item.chunks[lane] for item in output]
        assert [item.chunk_index for item in chunks] == list(range(len(chunks)))
        assert len({item.chunk_id for item in chunks}) == len(chunks)
        assert sum(item.image_height for item in chunks) == total


def test_full_queue_retains_pending_bundle_and_retries_do_not_read_more_source():
    config = scan_config(chunk_height=2, total_lines=10, queue_capacity=1)
    source = Mock(wraps=SimulatedLineScanSource(config))
    runtime = SimulatedStreamRuntime(config, source=source)
    for _ in range(4):
        status = runtime.step()
    assert status.last_event == "backpressure" and status.queued_bundles == 1 and status.pending_line_ticks == 2
    assert status.accepted_line_ticks == 4 and source.read.call_count == 4
    for _ in range(5):
        assert runtime.step().accepted_line_ticks == 4
    assert source.read.call_count == 4
    assert runtime.take().chunks[0].chunk_index == 0
    assert runtime.take().chunks[0].chunk_index == 1
    rest = run_to_end(runtime)
    assert [item.chunks[0].chunk_index for item in rest] == [2, 3, 4]
    assert runtime.snapshot().delivered_line_ticks == 10


@pytest.mark.parametrize("accepted", [0, 1, 3, 8, 16])
def test_graceful_stop_preserves_every_accepted_line_including_partial(accepted):
    config = scan_config(total_lines=100, queue_capacity=4)
    source = Mock(wraps=SimulatedLineScanSource(config))
    runtime = SimulatedStreamRuntime(config, source=source)
    for _ in range(accepted):
        runtime.step()
    runtime.stop()
    output = run_to_end(runtime)
    snapshot = runtime.snapshot()
    assert snapshot.state == "stopped" and snapshot.stop_reason == "user_stop"
    assert snapshot.delivered_line_ticks == snapshot.accepted_line_ticks == accepted
    assert source.read.call_count == accepted
    assert sum(item.line_count for item in output) == accepted
    if accepted % config.chunk_height:
        assert all(chunk.partial and chunk.termination_reason == "user_stop" for chunk in output[-1].chunks)
    previous = snapshot.model_dump()
    assert runtime.stop().model_dump() == previous and runtime.step().model_dump() == previous


def test_stop_with_full_queue_is_nonblocking_and_requires_drain_before_terminal():
    runtime = SimulatedStreamRuntime(scan_config(chunk_height=2, total_lines=100))
    for _ in range(4):
        runtime.step()
    stopped = runtime.stop()
    assert stopped.state == "stopping" and stopped.pending_line_ticks == 2 and stopped.queued_bundles == 1
    output = run_to_end(runtime)
    assert len(output) == 2 and runtime.snapshot().state == "stopped"
    assert runtime.snapshot().delivered_line_ticks == 4


def test_partial_stop_flush_waits_behind_full_queue_without_overwriting_it():
    runtime = SimulatedStreamRuntime(scan_config(total_lines=100, chunk_height=8, queue_capacity=1))
    for _ in range(11):
        runtime.step()
    assert runtime.snapshot().buffered_line_ticks == 3 and runtime.snapshot().queued_bundles == 1
    status = runtime.stop()
    assert status.state == "stopping" and status.pending_line_ticks == 3
    output = run_to_end(runtime)
    assert [bundle.line_count for bundle in output] == [8, 3]
    assert all(chunk.partial and chunk.termination_reason == "user_stop" for chunk in output[-1].chunks)
    assert runtime.snapshot().delivered_line_ticks == 11


@pytest.mark.parametrize("failure", ["exception", "early_eof", "bad_metadata"])
def test_source_failure_is_visible_and_prior_accepted_partial_is_drained(failure):
    config = scan_config(chunk_height=4, total_lines=20)
    source = SimulatedLineScanSource(config)
    calls = []
    def read(index):
        calls.append(index)
        if index == 5:
            if failure == "exception":
                raise RuntimeError("fixture source fault")
            if failure == "early_eof":
                raise StopIteration("unexpected EOF")
            return source.read(index).model_copy(update={"lot_id": "other"})
        return source.read(index)
    runtime = SimulatedStreamRuntime(config, source=Mock(read=read))
    output = run_to_end(runtime)
    status = runtime.snapshot()
    assert status.state == "failed" and status.stop_reason == "source_or_builder_error" and status.error_type
    assert status.accepted_line_ticks == status.delivered_line_ticks == 5 and calls == list(range(6))
    assert [item.line_count for item in output] == [4, 1]
    assert all(chunk.termination_reason == "source_or_builder_error" for chunk in output[-1].chunks)
    assert runtime.step().state == "failed" and runtime.stop().state == "failed"


@pytest.mark.parametrize("capacity,eager", [(1, False), (1, True), (2, False), (3, True)])
def test_chunk_payload_and_metadata_are_deterministic_independent_of_queue_schedule(capacity, eager):
    expected = run_to_end(SimulatedStreamRuntime(scan_config(queue_capacity=1)), eager=True)
    actual = run_to_end(SimulatedStreamRuntime(scan_config(queue_capacity=capacity)), eager=eager)
    assert actual == expected


def test_empty_queue_is_explicit_before_and_after_stop():
    runtime = SimulatedStreamRuntime(scan_config())
    with pytest.raises(Empty):
        runtime.take()
    assert runtime.snapshot().accepted_line_ticks == 0
    runtime.stop()
    with pytest.raises(Empty):
        runtime.take()
    assert runtime.snapshot().state == "stopped"
