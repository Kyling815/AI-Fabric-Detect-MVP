import pytest

from backend.app.core.physical_coordinates import SimulatedLinePosition
from backend.app.services.simulated_acquisition import (CameraSource, EncoderSource, SimulatedCameraConfig,
    SimulatedCameraSource, SimulatedEncoderConfig, SimulatedEncoderSource)


def encoder(step=100):
    return SimulatedEncoderSource(SimulatedEncoderConfig(encoder_id="simulated_encoder", start_count=1000, counts_per_frame=step))


def camera(**updates):
    fields = dict(camera_id="lane_left", fabric_id="fixture_fabric", image_width=48, image_height=32,
                  frame_prefix="fixture", lot_id="fixture_lot", roll_id="fixture_roll")
    fields.update(updates)
    return SimulatedCameraSource(SimulatedCameraConfig(**fields))


@pytest.mark.parametrize("index,step", [(0, 100), (1, 100), (10, 100), (2, -100), (2, 0)])
def test_encoder_source_finite_repeatable_and_supports_reverse_stationary(index, step):
    source = encoder(step)
    assert isinstance(source, EncoderSource)
    reading = source.read(index)
    assert reading == source.read(index) and reading.count == 1000 + index * step and reading.simulated


@pytest.mark.parametrize("index", [-1, True, 1.5, "1"])
def test_simulated_encoder_rejects_invalid_frame_index(index):
    with pytest.raises(ValueError):
        encoder().read(index)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_camera_source_memory_only_deterministic_and_identified(index):
    source = camera()
    assert isinstance(source, CameraSource)
    capture = source.capture(index, encoder().read(index))
    assert capture == source.capture(index, encoder().read(index))
    assert capture.simulated and capture.frame.frame_id == f"fixture_{index}"
    assert capture.frame.camera_id == "lane_left" and capture.frame.roll_id == "fixture_roll"
    assert capture.rgb_bytes == bytes([index]) * (48 * 32 * 3)


def test_camera_accepts_explicit_simulated_line_but_not_untyped_position():
    assert camera().capture(1, SimulatedLinePosition(frame_index=1, line_index=40)).frame.position.kind == "simulated_line"
    with pytest.raises(ValueError):
        camera().capture(1, 1100)


@pytest.mark.parametrize("failure", ["wrong_frame", "invalid_size", "allocation_limit", "blank_id", "bool_size"])
def test_camera_validates_alignment_geometry_and_resource_limit(failure):
    with pytest.raises(ValueError):
        if failure == "wrong_frame":
            camera().capture(1, encoder().read(2))
        else:
            camera(**{"invalid_size": {"image_width": 0}, "allocation_limit": {"image_width": 4000001, "image_height": 1},
                      "blank_id": {"camera_id": " "}, "bool_size": {"image_width": True}}[failure])


@pytest.mark.parametrize("field,value", [("start_count", True), ("counts_per_frame", "100"), ("counts_per_frame", 0.5)])
def test_encoder_configuration_has_explicit_integer_counts(field, value):
    fields = dict(encoder_id="simulated_encoder", start_count=1000, counts_per_frame=100)
    fields[field] = value
    with pytest.raises(ValueError):
        SimulatedEncoderSource(SimulatedEncoderConfig(**fields))
