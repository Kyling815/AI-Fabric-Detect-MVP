"""Deterministic four-lane virtual-clock source; no SDK, wall-clock sleep or disk IO."""

from typing import Protocol

from backend.app.core.line_scan import CameraLine, LineScanConfig, ScanLineBundle
from backend.app.services.simulated_acquisition import SimulatedEncoderConfig, SimulatedEncoderSource


class SimulatedEncoder(SimulatedEncoderSource):
    """Reuse Phase 12 count generator, interpreting each input index as one scan line."""

    def __init__(self, config: LineScanConfig):
        config = LineScanConfig.model_validate(config.model_dump())
        super().__init__(SimulatedEncoderConfig(encoder_id=config.encoder_id, start_count=config.encoder_start_count,
                                               counts_per_frame=config.counts_per_line))


class LineScanSource(Protocol):
    def read(self, line_index: int) -> ScanLineBundle: ...


class SimulatedLineScanSource:
    def __init__(self, config: LineScanConfig):
        self.config = LineScanConfig.model_validate(config.model_dump())
        self.encoder = SimulatedEncoder(self.config)

    def read(self, line_index: int) -> ScanLineBundle:
        if type(line_index) is not int or line_index < 0:
            raise ValueError("Explicit nonnegative integer line index required")
        if line_index >= self.config.total_lines:
            raise StopIteration("Configured simulated source exhausted")
        config = self.config
        rows = tuple(CameraLine(camera_id=camera, image_width=config.image_width,
            rgb_bytes=bytes([(line_index + lane * 53) % 256, (line_index * 3 + lane * 17) % 256, lane * 61 % 256]) * config.image_width)
            for lane, camera in enumerate(config.camera_ids))
        identity = config.model_dump(include={"simulation_id", "fabric_id", "lot_id", "roll_id"})
        return ScanLineBundle(**identity, line_index=line_index, timestamp=config.timestamp_at(line_index),
                              encoder=self.encoder.read(line_index), rows=rows)
