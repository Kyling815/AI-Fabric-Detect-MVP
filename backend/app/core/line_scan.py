"""Four-lane simulated scan contracts. Encoder counts are not physical units."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Literal

from pydantic import Field, field_validator, model_validator

from backend.app.core.physical_coordinates import CoordinateContract, EncoderReading, FrameCoordinates


class ScanContract(CoordinateContract):
    @field_validator("simulated", mode="before", check_fields=False)
    @classmethod
    def simulation_only(cls, value):
        if value is not True:
            raise ValueError("This runtime accepts simulated data only")
        return value

    @field_validator("start_timestamp", "timestamp", "timestamp_end_exclusive", check_fields=False)
    @classmethod
    def aware_clock(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Explicit timezone-aware simulation timestamp required")
        return value.astimezone(timezone.utc)

    @field_validator("start_timestamp", "timestamp", "timestamp_end_exclusive", mode="before", check_fields=False)
    @classmethod
    def explicit_clock_units(cls, value):
        if not isinstance(value, (datetime, str)):
            raise ValueError("Use an aware datetime or ISO timestamp, not an inferred Unix numeric unit")
        return value


class ScanIdentity(ScanContract):
    simulation_id: str = Field(min_length=1, max_length=128)
    fabric_id: str = Field(min_length=1, max_length=128)
    lot_id: str = Field(min_length=1, max_length=128)
    roll_id: str = Field(min_length=1, max_length=128)


class LineScanConfig(ScanIdentity):
    camera_ids: tuple[str, str, str, str]
    image_width: int = Field(gt=0, le=4096, strict=True)
    chunk_height: int = Field(gt=0, le=2048, strict=True)
    total_lines: int = Field(ge=0, le=1_000_000, strict=True)
    encoder_id: str = Field(min_length=1, max_length=128)
    encoder_start_count: int = Field(ge=-(2 ** 63), le=2 ** 63 - 1, strict=True)
    counts_per_line: int = Field(gt=0, le=2 ** 63 - 1, strict=True)
    start_timestamp: datetime
    line_period_us: int = Field(gt=0, strict=True)
    queue_capacity: int = Field(ge=1, le=128, strict=True)  # Capacity in four-lane bundles.

    @model_validator(mode="after")
    def bounded_simulation(self):
        if (len(set(self.camera_ids)) != 4 or any(not value.strip() or value != value.strip() or len(value) > 128 for value in self.camera_ids)):
            raise ValueError("Exactly four distinct nonblank camera IDs required")
        if self.image_width * self.chunk_height * 3 * 4 * (self.queue_capacity + 2) > 64 * 1024 * 1024:
            raise ValueError("Configured queued/pending/builder RGB payload budget exceeds 64 MiB")
        if self.encoder_start_count + self.total_lines * self.counts_per_line > 2 ** 63 - 1:
            raise ValueError("Simulated encoder range exceeds signed 64-bit counts")
        try:
            self.start_timestamp + timedelta(microseconds=self.total_lines * self.line_period_us)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Simulation clock exceeds datetime range") from exc
        return self

    def timestamp_at(self, line_index):
        return self.start_timestamp + timedelta(microseconds=line_index * self.line_period_us)


class CameraLine(ScanContract):
    camera_id: str = Field(min_length=1, max_length=128)
    image_width: int = Field(gt=0, strict=True)
    rgb_bytes: bytes = Field(strict=True)

    @model_validator(mode="after")
    def one_rgb_row(self):
        if len(self.rgb_bytes) != self.image_width * 3:
            raise ValueError("RGB line payload must match exact width")
        return self


class ScanLineBundle(ScanIdentity):
    line_index: int = Field(ge=0, strict=True)
    timestamp: datetime
    encoder: EncoderReading
    rows: tuple[CameraLine, CameraLine, CameraLine, CameraLine]
    simulated: Literal[True] = True

    @model_validator(mode="after")
    def common_transport_tick(self):
        if not self.encoder.simulated or self.encoder.frame_index != self.line_index:
            raise ValueError("Simulated encoder tick must match synchronized scan line")
        if len({row.camera_id for row in self.rows}) != 4:
            raise ValueError("Duplicate scan camera row")
        return self


TerminationReason = Literal["source_exhausted", "user_stop", "source_or_builder_error"]


class LineScanChunk(ScanIdentity):
    camera_id: str = Field(min_length=1, max_length=128)
    chunk_index: int = Field(ge=0, strict=True)
    start_line: int = Field(ge=0, strict=True)
    image_width: int = Field(gt=0, strict=True)
    image_height: int = Field(gt=0, strict=True)
    nominal_height: int = Field(gt=0, strict=True)
    encoder_id: str = Field(min_length=1, max_length=128)
    encoder_start: int = Field(strict=True)
    encoder_end_exclusive: int = Field(strict=True)
    counts_per_line: int = Field(gt=0, strict=True)
    timestamp: datetime
    timestamp_end_exclusive: datetime
    line_period_us: int = Field(gt=0, strict=True)
    partial: bool = Field(strict=True)
    termination_reason: TerminationReason | None = None
    rgb_bytes: bytes = Field(strict=True)
    rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulated: Literal[True] = True

    @model_validator(mode="after")
    def complete_chunk_metadata(self):
        if self.image_height > self.nominal_height or self.partial != (self.image_height < self.nominal_height):
            raise ValueError("Partial chunk flag must match real line count")
        if self.partial and self.termination_reason is None:
            raise ValueError("Partial chunks require explicit flush reason")
        if self.encoder_end_exclusive != self.encoder_start + self.image_height * self.counts_per_line:
            raise ValueError("Encoder range must cover every chunk line")
        if self.timestamp_end_exclusive != self.timestamp + timedelta(microseconds=self.image_height * self.line_period_us):
            raise ValueError("Chunk clock range must match line interval")
        if len(self.rgb_bytes) != self.image_width * self.image_height * 3 or hashlib.sha256(self.rgb_bytes).hexdigest() != self.rgb_sha256:
            raise ValueError("Chunk RGB bytes/geometry/checksum mismatch")
        return self

    @property
    def chunk_id(self):
        identity = [self.simulation_id, self.fabric_id, self.lot_id, self.roll_id, self.camera_id, str(self.chunk_index)]
        return "scan_" + hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode("utf-8")).hexdigest()

    def to_frame_coordinates(self):
        return FrameCoordinates(fabric_id=self.fabric_id, camera_id=self.camera_id, frame_id=self.chunk_id, frame_index=self.chunk_index,
            image_width=self.image_width, image_height=self.image_height, coordinate_frame="oriented_source_pixels",
            position=EncoderReading(encoder_id=self.encoder_id, frame_index=self.chunk_index, count=self.encoder_start, simulated=True),
            lot_id=self.lot_id, roll_id=self.roll_id)

    def metadata(self):
        return {"chunk_id": self.chunk_id, **self.model_dump(mode="json", exclude={"rgb_bytes"})}


class ChunkBundle(ScanContract):
    chunks: tuple[LineScanChunk, LineScanChunk, LineScanChunk, LineScanChunk]

    @model_validator(mode="after")
    def synchronized_lanes(self):
        if len({chunk.camera_id for chunk in self.chunks}) != 4:
            raise ValueError("Exactly four distinct chunk lanes required")
        reference = self.chunks[0].model_dump(exclude={"camera_id", "rgb_bytes", "rgb_sha256"})
        if any(chunk.model_dump(exclude={"camera_id", "rgb_bytes", "rgb_sha256"}) != reference for chunk in self.chunks[1:]):
            raise ValueError("Chunk bundle metadata must share one transport/time interval")
        return self

    @property
    def line_count(self):
        return self.chunks[0].image_height


class StreamSnapshot(ScanContract):
    state: Literal["running", "stopping", "stopped", "failed"]
    last_event: Literal["ready", "progress", "enqueued", "backpressure", "stopping", "stopped", "failed"]
    stop_reason: TerminationReason | None = None
    error_type: str | None = None
    accepted_line_ticks: int = Field(ge=0, strict=True)
    buffered_line_ticks: int = Field(ge=0, strict=True)
    queued_line_ticks: int = Field(ge=0, strict=True)
    pending_line_ticks: int = Field(ge=0, strict=True)
    delivered_line_ticks: int = Field(ge=0, strict=True)
    created_bundles: int = Field(ge=0, strict=True)
    enqueued_bundles: int = Field(ge=0, strict=True)
    delivered_bundles: int = Field(ge=0, strict=True)
    queued_bundles: int = Field(ge=0, strict=True)
    queue_capacity: int = Field(ge=1, strict=True)
    backpressure_events: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def no_silent_loss_accounting(self):
        if (self.state == "running") != (self.stop_reason is None):
            raise ValueError("Stop/error lifecycle must have an explicit reason")
        if (self.error_type is not None) != (self.stop_reason == "source_or_builder_error"):
            raise ValueError("Source/builder failure must remain visible")
        if self.accepted_line_ticks != self.buffered_line_ticks + self.queued_line_ticks + self.pending_line_ticks + self.delivered_line_ticks:
            raise ValueError("Accepted scan ticks must remain buffered/pending/queued or delivered")
        if (self.enqueued_bundles != self.delivered_bundles + self.queued_bundles
                or self.created_bundles != self.enqueued_bundles + int(self.pending_line_ticks > 0)
                or self.queued_bundles > self.queue_capacity):
            raise ValueError("Queue/bundle accounting mismatch")
        if self.state in {"stopped", "failed"} and (self.buffered_line_ticks or self.queued_line_ticks or self.pending_line_ticks):
            raise ValueError("Terminal state requires graceful drain of all accepted ticks")
        return self
