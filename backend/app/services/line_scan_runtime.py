"""Cooperative stream simulator. Bounded atomic four-lane delivery, no silent drops.

step() advances at most one source tick; take() transfers a whole chunk bundle.
No background worker, inference scheduler or durable delivery is implied.
"""

import hashlib
from queue import Full, Queue
import threading

from backend.app.core.line_scan import ChunkBundle, LineScanChunk, LineScanConfig, ScanLineBundle, StreamSnapshot
from backend.app.services.line_scan_source import LineScanSource, SimulatedLineScanSource


class ChunkBuilder:
    def __init__(self, config: LineScanConfig):
        self.config = LineScanConfig.model_validate(config.model_dump())
        self._lines = []
        self.next_line = 0
        self.chunk_index = 0
        self.closed = False

    @property
    def buffered_line_ticks(self):
        return len(self._lines)

    def push(self, line: ScanLineBundle) -> ChunkBundle | None:
        if self.closed:
            raise ValueError("ChunkBuilder already finished")
        line = ScanLineBundle.model_validate(line.model_dump())
        config = self.config
        identity = ("simulation_id", "fabric_id", "lot_id", "roll_id")
        if (line.line_index != self.next_line or line.line_index >= config.total_lines
                or any(getattr(line, key) != getattr(config, key) for key in identity)
                or line.timestamp != config.timestamp_at(line.line_index)
                or line.encoder.encoder_id != config.encoder_id
                or line.encoder.count != config.encoder_start_count + line.line_index * config.counts_per_line
                or tuple(row.camera_id for row in line.rows) != config.camera_ids
                or any(row.image_width != config.image_width for row in line.rows)):
            raise ValueError("Out-of-order or unsynchronized scan bundle")
        pending = self._lines + [line]
        completed = self._build(pending, None) if len(pending) == config.chunk_height else None
        self.next_line += 1  # Commit only after all four lanes and a completed bundle validate.
        self._lines = [] if completed is not None else pending
        self.chunk_index += int(completed is not None)
        return completed

    def _build(self, lines, reason):
        config, first, count = self.config, lines[0], len(lines)
        identity = config.model_dump(include={"simulation_id", "fabric_id", "lot_id", "roll_id"})
        chunks = []
        for lane, camera in enumerate(config.camera_ids):
            rgb = b"".join(line.rows[lane].rgb_bytes for line in lines)
            chunks.append(LineScanChunk(**identity, camera_id=camera, chunk_index=self.chunk_index, start_line=first.line_index,
                image_width=config.image_width, image_height=count, nominal_height=config.chunk_height, encoder_id=config.encoder_id,
                encoder_start=first.encoder.count, encoder_end_exclusive=first.encoder.count + count * config.counts_per_line,
                counts_per_line=config.counts_per_line, timestamp=first.timestamp,
                timestamp_end_exclusive=config.timestamp_at(first.line_index + count), line_period_us=config.line_period_us,
                partial=count < config.chunk_height, termination_reason=reason, rgb_bytes=rgb,
                rgb_sha256=hashlib.sha256(rgb).hexdigest()))
        return ChunkBundle(chunks=tuple(chunks))

    def finish(self, reason="user_stop") -> ChunkBundle | None:
        if type(reason) is not str or reason not in {"source_exhausted", "user_stop", "source_or_builder_error"}:
            raise ValueError("Explicit supported stream termination reason required")
        if self.closed:
            return None
        result = self._build(self._lines, reason) if self._lines else None
        self._lines = []
        self.closed = True
        self.chunk_index += int(result is not None)
        return result


class SimulatedStreamRuntime:
    def __init__(self, config: LineScanConfig, *, source: LineScanSource | None = None):
        self.config = LineScanConfig.model_validate(config.model_dump())
        self.source = source if source is not None else SimulatedLineScanSource(self.config)
        self.builder = ChunkBuilder(self.config)
        self._queue = Queue(maxsize=self.config.queue_capacity)
        self._lock = threading.RLock()
        self._pending = None
        self._next_line = self._created = self._enqueued = self._delivered = 0
        self._queued_lines = self._delivered_lines = self._backpressure = 0
        self._state, self._event = "running", "ready"
        self._reason = self._error_type = None

    def snapshot(self) -> StreamSnapshot:
        with self._lock:
            return StreamSnapshot(state=self._state, last_event=self._event, stop_reason=self._reason, error_type=self._error_type,
                accepted_line_ticks=self._next_line, buffered_line_ticks=self.builder.buffered_line_ticks,
                queued_line_ticks=self._queued_lines, pending_line_ticks=self._pending.line_count if self._pending else 0,
                delivered_line_ticks=self._delivered_lines, created_bundles=self._created, enqueued_bundles=self._enqueued,
                delivered_bundles=self._delivered, queued_bundles=self._queue.qsize(), queue_capacity=self.config.queue_capacity,
                backpressure_events=self._backpressure)

    def _publish(self):
        if self._pending is not None:
            try:
                self._queue.put_nowait(self._pending)
            except Full:
                self._backpressure += 1
                self._event = "backpressure"
                return False
            self._queued_lines += self._pending.line_count
            self._enqueued += 1
            self._pending = None
            self._event = "enqueued"
        return True

    def _settle(self):
        if self._reason is not None and self._pending is None and self._queue.empty() and self.builder.closed:
            self._state = "failed" if self._error_type is not None else "stopped"
            self._event = self._state

    def _request_stop(self, reason, error_type=None):
        if self._reason is None:
            self._reason, self._error_type = reason, error_type
            self._state, self._event = "stopping", "stopping"
            partial = self.builder.finish(reason)
            if partial is not None:
                if self._pending is not None:
                    raise RuntimeError("Cannot overwrite an undelivered pending chunk bundle")
                self._pending = partial
                self._created += 1
        self._publish()
        self._settle()

    def step(self) -> StreamSnapshot:
        with self._lock:
            if self._state in {"stopped", "failed"}:
                return self.snapshot()
            if self._reason is not None:
                self._publish()
                self._settle()
                return self.snapshot()
            if self._pending is not None:
                self._publish()  # A blocked retry never reads or skips another source tick.
                return self.snapshot()
            if self._next_line >= self.config.total_lines:
                self._request_stop("source_exhausted")
                return self.snapshot()
            try:
                bundle = self.builder.push(self.source.read(self._next_line))
            except Exception as exc:
                self._request_stop("source_or_builder_error", type(exc).__name__)
                return self.snapshot()
            self._next_line += 1
            self._event = "progress"
            if bundle is not None:
                self._pending = bundle
                self._created += 1
                self._publish()
            if self._next_line == self.config.total_lines:
                self._request_stop("source_exhausted")
            return self.snapshot()

    def take(self) -> ChunkBundle:
        with self._lock:
            bundle = self._queue.get_nowait()  # Empty is explicit; no blocking consumer/deadlock on stop.
            self._queue.task_done()
            self._queued_lines -= bundle.line_count
            self._delivered_lines += bundle.line_count
            self._delivered += 1
            self._publish()
            self._settle()
            return bundle

    def stop(self) -> StreamSnapshot:
        with self._lock:
            self._request_stop("user_stop")
            return self.snapshot()
