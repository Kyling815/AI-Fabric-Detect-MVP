# Phase 13 — Stream / Line-Scan Simulation

## 1. Goal

Thêm simulated runtime riêng để tiến từ upload từng ảnh sang chuỗi scan/chunks của bốn camera lanes. Giữ /api/inspect nguyên vẹn, không silently drop chunks khi queue đầy. Chỉ Phase 13, ngày 2026-10-15, Asia/Bangkok; Phase 14 chưa bắt đầu.

## 2. What I inspected before coding

- Exact Phase 13 requirements, mandatory phase reporting và giới hạn Phase 14/15 trong pasted request.
- Phase 12 CameraSource/EncoderSource, deterministic encoder, explicit calibration/frame contracts và coordinate mapper.
- AcquisitionMetadata, InspectionService/contracts, fusion original-pixel/model-grid frames và QC Phase 11.
- Golden API/OpenAPI/SQLite/annotation fixtures, safe baseline runner và Phase 12 report (1332 tests).
- Repo/parent AGENTS.md checks, existing stream files/interfaces: không có AGENTS.md áp dụng hoặc stream/chunk runtime tương đương tìm thấy.
- Branch main, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; hash 327 pre-existing tracked/untracked files trước sửa. Worktree có prior phases/user changes; không stage/commit/reset/clean.
- Existing A/B registry vẫn INACTIVE, active references None. CNN/FAISS backbone, adapter, references và QC không cần sửa để mô phỏng transport.

## 3. Implementation performed

Tạo frozen/typed LineScanConfig, synchronized ScanLineBundle, LineScanChunk, ChunkBundle và StreamSnapshot. Config bắt buộc simulation/fabric/lot/roll/encoder IDs, đúng bốn distinct camera IDs, width/chunk height, finite line count, integer encoder counts, explicit aware clock và queue capacity. Count range signed 64-bit, forward-only counts_per_line > 0; từ chối naive/numeric timestamps và invalid/resource-exceeding config trước cấp phát.

SimulatedEncoder tái sử dụng Phase 12 generator, mỗi index là một scan line. SimulatedLineScanSource sinh bốn RGB rows khác nhau tại cùng virtual timestamp và encoder tick. Timestamp = explicit origin + line_index × line_period_us, chuẩn hóa UTC; không đọc wall clock, sleep hoặc camera SDK.

ChunkBuilder nhận đúng thứ tự từ line 0, kiểm tra identity/time/count/camera order/width/payload cho toàn bộ bundle trước commit. Ghép liên tiếp mỗi H rows thành bốn camera chunks. Chunk metadata chứa fabric_id, lot_id, roll_id, camera_id, encoder_start/end_exclusive, timestamp/end_exclusive, start_line, chunk_index, dimensions, RGB SHA-256 và deterministic ID. Partial flush ghi chiều cao thực và termination reason, không padding hoặc drop tail.

SimulatedStreamRuntime là **cooperative driver**: step() tiến tối đa một source tick, take() lấy một atomic bundle gồm bốn camera chunks. Không background worker/inference scheduler. Queue có capacity theo bundle; nếu full, giữ tối đa một pending bundle riêng và báo backpressure. Retry bị block không đọc thêm source, không drop-oldest/newest hoặc overwrite pending.

stop() ngưng acquisition, flush accepted partial rows và chuyển stopping; consumer tiếp tục take() để drain đến stopped. Source/builder error được báo rõ reason/error_type, không skip lỗi để tiếp tục; accepted prefix vẫn flush/drain rồi terminal failed. take() chuyển ownership cho caller, **không phải** inference completion hoặc durable ACK.

Accounting được kiểm tra tại mỗi snapshot:

```text
accepted ticks = buffered + pending + queued + delivered ticks
enqueued bundles = queued + delivered bundles
created bundles = enqueued bundles + retained pending bundle
```

Một tick là bốn camera rows đồng bộ; counter 19 nghĩa là 19 rows mỗi camera, tức tổng 76 camera rows. Payload budget nominal RGB <=64 MiB có allowance cho queue/pending/builder; không phải measured process RSS.

Standalone runtime không được gọi bởi upload/API. Không sửa existing files, .env, registry, models, DB, frontend hoặc thresholds. Không thực hiện seam merging, physical QC hoặc Phase 15 profiling/scheduling.

## 4. Files created

- backend/app/core/line_scan.py
- backend/app/services/line_scan_source.py
- backend/app/services/line_scan_runtime.py
- config/line_scan_simulation_example.json
- scripts/check_phase13_stream.py
- tests/line_scan_fixtures.py
- tests/test_line_scan_contracts.py
- tests/test_line_scan_source_builder.py
- tests/test_line_scan_runtime.py
- tests/test_line_scan_integration.py
- docs/implementation/phase_13_line_scan_simulation.md
- docs/implementation/phase_13_stream_report.json

## 5. Files modified

**Không sửa file có sẵn.** Hash của cả 327 pre-existing files giữ nguyên, gồm prior phases, existing tests, Revision.md, research report và user binary.

## 6. Architecture before

Upload → InspectionService → CNN/FAISS → optional fusion/QC → response/SQLite. Phase 12 chỉ có finite frame-at-index simulation và standalone coordinate mapping; chưa có scan-line assembly, stream queue hoặc stop/drain lifecycle.

## 7. Architecture after

Upload path giữ nguyên. Nhánh mô phỏng riêng:

```text
Four synchronized synthetic rows + virtual encoder/clock
    → atomic ChunkBuilder → four-camera ChunkBundle
    → bounded FIFO + retained pending bundle → explicit take() consumer
```

Full queue → backpressure, giữ ownership; stop/error → flush partial + drain. Không auto-inference hoặc automatic persistence. Chunk.to_frame_coordinates() tạo oriented-source pixel context với encoder anchor ở first-row edge để dùng Phase 12 khi có calibration phù hợp; không tự suy ra mm.

## 8. Backward compatibility

- Năm golden HTTP/SQLite cases, OpenAPI schema, labels/reasons/confidence/scores/boxes và served annotation hashes giữ nguyên.
- Upload không gọi stream runtime; chạy stream offline không tăng CNN encoder calls hoặc tạo inspection records.
- Phase 12 mapping vẫn require explicit calibration; partial chunk không đúng calibrated height trả unavailable, không âm thầm đổi geometry.
- 327 existing file hashes giữ nguyên; 55 protected artifacts không có mismatch. Scope hashes storage 311/Data 643/user_files 10/torch checkpoints 3 giữ nguyên.
- Không inference/model quality changes, activation, registry/reference rewrite hoặc database migration.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| test_line_scan_contracts.py — 48 cases | Strict IDs/units/clock/resources, synchronized intervals/checksum, accounting guards | PASS |
| test_line_scan_source_builder.py — 39 cases | Four lanes, encoder/virtual clock, ordered rows, atomic validation, partial flush | PASS |
| test_line_scan_runtime.py — 22 cases | No silent drop, queue full/retry, deterministic schedules, stop/drain and visible source failures | PASS |
| test_line_scan_integration.py — 8 cases | Golden API/SQLite/images, Phase12 calibration bridge, deterministic read-only CLI | PASS |
| CLI — 19 ticks / 12 camera chunks | Delayed consumer/backpressure, final partial rows, complete accounting | PASS |
| Full suite — 1449 cases | All existing and new regressions | PASS |
| Frontend — 22 modules | TypeScript/Vite compatibility | PASS |
| Baseline smoke — 77 assertions | Cached model/index/registry plus isolated synthetic API/SQLite/storage | PASS |
| compileall, diff/whitespace, fingerprints | Syntax and preservation | PASS |

Các targeted runs đều pass; không có new test failure cần workaround. Không sửa hoặc weaken existing tests. Warning FAISS/NumPy đã tồn tại từ baseline, không phải lỗi stream.

## 10. Full regression test result

Actual command tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Actual result: **1449 passed, 1 warning in 115.33s (0:01:55)**. Frontend build PASS, 22 modules; 77 smoke assertions PASS; process exit code 0. Previous 1332 + 117 new Phase 13 tests = 1449. Không có existing test bị hỏng.

Runner dùng temporary DB/storage cho regression/API. Smoke đọc existing cached CNN backbone/references, không train model hoặc activate fabric. Actual A/B vẫn INACTIVE và upload vẫn 409. Warning là FAISS dùng deprecated NumPy private namespace.

Targeted command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_line_scan_contracts.py tests/test_line_scan_source_builder.py tests/test_line_scan_runtime.py tests/test_line_scan_integration.py -q
```

Actual result: **117 passed, 1 warning in 4.37s**.

## 11. Hypothetical / synthetic cases tested

Example: bốn lanes 48 px, chunk height 8, total 19 scan ticks, encoder origin 1000 và 2 counts/line, virtual clock bắt đầu 2026-10-15T09:00:00+07:00, interval 1000 µs. Đây là virtual interval, không phải measured 1000 FPS capability.

Mỗi lane tạo chunks heights 8/8/3, start lines 0/8/16, encoder ranges [1000,1016), [1016,1032), [1032,1038). Tổng 3 atomic bundles = 12 camera chunks; delivered 19 ticks mỗi lane = 76 camera rows. Demo cố ý chậm consumer để tạo backpressure; cuối cùng state stopped, buffered/pending/queued=0.

Đã kiểm tra zero/one/full/partial sources, queue capacities khác nhau, eager/delayed consumer, repeated blocked retries, stop trước capture/giữa chunk/queue full, source exception/early EOF/bad metadata và deterministic output độc lập consumption schedule. Source errors giữ accepted prefix và terminal failed, không giả successful completion.

Tất cả pixels/counts/timestamps là synthetic. Golden API fixtures dùng fake CNN encoder với FAISS/HTTP/SQLite thật trong sandbox. Không real camera/encoder acquisition, hardware synchronization validation, fresh accuracy hoặc industrial throughput benchmark.

## 12. Known limitations

- Cooperative in-memory simulation; caller phải pump step/take, không real-time capture threads, async inference, GPU scheduling hoặc backpressure tới camera thật.
- Queue không durable: không crash recovery/replay/consumer ACK/exactly-once inference. Sau take(), caller chịu trách nhiệm xử lý bundle; runtime chỉ bảo toàn ownership trước delivery.
- Stop/error giữ những dòng đã accepted; configured lines chưa acquire không được coi là delivered. Caller phải drain trước khi chờ terminal stopped/failed.
- Uniform lane width, đúng bốn lanes, forward counts và finite source. Virtual synchronization không chứng minh timestamp/trigger alignment ngoài nhà máy.
- Chunk IDs deterministic theo simulation/camera/chunk context; simulation_id phải định danh đúng run configuration, không phải global production idempotency key.
- Không seam/duplicate merging (Phase 14), profiling/async scheduler (Phase 15), physical QC hoặc nối vào current inference API.
- Physical mapping cần calibration/row pitch/count scale coherent và exact chunk geometry. Partial-height chunk không tự được cấp calibration của full-height frame.

## 13. Risks introduced

Consumer bỏ qua backpressure hoặc không drain sau stop có thể làm simulation đứng ở stopping; đây là explicit retained work, không silently dropped chunks. Queue capacity tính theo bundle bốn cameras, cộng một pending bundle nên không được hiểu là tổng số single-camera chunks đang giữ trong RAM.

Metadata sai từ source bị reject atomically và dừng stream, nhưng validated synthetic metadata không phải proof về hardware hoặc image provenance. Virtual line period không dùng để claim factory performance. No-silent-drop accounting không thay thế durable delivery khi process/caller crash.

Không có production consumer/import mới nên chưa đưa simulation vào quyết định inspection hiện hành.

## 14. Manual verification steps for me

Tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase13_stream.py --demo
.\.venv\Scripts\python.exe -m pytest tests/test_line_scan_contracts.py tests/test_line_scan_source_builder.py tests/test_line_scan_runtime.py tests/test_line_scan_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

CLI cần camera_count=4, bundle_count=3, chunk_count=12; accepted_line_ticks=delivered_line_ticks=19, backpressure_events>0, cuối state stopped và pending/queued/buffered=0. production_api_integrated/models_trained_or_loaded/vendor_sdk_integrated/physical_calibration_inferred/writes_performed/seam_merging_implemented đều false.

stop() dừng acquisition trước, không tự discard queue: tiếp tục take() để drain. Nếu đang chờ explicit driver, không chờ thread/join vì không có background worker. Tests queue-full/partial-stop/source-error minh họa lifecycle đầy đủ.

Không cần đổi .env hoặc registry thật. CLI không persist business images/chunks/DB/registry; Python cache và frontend ignored dist có thể được tạo bởi verification như bình thường. JSON report lưu lần chạy này, không thay cho verification riêng của bạn.

Rollback: dừng gọi standalone runtime/demo; không có production toggle/DB migration phải undo. Code rollback chỉ backup/remove đúng 12 files mới mục 4; không reset/clean prior phases/user work. Không thực hiện deletion/rollback dữ liệu nào.

## 15. Git diff summary

Phase 13: **12 files mới, 0 existing files modified**; chưa stage/commit. 327 pre-existing file hashes giữ nguyên. New files untracked nên không xuất hiện trong tracked diffstat.

Cumulative tracked git diff --stat giữ nguyên **7 files changed, 432 insertions, 101 deletions** từ prior phases/preexisting user binary; không tính đó là Phase 13 changes. Staged diff rỗng.

## 16. Recommendation

**SAFE TO APPROVE** cho standalone four-lane scan simulation, atomic chunks, bounded queue/backpressure và graceful stop/accounting. Không approve production camera integration, industrial throughput, durable transport hoặc inference scheduling. Chờ user review; Phase 14 chưa bắt đầu.

============================================================
PHASE 13 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
