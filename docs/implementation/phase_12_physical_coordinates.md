# Phase 12 — Physical Coordinate Mapper + Encoder Simulation

## 1. Goal

Chuẩn bị localization cho nhà máy bằng CameraSource, EncoderSource và CoordinateMapper với implementations mô phỏng. Mapping phải có explicit calibration, không tự gọi pixel/count là tọa độ vật lý. Thực hiện ngày 2026-10-15, Asia/Bangkok. Chỉ Phase 12; Phase 13 chưa bắt đầu.

## 2. What I inspected before coding

- Yêu cầu Phase 12 và mandatory phase reporting trong pasted request; hồ sơ Phase 0–11.
- AcquisitionMetadata/InspectionInput: encoder_position hiện là optional float không có units; không thể mặc nhiên xem là encoder count hoặc mm.
- Internal context/result, InspectionService, fusion contracts/adapters, original-pixel anomaly boxes và model-grid mask descriptors.
- QC Phase 11: severity dùng model-grid area fraction, không phải physical acceptance limits. HTTP/OpenAPI/SQLite/front-end và năm golden API cases.
- Safe regression runner, cached model/FAISS artifacts và preservation fingerprints. Không có AGENTS.md áp dụng đã tìm thấy; repo không có physical mapper hoặc camera/encoder interface tương đương.
- Branch main, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; worktree có prior phases/user changes. Freeze hash315 pre-existing files; không stage/commit/reset/clean.
- A/B thực vẫn INACTIVE, active reference None. Existing architecture vẫn ResNet18 CNN layer2+3 + adapter v4 + FAISS + QC; không thay model hoặc registry.

## 3. Implementation performed

Thêm typed/frozen contracts cho PhysicalCalibration, camera/transport calibration, EncoderReading, SimulatedLinePosition, FrameCoordinates, pixel point/candidate và physical mapping result. Calibration yêu cầu version, world frame, camera IDs/dimensions, axis directions, camera offsets, origin và explicit mm-per-pixel/count/line. Không có guessed/default scale.

Pixel convention là **pixel_edges** trong EXIF-oriented source image: x trong [0,width], y trong [0,height]; pixel centre tại column+.5,row+.5; bounding boxes half-open và phải nằm trong calibrated geometry. Model-grid/raw pixel frame không được âm thầm chuyển thành oriented source pixels.

Formulas:

```text
x_mm = camera.x_origin_mm + x_direction * pixel_x * x_mm_per_pixel
encoder_anchor_mm = y_origin_mm + travel_direction * (count - encoder_origin_count) * mm_per_encoder_count + camera.y_offset_mm
line_anchor_mm = y_origin_mm + travel_direction * (line_index - line_origin_index) * mm_per_line + camera.y_offset_mm
y_mm = selected_anchor_mm + y_direction * pixel_y * y_mm_per_pixel
y_m = y_mm / 1000
```

Position anchors first-row edge của frame đã được đồng bộ. Frame index phải khớp reading; encoder ID phải khớp calibration. Encoder count và simulated line có discriminated units riêng, không suy từ AcquisitionMetadata.encoder_position.

CalibratedCoordinateMapper kiểm tra lại contracts tại mỗi mapping. Missing/invalid calibration, sai lane/encoder/geometry/frame, ngoài ảnh, nonfinite hoặc unusable numeric range trả unavailable, không kèm physical coordinates. Không clip input ngoài biên để tạo physical claim. Mapped result giữ calibration snapshot và frame/roll/lot identity.

Candidate được map thành world min/max cùng width_mm/length_mm, kể cả mirrored axes. Camera overlap tính bằng giao vùng phủ ngang có width dương; touching không coi là overlap. Các candidate ở overlap giữ IDs riêng, không deduplicate hoặc fuse. Bridge map_anomaly_boxes chỉ nhận validated FusionBox có oriented-source coordinates; raw dict/model-grid mask không được suy diễn geometry. Candidate ID scoped theo camera/frame trong output context.

CameraSource/EncoderSource/CoordinateMapper là protocols. SimulatedEncoderSource tạo count deterministically tại explicit frame index, hỗ trợ advance/reverse/stationary. SimulatedCameraSource tạo immutable RGB bytes trong RAM, giới hạn bốn triệu pixels/capture. Không file image writes, vendor SDK, hardware polling hoặc stream loop.

Implementation chạy **standalone/offline**, không thêm consumer vào upload runtime. Không cần bật/tắt feature flag mới vì hệ thống hiện tại không tự gọi mapper. Config example và CLI đều ghi rõ hypothetical; không load model hoặc activate registry.

## 4. Files created

- backend/app/core/physical_coordinates.py
- backend/app/services/coordinate_mapper.py
- backend/app/services/simulated_acquisition.py
- config/physical_calibration_example.json
- scripts/check_phase12_coordinates.py
- tests/physical_coordinate_fixtures.py
- tests/test_physical_coordinate_contracts.py
- tests/test_coordinate_mapper.py
- tests/test_simulated_acquisition.py
- tests/test_coordinate_mapping_integration.py
- docs/implementation/phase_12_physical_coordinates.md
- docs/implementation/phase_12_coordinates_report.json

## 5. Files modified

**Không sửa file có sẵn.** Cả 315 pre-existing tracked/untracked files giữ nguyên SHA-256, bao gồm toàn bộ Phase 11, old tests, Revision.md, research document và user binary.

## 6. Architecture before

Upload/API → InspectionService → CNN/FAISS evidence → legacy hoặc optional hybrid QC → HTTP/SQLite/annotations. Acquisition metadata optional, không định nghĩa encoder physical units. Model-grid mask và original-pixel anomaly boxes có frames khác nhau; chưa có physical mapper.

## 7. Architecture after

Luồng upload/API giữ nguyên. Một nhánh development độc lập:

```text
Explicit simulated camera frame + typed encoder/line position
    → explicit linear calibration + CoordinateMapper
    → mapped point/candidate in world mm (y_m available) OR unavailable
```

Offline bridge có thể map existing oriented-source anomaly boxes; không gọi lại CNN, không sửa evidence, QC, HTTP/DB hay annotations. Không physical QC policy, stream runtime hoặc PLC.

## 8. Backward compatibility

- Năm golden HTTP/SQLite cases, OpenAPI response schema, labels/reasons/confidence, scores/boxes và served annotation hashes giữ nguyên.
- Upload không gọi mapper; mapping offline không tăng encoder inference calls hoặc mutate source/calibration/QC decisions.
- AcquisitionMetadata.encoder_position vẫn giữ semantics cũ, không tự ép float sang encoder counts.
- 315 existing files không đổi; 55 protected artifacts không có mismatch. Scope hashes storage 311/Data 643/user_files 10/torch checkpoints 3 giữ nguyên.
- Không sửa .env thật, registry, DB schema, model weights, references, thresholds, datasets hoặc frontend.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| test_physical_coordinate_contracts.py — 51 cases | Explicit units, finite/strict values, calibration/geometry/sync/output guards | PASS |
| test_coordinate_mapper.py — 65 cases | Known calibration, offsets, encoder/line origins, reversed axes, boundaries, lanes/overlap, fail-closed | PASS |
| test_simulated_acquisition.py — 21 cases | Source protocols, deterministic frames/counts, resource/sync checks | PASS |
| test_coordinate_mapping_integration.py — 14 cases | Golden HTTP/SQLite/schema/annotation parity, bridge safety, QC independence, read-only CLI | PASS |
| CLI — 7 hypothetical cases | Encoder/line equivalence, boundary, two overlap candidates, missing/invalid calibration | PASS |
| Full suite — 1332 cases | All existing and Phase12 regressions | PASS |
| Frontend — 22 modules | TypeScript/Vite compatibility | PASS |
| Baseline smoke — 77 assertions | Existing cached backbone/indexes/registry and isolated API/SQLite/storage | PASS |
| compileall, whitespace, fingerprints | Syntax and preservation | PASS |

Initial new-test collection error: Pydantic không cho wildcard before field-validator trên discriminator kind của union. Đã đổi sang model-level before validation, vẫn giữ strict IDs/axis/units, rồi chạy lại toàn bộ new tests. Không sửa hoặc weaken old tests. Không có existing test bị hỏng ở full run cuối.

## 10. Full regression test result

Actual command tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Actual result: **1332 passed, 1 warning in 174.37s (0:02:54)**; frontend build PASS, 22 modules; 77 smoke assertions PASS; exit code 0. Phase 11 suite 1181 + 151 new Phase 12 cases = 1332.

Warning là FAISS loader dùng deprecated NumPy private namespace từ baseline. Runner dùng temporary DB/storage cho regression/API; smoke đọc existing cached backbone/references, không train/activate model. A/B thực vẫn INACTIVE, upload vẫn 409.

Targeted command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_physical_coordinate_contracts.py tests/test_coordinate_mapper.py tests/test_simulated_acquisition.py tests/test_coordinate_mapping_integration.py -q
```

Actual result: **151 passed, 1 warning in 6.35s**.

## 11. Hypothetical / synthetic cases tested

Example calibration: hai lanes 48×32, 0.5 mm/px ngang, 0.25 mm/px dọc, origins ngang 0/20 mm; overlap 20–24 mm. Encoder origin 1000 counts, 0.1 mm/count, world Y origin 100 mm, camera Y offset 2 mm. Count 1100 hoặc simulated line 40 với 0.25 mm/line đều cho first-row Y = 112 mm.

Point (4,8) lane_left → (2,114) mm → Y = 0.114 m. Candidate left x = 42 và right x = 2, cùng y = 8, w = 4, h = 4 → cùng rectangle x = 21–23/y = 114–115 mm nhưng vẫn là hai candidates riêng.

Đã kiểm tra camera/transport axis ±1, negative/stationary counts, nonzero line origin, image edge/centre boundaries, touching/disjoint/coincident/mirrored/three lanes, absent/invalid calibration, wrong IDs/frame/geometry, overflow/precision loss và model-grid rejection.

Ảnh camera chỉ synthetic RGB bytes; API fixtures dùng fake encoder với FAISS/HTTP/SQLite thật trong temporary sandbox. Không có real camera/encoder experiment, physical calibration residuals, industrial localization accuracy hoặc fresh model evaluation.

## 12. Known limitations

- Chỉ linear axis-aligned calibration; chưa xử lý lens distortion, perspective, skew, slip/stretch, encoder wraparound, drift hoặc trigger/latency compensation.
- Frame-index equality chỉ kiểm tra metadata consistency, không chứng minh hardware synchronization. Caller chịu trách nhiệm calibration/world origin và roll/lot alignment.
- Model-grid supervised masks chưa có verified transform sang source pixel frame, nên không map trực tiếp. Không physical output fields trong HTTP/DB/annotations.
- Overlap metadata chỉ shared horizontal coverage, không chứng minh hai detections cùng lỗi/cùng thời điểm; không dedup hoặc temporal association.
- Calibration profile hiện yêu cầu khai báo cả count scale và simulated-line scale. Không có actual calibration approval hoặc accuracy guarantee.
- Chỉ finite frame-at-index simulation, không loop/stream/backpressure/PLC. Phase 13 chưa thực hiện. Không thay model-grid QC severity bằng mm thresholds.

## 13. Risks introduced

Sai calibration/origin/direction hoặc trigger anchor có thể cho physical coordinates sai dù input hợp lệ. Model không đo và không bù sai số cơ khí/quang học. Pixel-edge convention khác pixel-centre conventions của một số SDK; phải transform đúng trước dùng thật.

Resizing/cropping/EXIF khác geometry calibration phải được khai báo/kiểm chứng, không đổi tag để bỏ guard. Metadata là trusted declaration, không content/provenance attestation. Candidate overlap không được xem là deduplicated defect count. Synthetic fixture config không phải factory calibration.

Runtime production không import/gọi new modules nên chưa đưa các rủi ro mapping này vào quyết định inspection hiện hành.

## 14. Manual verification steps for me

Tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase12_coordinates.py --demo
.\.venv\Scripts\python.exe -m pytest tests/test_physical_coordinate_contracts.py tests/test_coordinate_mapper.py tests/test_simulated_acquisition.py tests/test_coordinate_mapping_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

CLI cần case_count = 7, HYPOTHETICAL_SOFTWARE_CHECK_ONLY, real_calibration_verified=false, production_runtime_integrated=false, writes_performed=false, vendor_sdk_integrated=false, stream_runtime_started=false. Encoder/line origin đều Y = 112 mm; left/right overlap coordinates giống nhau nhưng ID riêng. Missing/invalid calibration phải unavailable và không có point/candidate physical values.

Không cần bật .env hoặc registry thật. JSON report ghi actual results lần này, không thay cho lần verification riêng của bạn.

Rollback: dừng dùng offline mapper/demo; upload runtime không có integration nên không có toggle/DB migration để rollback. Nếu rollback code, backup/remove đúng 12 files mới mục 4; không reset/clean/revert toàn worktree hoặc prior phases/user files. Chưa xóa hoặc rollback dữ liệu nào.

## 15. Git diff summary

Phase 12: **12 files mới, 0 existing files modified**, chưa stage/commit. New files untracked nên không nằm trong tracked diffstat.

Cumulative tracked git diff --stat giữ nguyên **7 files changed, 432 insertions, 101 deletions** từ trước phase; gồm prior phases và preexisting user binary. Không tính các thay đổi đó là Phase 12. Staged diff rỗng. 315 pre-existing file hashes giữ nguyên.

## 16. Recommendation

**SAFE TO APPROVE** cho explicit calibration contracts, standalone coordinate mapper, finite simulated sources và backward compatibility. Không approve factory calibration, vendor hardware, physical QC, streaming hoặc production activation. Chờ review của người dùng; Phase 13 chưa bắt đầu.

============================================================
PHASE 12 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
