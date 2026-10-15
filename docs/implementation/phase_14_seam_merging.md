# Phase 14 — Multi-Camera Seam / Duplicate Merging

## 1. Goal

Xử lý candidates cùng lỗi xuất hiện ở vùng overlap của camera liền kề bằng SeamMerger. Giữ toàn bộ source lineage, không gộp class xung đột hoặc suy ra tọa độ vật lý khi thiếu calibration. Thực hiện ngày 2026-10-15, Asia/Bangkok; chỉ Phase 14, chưa bắt đầu Phase 15.

## 2. What I inspected before coding

- Exact Phase 14 requirements, mandatory report format và Phase 15 boundary trong pasted request.
- PhysicalCalibration, pixel-edge/oriented-source contracts, Phase 12 mapper và shared-FOV calculation; không suy physical coordinates từ model-grid masks.
- Phase 13 synchronized four-lane ChunkBundle, metadata/clock/encoder lifecycle, source candidate IDs và stream tests.
- Current inspection/QC contracts, golden HTTP/OpenAPI/SQLite/image fixtures và safe baseline runner. Previous suite 1449 tests.
- Repo/parent AGENTS.md và existing seam implementations: không có instructions áp dụng hoặc SeamMerger tương đương đã tìm thấy.
- Branch main, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; freeze hash 339 existing tracked/untracked files. Prior phases/user work còn dirty; không stage/commit/reset/clean.
- Existing A/B registry vẫn INACTIVE, active reference None. Không cần sửa CNN/FAISS, adapter, references hoặc business rules để triển khai grouping offline.

## 3. Implementation performed

Tạo frozen SeamPolicy, SeamObservation, SeamPairAssessment, SeamGroup và SeamMergeResult. Policy yêu cầu explicit camera adjacency khi muốn merge; mặc định adjacent_camera_pairs rỗng nên không tự đoán từ tên/order camera. Example dùng IoU >=0.5, clipped-center distance <=1.0 mm và time delta <=0 ms; đây là unapproved starting matching tolerances, không phải calibrated factory/QC thresholds.

Observation chứa source pixel candidate, frame/roll/lot/fabric identity, capture_group_id, aware timestamp, source_kind và declared model/hash/reference provenance. Anomaly branch cần reference version. Raw scores được giữ nguyên, không normalize thành probability, không chọn highest-confidence representative hoặc average scores.

SeamMerger tự gọi Phase 12 mapper từ source pixels với một calibration snapshot; không tin một physical bbox tùy ý do caller khai báo. Pair chỉ đủ điều kiện khi:

1. Khác camera và nằm trong declared adjacent pair có calibrated shared FOV dương.
2. Cùng capture group, fabric, verified lot/roll, frame index và typed encoder/line position; timestamps nằm trong explicit tolerance. Không mix real/simulated position declarations.
3. Hai physical bboxes đều giao vùng seam. Tính bbox IoU và center distance trên phần được clip vào shared horizontal FOV; cả hai ngưỡng phải đạt.
4. source_kind, class label và declared model version/hash/reference nhất quán.

Near-but-separate/touching boxes không merge chỉ vì proximity. Mép FOV có thể truncated một phía: matching dùng seam clip, nhưng output giữ envelope của full source boxes. Missing/invalid calibration trả unavailable với raw candidates giữ riêng, không có fabricated physical bounds. Numeric geometry overflow cũng giữ riêng để review.

Graph matching dùng conservative **complete link**: mỗi camera chỉ có một member, mọi member pair đều eligible và không có incident conflict/review warnings. One-to-many hoặc chain chỉ nối gián tiếp không được greedy/transitive merge; giữ từng observation và đánh dấu ambiguous. Complete clique nhiều camera có thể merge khi mọi cặp adjacency/geometry/identity đều được xác nhận.

Output giữ mọi accepted observation đúng một lần, các mapping riêng, full pair audit và deterministic group IDs. Group bbox là **envelope, không phải mask union hoặc defect area**. Diagnostic review_required không phải business QC; affects_qc luôn false.

Thêm explicit seam_observations_from_chunks bridge cho Phase 13: validate synchronized ChunkBundle/checksums, tạo shared capture-group ID và first-row frame context; không chạy detector. Unknown camera keys, raw/model-grid candidates hoặc over-limit input bị reject, không silently filter. Partial-height chunk vẫn phải qua exact calibration geometry guard của Phase 12.

Standalone/offline only: không thêm consumer/flag vào API hoặc stream runtime hiện hành. Không model training/activation mới, physical QC, mask stitching hoặc Phase 15 profiling/scheduler.

## 4. Files created

- backend/app/core/seam.py
- backend/app/services/seam_merger.py
- config/seam_merging_example.json
- scripts/check_phase14_seam.py
- tests/seam_fixtures.py
- tests/test_seam_contracts.py
- tests/test_seam_merger.py
- tests/test_seam_integration.py
- docs/implementation/phase_14_seam_merging.md
- docs/implementation/phase_14_seam_report.json

## 5. Files modified

**Không sửa file có sẵn.** SHA-256 của cả 339 pre-existing files giữ nguyên, gồm Phase 0–13, old tests, Revision.md, research report và user binary.

## 6. Architecture before

Upload → CNN/FAISS → optional fusion/QC → HTTP/SQLite. Phase 12 mapping giữ separate physical candidates; Phase 13 có synchronized chunks nhưng chưa xử lý duplicate candidates tại seam. No automatic multi-camera inference.

## 7. Architecture after

Upload/runtime production giữ nguyên. Nhánh development riêng:

```text
Explicit oriented-source candidates + synchronized capture metadata
    → Phase 12 calibrated mapping + declared camera adjacency
    → pair seam-IoU/proximity/class/provenance checks
    → complete-link unambiguous groups OR separate reviewable originals
```

Phase 13 chunks có thể đi qua explicit offline adapter. Không đổi API defect_count, annotations hoặc QC theo group count; không model-grid mask union hoặc temporal tracking.

## 8. Backward compatibility

- Năm golden HTTP/SQLite cases và OpenAPI response schema giữ nguyên; upload không gọi SeamMerger.
- Standalone merge không gọi lại CNN, không đổi scores/boxes/annotation image hashes hoặc business QC decisions.
- Phase 12 calibration guards và Phase 13 immutable chunk payload/IDs giữ nguyên. Missing/invalid/partial geometry không bị bypass để merge.
- 339 existing file hashes không đổi; 55 protected artifacts không có mismatch. Scope fingerprints storage 311/Data 643/user_files 10/torch checkpoints 3 giữ nguyên.
- Không sửa .env, real registry, DB schema, frontend, source data, weights, thresholds hoặc reference artifacts.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| test_seam_contracts.py — 35 cases | Strict policy/identity/time/frames, no fake QC/lineage/envelope/audit | PASS |
| test_seam_merger.py — 55 cases | Adjacent duplicates, separate/conflicting/edge cases, calibration, sync/provenance, ambiguity, numeric safety, permutations | PASS |
| test_seam_integration.py — 14 cases | Golden HTTP/SQLite/images, Phase12/13 bridge guards, QC independence, read-only CLI | PASS |
| CLI — 6 hypothetical scenarios | Required five scenarios plus one-to-many ambiguity | PASS |
| Full suite — 1553 cases | Existing and Phase14 regressions | PASS |
| Frontend — 22 modules | TypeScript/Vite compatibility | PASS |
| Baseline smoke — 77 assertions | Cached CNN/index/registry and isolated synthetic API/SQLite/storage | PASS |
| compileall, whitespace, fingerprints | Syntax and preservation | PASS |

Các targeted runs đều pass; không có failure cần workaround. Không sửa hoặc weaken existing tests. Warning FAISS/NumPy là warning từ baseline, không phải lỗi seam merging.

## 10. Full regression test result

Actual command tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Actual result: **1553 passed, 1 warning in 119.60s (0:01:59)**; frontend build PASS, 22 modules; 77 smoke assertions PASS; exit code 0. Previous 1449 + 104 new Phase 14 cases = 1553. Không existing test nào bị hỏng.

Regression/API dùng temporary DB/storage. Smoke đọc cached actual backbone/references; không activate production model/fabric hoặc thay registry. A/B thực vẫn INACTIVE, upload vẫn 409. Warning là FAISS loader dùng deprecated NumPy private namespace.

Targeted command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_seam_contracts.py tests/test_seam_merger.py tests/test_seam_integration.py -q
```

Actual result: **104 passed, 1 warning in 4.59s**.

## 11. Hypothetical / synthetic cases tested

Demo dùng Phase 12 hypothetical calibration: lane_left phủ x0–24 mm, lane_right phủ x20–44 mm, seam x20–24 mm. Candidate left x42 và right x2 cùng size4×4 source pixels → cùng physical bbox x21–23/y114–115 mm → một group, vẫn giữ hai raw observations.

Hai bbox chạm nhau nhưng không có area overlap → separate. Classes defect/scratch ở cùng vị trí → separate reviewable groups. Mép FOV left x46,w2 và right x6,w4 → seam IoU1, output envelope x23–25 mm, không tuyên bố mask/area union. No calibration → hai originals, không physical bbox. One-to-many → ba originals, ambiguous review.

Thêm tests explicit timestamp boundaries/timezone equivalence, encoder/position mismatch, missing roll/lot, model/reference conflict, same-camera candidates, mirrored/touching/disjoint FOVs, two separate defect pairs, complete clique vs transitive three-camera chain, input permutations và overflow geometry.

Tất cả matches là software fixtures/synthetic declarations, không real defect predictions hoặc verified calibration. API fixtures dùng fake CNN encoder nhưng FAISS/HTTP/SQLite thật trong sandbox. Không có real-data duplicate precision/recall, fresh accuracy hoặc industrial throughput claim.

## 12. Known limitations

- Matching chỉ dựa trên calibrated axis-aligned bboxes; chưa mask stitching/union, lens/perspective/slip/latency compensation hoặc feature-based re-identification.
- Starting IoU/distance/time tolerances chưa calibrated. Similar nearby defects vẫn có thể false-merge nếu geometry/class/metadata indistinguishable; raw lineage giữ lại để audit, không chứng minh accuracy.
- Exact provenance/count/frame identity có thể giữ lại real duplicates qua delayed/heterogeneous cameras. Chain không complete hoặc one-to-many giữ riêng, có thể còn double counts cho review.
- Caller capture IDs, version/hash và transport/time metadata không phải hardware/content/provenance attestation.
- Không same-camera tile NMS hoặc rolling temporal tracking. Group IDs deterministic với unchanged policy/calibration/inputs, không global production idempotency guarantee.
- Batch list/tuple bounded: default 128, hard max 256; invalid/duplicate/over-limit input raise explicit error, không partial filtering/truncation. Pair matching O(N²), chưa profiling throughput.
- Phase 13 bridge là explicit call, không automatic stream consumer; partial chunks cần calibration geometry phù hợp. Không physical QC thresholds hoặc Phase 15 implementation.

## 13. Risks introduced

Sai calibration/adjacency hoặc ambiguous detections có thể làm grouping sai. Group envelope có thể chứa vùng normal giữa các source boxes; tuyệt đối không dùng như defect-area measurement hoặc thay model-grid QC severity.

Diagnostic review_required=false chỉ nghĩa là không bị guard matching chặn, không phải GOOD/production acceptance. Không dùng grouped count để đổi API defect_count/QC trước real-data validation. Exact metadata gates có chủ đích tăng separate/review cases thay vì suppression thiếu chứng cứ.

Production path chưa import/gọi merger nên hiện không chịu inference/decision changes từ phase này.

## 14. Manual verification steps for me

Tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase14_seam.py --demo
.\.venv\Scripts\python.exe -m pytest tests/test_seam_contracts.py tests/test_seam_merger.py tests/test_seam_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

CLI cần case_count=6, HYPOTHETICAL_SOFTWARE_CHECK_ONLY; same_defect_adjacent và edge_of_fov có merged group, nearby/conflicting/no-calibration/ambiguous giữ riêng. No calibration không có physical bounds. affects_qc=false; production_runtime_integrated/models_trained_or_loaded/real_calibration_verified/mask_union_implemented/qc_changed/writes_performed đều false.

Không cần đổi .env hoặc real registry. CLI không persist business images/chunks/DB/registry; Python cache và ignored frontend dist có thể được tạo bởi verification. JSON report ghi lần chạy này, không thay verification riêng của bạn.

Rollback: ngừng gọi standalone merger/bridge/demo. Không có production toggle hoặc DB migration phải undo. Code rollback chỉ backup/remove đúng 10 files mới mục 4, giữ prior phases/user work; không reset/clean toàn worktree. Không thực hiện deletion/rollback dữ liệu nào.

## 15. Git diff summary

Phase 14: **10 files mới, 0 existing files modified**, chưa stage/commit. 339 pre-existing file hashes giữ nguyên. New files untracked nên không nằm trong tracked diffstat.

Cumulative tracked git diff --stat giữ nguyên **7 files changed, 432 insertions, 101 deletions** từ prior phases/preexisting user binary; không tính đó là Phase 14 changes. Staged diff rỗng.

## 16. Recommendation

**SAFE TO APPROVE** cho standalone conservative seam grouping, exact lineage/calibration guards và backward compatibility. Không approve industrial deduplication accuracy, mask fusion, production QC hoặc activation. Chờ user review; Phase 15 chưa bắt đầu.

============================================================
PHASE 14 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
