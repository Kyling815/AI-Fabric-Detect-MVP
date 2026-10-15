# Phase 9 — Supervised inference shadow mode

Ngày kiểm tra: 2026-10-14, Asia/Bangkok. Chỉ Phase 9; chưa bắt đầu fusion Phase 10.

## 1. Goal

Cho supervised segmentation chạy như diagnostic shadow bên cạnh anomaly branch, mặc định OFF. Ghi model version/output nhưng không dùng nó để thay GOOD/NEEDS_REVIEW/BAD, localization, annotated image hoặc dữ liệu inspection hiện hữu.

## 2. What I inspected before coding

- Attachment yêu cầu Phase 9 và báo cáo Phase 0–8; baseline trước phase 725 tests. Không có model segmentation production-ready: Phase 8 chỉ validate infrastructure trên tiny synthetic fixtures.
- Branch `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`. Worktree dirty từ các phase trước/file người dùng; không có AGENTS.md áp dụng trong repo/các thư mục cha đã kiểm tra. Không stage/commit hoặc ghi đè unrelated changes.
- `/api/inspect` → InspectionService → capture checks → CurrentAnomalyDetector → existing localization/QC → SQLite/response. Pattern observation đã là post-QC diagnostic; internal dataclasses không phải HTTP schemas.
- Registry per-fabric override, settings/.env example, five golden QC/API cases, static-image/SQLite fixtures; Phase 8 checkpoint format, trusted-local loader, preprocessing, taxonomy và independent encoder.
- Torch 2.5.1+cpu, torchvision 0.20.1; không CUDA. Existing anomaly CPU ResNet18 layer2+3/adapter v4, 128 dim, stride8. Fabric A/B registry vẫn INACTIVE, active=None. Không activate để demo shadow.
- Security advisory Torch và Python subprocess timeout semantics: timeout không tự kill child, cần kill/reap. Chọn disposable subprocess thay vì timeout thread rồi để native inference tiếp tục chạy nền.

## 3. Implementation performed

1. Settings mới `supervised_shadow_enabled=false`, `supervised_shadow_config_path=""`; môi trường dùng prefix `FABRIC_`. Per-fabric `supervised.enabled` có thể override; mode duy nhất hợp lệ `shadow`, không active/fusion. OFF không đọc optional config, load checkpoint, spawn worker hoặc emit shadow audit.
2. Shadow config bắt buộc pinned checkpoint SHA-256, expected model version và explicit fabric allowlist khi có checkpoint path. trusted_local=false và allow_synthetic=false mặc định; không network-share checkpoint. Example không trỏ artifact nào.
3. Internal InspectionContext thêm config/error; InspectionResult thêm `known_defects_shadow`. Không thay Pydantic HTTP schemas, frontend, DB tables hoặc model/reference fields hiện hữu.
4. InspectionService tính anomaly evidence/localization/QC trước, sau đó gọi diagnostic shadow. Shadow không được truyền vào decide_fabric_inspection hoặc visualization. Đây là cùng-input dual branches, không phải cam kết chạy song song theo thời gian hoặc scheduler Phase 15.
5. Process runner khởi chạy đúng Python module trong worker CPU riêng, không shell/console window; mỗi server process tối đa một worker, không queue. Request khác nhận busy diagnostic và anomaly/QC vẫn tiếp tục.
6. Timeout budget bao gồm startup, checkpoint validation, decode và inference. Khi timeout kill/reap child; cleanup xác nhận không được thì capacity bị quarantine, request sau báo busy tới khi operator restart worker/server process. OS startup/cleanup có thể vượt budget nhỏ, đây không phải hard-real-time deadline.
7. Worker chỉ đọc image/checkpoint, không API/DB/training/registry calls. SHA được kiểm trên cùng open stream dùng để deserialize; trusted-local + pinned hash không phải sandbox hoặc chữ ký bảo mật. Version mismatch, synthetic disallowed, missing/corrupt/oversized checkpoint đều unavailable, không fabricate Good/defect evidence.
8. Dùng checkpoint normalization/image_size Phase 8, oriented RGB/bilinear resize và model.predict. CPU only, tối đa 512×512 model grid, checkpoint tối đa 256 MiB default, một Torch thread default. Không normalize hai lần hoặc thay anomaly embeddings.
9. Observation có verified model version/hash/synthetic flag, oriented original shape, model-grid mask shape/checksum, taxonomy, class pixel counts, mean class probability và mean winning confidence. Optional row-major class-ID RLE bounded; checksum/counts/geometry được validate. Quá run limit thì omission explicit, giữ checksum/counts, không bịa mask.
10. Audit log JSON `known_defects_shadow` sau khi inspection persist thành công, có inspection_id/fabric_id/anomaly_reference_id. Không log raw image, checkpoint paths, RLE hoặc full probability arrays. Dedicated INFO logger, lazy stderr fallback nếu app chưa cấu hình handler; không thay root logger. Log-sink failures không turn persisted success thành HTTP 500.
11. Misconfiguration/runner/model failures trở thành internal diagnostic statuses, không invalid reference hoặc fail inspection hợp lệ. Prior quality/registry/anomaly/decode failures giữ status và cleanup hiện hữu, không gọi shadow.
12. Synthetic demo train/reuse Phase 8 infrastructure chỉ để tạo ephemeral fixture checkpoint, kiểm real subprocess prediction/determinism/timeout/unavailable. Không train Company real dataset, không persist candidate production hoặc activate reference.

## 4. Files created

12 files mới:

- `backend/app/core/segmentation_shadow.py`
- `backend/app/services/segmentation_shadow.py`
- `backend/app/ai/segmentation_shadow_worker.py`
- `config/supervised_shadow_example.json`
- `scripts/check_phase9_supervised_shadow.py`
- `tests/segmentation_shadow_fixtures.py`
- `tests/test_segmentation_shadow_contracts.py`
- `tests/test_segmentation_shadow_worker.py`
- `tests/test_segmentation_shadow_runner.py`
- `tests/test_segmentation_shadow_integration.py`
- `docs/implementation/phase_9_shadow_report.json`
- `docs/implementation/phase_09_supervised_shadow.md`

## 5. Files modified

- `.env.example`: hai optional default-OFF settings và comment.
- `backend/app/core/config.py`: shadow flag/config path defaults.
- `backend/app/core/inspection_contracts.py`: additive internal context/result fields; file đã có từ phase trước, hiện vẫn untracked.
- `backend/app/services/inspection_service.py`: config policy, runner injection, post-QC observation, sanitized correlated audit logging.

Không sửa pre-existing tests/HTTP schemas, model/trainer Phase 8, registry JSON, DB, thresholds, FAISS, model artifacts, dữ liệu hoặc frontend.

## 6. Architecture before

`upload → existing capture/anomaly/localization/QC → persistence + HTTP response`

Supervised CNN/trainer/checkpoint chỉ có ở nhánh offline Phase 8, không có runtime observation.

## 7. Architecture after

QC path cũ giữ nguyên. Sau khi tính QC:

`same uploaded image → optional supervised policy → isolated CPU worker → known_defects_shadow`

`persisted inspection ID + shadow diagnostic → structured audit log`

Không có arrow từ shadow về anomaly scores, candidates, thresholds, final_result, annotated image hoặc public response. Disabled/unavailable/timeout/error/busy đều không promote/demote QC.

## 8. Backward compatibility

- Five golden QC POST/GET/SQLite/OpenAPI cases giữ nguyên cho observed/unavailable/timeout/error/busy/raised observer (30 combinations). Config failures cũng không reject otherwise valid inspection.
- Actual subprocess checkpoint giả lập foreground confidence >0.99 vẫn giữ GOOD/BAD theo anomaly. Original/annotated image hashes ON/OFF giống nhau.
- Default OFF không worker/load/log. Real fabric registry vẫn inactive; existing inactive inspect vẫn 409. Legacy placeholder inspection path không được thay đổi.
- 55 protected artifacts và bốn protected scope hashes giữ nguyên: storage311, Data643, user_files10, torch checkpoints3. Không activation/calibration/index rebuild/download hoặc dependency change.
- 280 pre-existing tracked/untracked files được fingerprint: chỉ đúng bốn files ở mục5 thay đổi; 276 files còn lại không đổi. Unrelated dirty user/phase work được giữ nguyên.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| test_segmentation_shadow_contracts.py, 30 cases | Safe defaults, strict flags/mode, pinned artifact policy, bounded classes/RLE/hash/geometry | PASS |
| test_segmentation_shadow_worker.py, 15 cases | Real model output/version, hash-before-load, unavailable/corrupt/synthetic/version gates, EXIF, bounded mask, inference exception | PASS |
| test_segmentation_shadow_runner.py, 14 cases | No-spawn early exits, busy/backpressure, IPC validation, process failures, kill/reap, cleanup quarantine, real subprocess deadline | PASS |
| test_segmentation_shadow_integration.py, 50 cases | 30 golden state combinations, internal evidence invariance, config/fabric isolation, prior failures, logging, actual API worker/timeout, CLI demo | PASS |
| Existing 725 tests | Regression Phase 0–8 | PASS |
| Frontend build | TypeScript/Vite compatibility | PASS |
| Baseline artifact/API smoke | Existing real model/index checks + isolated synthetic API cases | PASS, 77 assertions |
| Explicit log fallback diagnostic | Enabled INFO event appears without altering root logging/DB | PASS |
| compileall / git diff --check | Python syntax / tracked whitespace | PASS |
| Protected artifact/source/scope hashes | No accidental data/model/user changes | Không đổi ngoài bốn authorized source/config files |

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Full regression cuối trên source đã chốt: **834 passed, 1 warning in 136.00s**, frontend build PASS (22 modules), artifact/API smoke PASS (77 assertions), exit0. Warning FAISS/NumPy private namespace đã có trước phase.

Targeted trước ba EXIF/hash cases bổ sung: **106 passed, 1 warning in 44.13s**. Final collect: 109 new tests, tổng 834. Không sửa hoặc hạ assertions của tests cũ.

## 11. Hypothetical / synthetic cases tested

- Tiny pretrained synthetic checkpoint được train trong temp directory qua Phase 8, rồi hai disposable worker cho deterministic mask checksum. Không dùng Company masks hoặc locked pixels; không claim real accuracy.
- Separate artificial checkpoint cố tình output toàn bộ foreground với confidence >0.99 để chứng minh shadow evidence không reject Good/anomaly-independent Bad. Đây không phải trained meaningful model.
- Forced 0.05s deadline kill/reap actual worker và API vẫn201 theo frozen anomaly response. Missing/untrusted/oversized/wrong-SHA/wrong-version/synthetic-disallowed checkpoints không sinh mask hoặc fake verified version.
- Corrupt image, invalid EXIF, native predictor exception, invalid child JSON/exit/output, busy capacity, failed cleanup và log sink exceptions được kiểm fail-isolated.
- Mask RLE digest/coverage/labels và class probability support được kiểm. Raw mask chưa map về original/physical coordinates; log ghi rõ model_grid.

## 12. Known limitations

- Không có production segmentation candidate đủ dữ liệu/thẩm định. Phase 8 readiness vẫn UNVERIFIED_REAL_PIXEL_LABELS_AND_CAPTURE_GROUP_METADATA; actual demo chỉ software verification. Không segmentation activation/shadow enabling trong production registry/.env hiện hữu.
- Worker CPU khởi tạo/load model mỗi enabled request, latency có thể vài giây; timeout/busy có thể gây thiếu shadow observations nhưng có explicit diagnostic, không silent queue/drop. Không benchmark SLA hoặc GPU contention.
- Process timeout không là hard-real-time/sandbox. Startup/OS cleanup có thể vượt timeout; cleanup failure quarantine cần operator restart. Slot per server process, không global across multiple Uvicorn workers.
- Diagnostics chỉ logs/internal result, không durable DB history hoặc API debug endpoint. Log-sink failure có thể mất observation; có chủ ý không làm hỏng QC. Handler routing/retention do operator quản lý.
- RLE bounded có thể omitted; full class probability maps không giữ/log. Confidence chưa calibrated, class counts không phải physical area hoặc QC severity.
- Checkpoint pin/hash/allowlist không chứng minh provenance/accuracy hoặc training coverage đúng. Only explicitly trusted local artifacts: Torch 2.5.1 có [CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6); weights_only không đủ để mở checkpoint lạ. Không upgrade dependencies trong phase này.
- Không implement fusion, learned fusion, physical mapping, camera SDK, streaming, async scheduling hoặc supervised QC thresholds của các phase sau.

## 13. Risks introduced

- Enabled shadow tăng request latency/RAM/disk I/O; default OFF và disposable limited worker giảm scope nhưng không bảo đảm throughput industrial. Không share trainable weights/torch runtime state với anomaly.
- Optional per-fabric/global config bị khai sai → configuration_error/unavailable logs; phải theo dõi diagnostics để phân biệt không model với model không phát hiện foreground.
- High supervised confidence không phải rejectable defect hoặc production validity; Phase 9 cố ý không đổi QC. Failed shadow không làm uncertainty của nhánh anomaly thay đổi.
- Audit data có class counts/model fingerprint, không raw image/paths. Cần retention/privacy policy riêng nếu operational logs gửi ra external sink; phase này không gửi external messages/services.
- Cleanup implementation dựa trên [Python subprocess communicate/timeout guidance](https://docs.python.org/3.11/library/subprocess.html); kill chỉ áp dụng worker đã tạo, không kill parent hoặc unrelated processes.

## 14. Manual verification steps for me

Tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase9_supervised_shadow.py --demo
.\.venv\Scripts\python.exe -m pytest -q tests/test_segmentation_shadow_contracts.py tests/test_segmentation_shadow_worker.py tests/test_segmentation_shadow_runner.py tests/test_segmentation_shadow_integration.py
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

Demo kỳ vọng observed/model_version populated, deterministic_mask_across_workers=true, timeout.status=timeout, unavailable.status=unavailable, affects_qc=false, confidence_calibrated=false, artifacts_persisted=false. Các fixtures/checkpoints tự quản lý trong temp directories, không để lại candidate production.

Chỉ với candidate local được người dùng review/tin cậy và một existing valid anomaly reference trong môi trường phát triển: tạo config riêng từ example, điền checkpoint_path, expected_model_version đúng metadata, SHA-256 lowercase và allowed_fabric_ids đúng scope. Không sửa/activate real registry chỉ để demo.

```powershell
(Get-FileHash -LiteralPath 'C:\path\to\trusted_candidate.pt' -Algorithm SHA256).Hash.ToLowerInvariant()
$env:FABRIC_SUPERVISED_SHADOW_ENABLED = 'true'
$env:FABRIC_SUPERVISED_SHADOW_CONFIG_PATH = 'C:\path\to\reviewed_shadow_config.json'
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Đây là conditional manual workflow, chưa chạy với production data. Synthetic artifacts cần explicit allow_synthetic=true chỉ trong fixture/dev sandbox, không gọi model production-ready. Log `known_defects_shadow` có inspection_id và verified model_version khi observed; HTTP response không thêm field. A/B hiện INACTIVE nên POST vẫn409, không activate để làm bypass.

Rollback toggle trong cùng shell (restart app để cached settings refresh):

```powershell
$env:FABRIC_SUPERVISED_SHADOW_ENABLED = 'false'
$env:FABRIC_SUPERVISED_SHADOW_CONFIG_PATH = ''
```

Nếu per-fabric supervised.enabled=true, cũng disable explicit override trong config phát triển được kiểm soát; global OFF không thắng per-fabric explicit ON. Code rollback chỉ revert các additions Phase 9 trong bốn files mục5 và backup đúng12 files mới; không reset/clean toàn repo hoặc đụng các phase trước/user files. Chưa xóa/rollback dữ liệu nào.

## 15. Git diff summary

Phase 9: **12 files mới, 4 files có sẵn modified**, chưa stage/commit. Trong bốn files modified có internal contracts untracked từ phase trước nên `git diff --stat` không thể hiện file này.

Cumulative tracked stat sau Phase 9: **5 files changed, 254 insertions, 53 deletions**, gồm changes từ phase trước và user tempCodeRunnerFile.exe. Delta tracked Phase9 là79 insertions so với254/53 hiện tại và175/53 trước phase; không coi toàn cumulative diff là Phase9. FabricRegistry code không sửa thêm trong phase này. Staged diff rỗng.

## 16. Recommendation

**SAFE TO APPROVE** cho default-OFF/shadow-only infrastructure, backwards compatibility và controlled trusted-local synthetic verification. Không approve production segmentation quality, activation, fusion hoặc throughput. Chờ review của người dùng trước khi triển khai Phase 10.

============================================================
PHASE 9 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
