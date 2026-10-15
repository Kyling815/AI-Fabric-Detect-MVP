# Phase 10 — Rule-based hybrid FusionEngine

Ngày kiểm tra: 2026-10-14, Asia/Bangkok. Chỉ Phase 10; chưa bắt đầu Phase 11 QC business rules.

## 1. Goal

Triển khai `FusionEngine`: `SupervisedEvidence + AnomalyEvidence + CaptureEvidence → DefectEvidence`, rule-based, deterministic và auditable. Không learned fusion, không trộn NN distance với CNN probability bằng một weighted score không có đơn vị rõ.

Giữ final QC riêng: Phase 10 tạo evidence, không activate hybrid GOOD/REVIEW/BAD business mapping thuộc Phase 11. Runtime integration mặc định OFF, chỉ observe_only/internal/audit logs. Fail-safe áp dụng cho fused evidence: thiếu/lỗi/không xác minh capture không được trở thành NORMAL.

## 2. What I inspected before coding

- Attachment yêu cầu Phase 10, phân định Phase 11; báo cáo Phase 0–9 và baseline 834 tests. Chưa có meaningful production segmentation model được xác minh từ Phase 8.
- Branch `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; dirty worktree từ prior phases/user files. Không có AGENTS.md áp dụng trong repo/các thư mục cha đã kiểm tra. Không stage/commit hoặc reset/clean.
- `/api/inspect`, InspectionService/internal dataclasses, per-fabric registry/settings, capture quality gate, dense anomaly scores/boxes/localized review và QC function hiện hữu.
- Phase 9 shadow diagnostics: per-class mean winning confidence, mask checksum/RLE/model-grid geometry, version/hash/synthetic flag và failure statuses. Background confidence không phải known-defect confidence.
- Existing scores là dense nearest-neighbor distance aggregate; có nhiều aggregation methods/contrast scores. Không giả định nằm [0,1] hoặc dùng cùng threshold cho mọi method/fabric.
- Five frozen API/QC cases, real FAISS/SQLite temporary fixtures và shadow tests. A/B production registry vẫn INACTIVE; không activate để test fusion.

## 3. Implementation performed

1. Frozen/extra-forbid contracts cho FusionPolicy, identity, capture, foreground classes/mask descriptors, anomaly boxes/evidence và DefectEvidence. Strict IDs/counts/booleans, finite numeric values, conditional required provenance/geometry. Failed branches không chứa fake scores/masks/verified versions.
2. FusionEngine pure/read-only, không model load/training/DB/API writes. Revalidate models kể cả model_copy bypass; invalid payload trở thành invalid branch, không reinterpret thành score0.
3. Priority rules với inclusive threshold boundaries và fail-safe gates:

| Điều kiện | Fused kind | Ghi chú |
| --- | --- | --- |
| Capture invalid | INVALID_CAPTURE | Không tự gọi vải BAD |
| Capture unchecked/error hoặc capture/fabric/geometry mismatch | NEEDS_REVIEW | Không fuse evidence khác capture |
| Foreground confidence >= T_known | KNOWN_DEFECT | Giữ raw anomaly evidence, không discard unknown signal |
| Anomaly >= T_high | UNKNOWN_DEFECT | Threshold chỉ khi score method khớp policy |
| Anomaly >= T_review hoặc existing strong localized trigger | NEEDS_REVIEW | Không biến localized anomaly thành NORMAL |
| Missing/failed/disabled/busy/invalid branch, synthetic disallowed hoặc incomplete mask | NEEDS_REVIEW nếu không có positive branch hợp lệ | Positive surviving branch giữ kind nhưng review_required=true |
| Hai branches available, capture valid, compatible/trusted inputs, tất cả triggers dưới threshold | NORMAL | Evidence only, không phải QC GOOD |

4. Known confidence = max của mean winning pixel probability giữa foreground classes có predicted pixels; không lấy background confidence hoặc mean toàn ảnh. Lưu confidence definition/calibration flag; vẫn chưa có calibration/production acceptance claim.
5. Policy example starting values T_known0.9/T_high0.7/T_review0.3, method top_k_mean; default allow_synthetic_supervised=false. Mode duy nhất observe_only. Có policy_version, exact threshold snapshot, selected rule_id, source statuses/provenance, capture reasons, uncertainties, review_required và 10 rule checks.
6. Giữ known class mask descriptor trong model_grid và anomaly boxes trong oriented_source_pixels; không merge tọa độ hoặc đổi thành mm. RLE/full mask vẫn nằm ở Phase 9 internal observation khi available; không copy large arrays vào FusionEngine/logs.
7. Adapter đọc evidence từ cùng inspection invocation. Hash local input path để correlation không lộ path, KHÔNG phải image-content SHA/provenance attestation. Check shape/identity compatibility, dense scores/features/map/backbone consistency; empty/nonfinite/incompatible dense payload không được coi là available normal score0.
8. Quality gate None → capture unchecked, không fabricate valid. Phase 9 unavailable/timeout/error/busy/config errors → explicit branch statuses; background-only → không foreground candidates. Synthetic cần policy opt-in riêng cho fixture/dev, mặc định disallowed.
9. Additive runtime context/result fields, optional injected engine, settings `fusion_enabled=false`, `fusion_config_path=""`, per-fabric `fusion.enabled`. OFF không đọc optional policy hoặc gọi adapter/engine/log. Enabled không tự bật supervised/quality features.
10. Runtime vẫn tính legacy QC trước, rồi pattern/shadow/fused evidence. Audit sau persistence có inspection_id, qc_source=legacy_anomaly, legacy_qc_label, hybrid_qc_active=false và defect_evidence. Không public response/DB schema changes. Engine/config/log failures không làm hỏng existing legacy inspection; config/engine error tạo fused NEEDS_REVIEW.
11. CLI read-only hypothetical truth table: 13 cases và default synthetic-disallowed check. Không model load, tensor inference, training, output artifacts hoặc registry activation trong CLI.

## 4. Files created

12 files mới:

- `backend/app/core/fusion.py`
- `backend/app/ai/fusion.py`
- `backend/app/ai/fusion_adapter.py`
- `config/fusion_example.json`
- `scripts/check_phase10_fusion.py`
- `tests/fusion_fixtures.py`
- `tests/test_fusion_contracts.py`
- `tests/test_fusion_engine.py`
- `tests/test_fusion_adapter.py`
- `tests/test_fusion_integration.py`
- `docs/implementation/phase_10_fusion_report.json`
- `docs/implementation/phase_10_hybrid_fusion.md`

## 5. Files modified

- `.env.example`: default-OFF fusion settings/comment.
- `backend/app/core/config.py`: fusion_enabled/fusion_config_path.
- `backend/app/core/inspection_contracts.py`: internal fusion_policy/error và defect_evidence, không HTTP/DB schemas.
- `backend/app/services/inspection_service.py`: policy, optional engine injection, post-QC fusion and explicit audit.

Không sửa existing tests, classifier/QC function, shadow/model/trainer internals, frontend, registry JSON, .env thật, FAISS, thresholds/artifacts, database hoặc dataset.

## 6. Architecture before

`upload → capture/anomaly/localization → legacy QC → persistence/HTTP`

`optional independent segmentation → known_defects_shadow → audit`

Hai loại evidence chưa có shared rule fusion contract.

## 7. Architecture after

Giữ path cũ. Thêm nhánh opt-in evidence:

`dense anomaly + known_defects_shadow + capture quality → validated adapters → FusionEngine → DefectEvidence → internal result/audit`

Không arrow từ DefectEvidence về legacy decision/boxes/annotated output hoặc HTTP response. Business mapping severity/allowance/reject/review chưa triển khai; đó là Phase 11, chưa bắt đầu.

## 8. Backward compatibility

- Five golden cases × five supervised states (known/low/missing/timeout/failed): POST/GET/OpenAPI/SQLite giữ nguyên. Diagnostic high-known có thể khác legacy QC label, audit ghi rõ hybrid_qc_active=false; không được gọi đây là active fail-safe hybrid QC.
- ON/OFF original và annotated image hashes giống nhau; unchanged embeddings/anomaly scores/candidates/decision. Existing quality/encoder/registry/decode failures giữ status400/500/409/400 và skip fusion/cleanup đúng.
- Existing Phase 9 worker independence/timeout/shadow/API tests còn pass. Legacy placeholder path không đổi. Production A/B inactive vẫn409, không activate.
- 292 pre-existing tracked/untracked files fingerprinted: chỉ bốn files mục5 đổi, 288 unrelated files không đổi. 55 protected artifacts và storage311/Data643/user_files10/torch checkpoints3 scope fingerprints giữ nguyên.
- Không upgrade dependency, train/build/overwrite production model/index, evaluate locked pixels hoặc mutate registry/database/data. Existing synthetic regression fixtures vẫn được chạy isolated, không phải real accuracy evaluation.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| test_fusion_contracts.py, 55 cases | Strict finite fields, provenance, geometry, invalid policies, no GOOD/active-QC, complete NORMAL trace | PASS |
| test_fusion_engine.py, 75 cases | 36 boundary-grid combinations, missing/failure statuses, capture precedence, identity/method/synthetic/mask fail-safe, positive signal retention | PASS |
| test_fusion_adapter.py, 16 cases | Shadow conversion, background exclusion, dense validity, quality/capture correlation, read-only evidence | PASS |
| test_fusion_integration.py, 43 cases | 25 golden API state combinations, defaults/config/fabric isolation, prior failures, engine/log error, image hashes, CLI | PASS |
| Existing 834 tests | Regression Phase 0–9 | PASS |
| Frontend build | TypeScript/Vite compatibility | PASS, 22 modules |
| Artifact/API smoke | Real cached encoder/FAISS/reference + isolated synthetic API | PASS, 77 assertions |
| compileall / git diff --check | Python syntax / tracked whitespace | PASS |
| Source/artifact/scope fingerprints | Preserve user work/data/production artifacts | Chỉ bốn authorized source/config files đổi |

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

**1023 passed, 1 warning in 107.67s**, frontend build PASS, artifact/API smoke PASS (77 assertions), exit0. Targeted final Phase10: **189 passed, 1 warning in 6.89s**. Không existing test nào bị sửa/hạ assertions hoặc bị hỏng. Warning FAISS/NumPy private namespace đã có sẵn.

## 11. Hypothetical / synthetic cases tested

- Read-only CLI: high-supervised/low-anomaly, high/high, low/high, low/review, low/low; missing each/both branches; failure each branch; timeout; invalid/unchecked capture. 13 cases expected kinds đúng, default synthetic gate trả NEEDS_REVIEW.
- Unit 6×6 grid gồm equality0.9/0.7/0.3 và just-below boundaries, score1.2/1.6 không bị clip như probability. Không performance/calibration claim từ các số khai báo.
- All branch statuses và model_copy NaN/negative bypass → no fused NORMAL. Positive surviving branch vẫn giữ evidence và mandatory review. Fabric/capture/geometry mismatch suppresses qualified fusion.
- Background confidence không thành known class; mask omission, missing localization và score-method mismatch không được kết luận fully normal/complete. Strong localized anomaly ở low global score vẫn review.
- Integration dùng fake dense encoder + real FAISS/SQLite/HTTP và injected synthetic shadow; actual worker correctness đã được kiểm trong Phase9/full regression. Không pretend integration fixtures là real fabric accuracy.

## 12. Known limitations

- Fused evidence không là final QC. Legacy HTTP có thể GOOD trong khi fused diagnostic NEEDS_REVIEW; audit ghi rõ legacy source/hybrid inactive. Không claim active fail-safe hybrid rejection đã hoàn thành. Phase11 chưa triển khai.
- Không có meaningful production segmentation candidate, calibrated fusion thresholds hoặc customer acceptance/SLA. Vẫn cần real verified pixel labels/capture provenance và independent evaluation; historic postmortem locked set không là fresh acceptance holdout.
- Confidence dựa trên per-class mean winning pixels có thể bỏ qua pixel-peak nhỏ; cần eval/thay aggregation được versioned nếu có dữ liệu thực. Không học/calibrate fusion weights trong phase này.
- Identity path hash chỉ invocation correlation từ adapters, không cryptographic image identity hay provenance proof. Mask descriptor tham chiếu Phase9 checksum, không chứa/full reconstruct mask trong fusion output. Không map model_grid mask và anomaly boxes vào một physical frame.
- Capture gate OFF khiến fused state unchecked/review; supervised OFF/missing/fail khiến incomplete/review. Không tự tắt các fail-safe gates để current Company data trở thành NORMAL.
- Diagnostics log/internal only, không durable fused history/debug API; logging failure có thể mất record nhưng không mutate legacy QC. Branch infrastructure/runtime limits Phase9 vẫn áp dụng nếu bật supervised shadow.

## 13. Risks introduced

- Config/score-method mismatch có thể tăng diagnostic review; không dùng config example như threshold đã calibrated. Per-fabric score scales không interchangeable.
- Operators có thể nhầm NORMAL evidence thành GOOD hoặc KNOWN_DEFECT thành BAD; contract không có GOOD/BAD kinds, affects_qc=false, modeobserve_only và audit explicit legacy/hybrid separation để ngăn sự nhầm lẫn đó.
- Optional computation/log tăng overhead khi bật; không measure industrial throughput trong phase này. Pure engine không thay tensor weights/FAISS geometry hoặc introduce GPU contention.
- Future QC consumer phải tôn trọng review_required/capture status/uncertainties và provenance, không chỉ đọc kind rồi waive review. Business implementation cần review ở Phase11, không inference từ phase hiện tại.

## 14. Manual verification steps for me

Tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase10_fusion.py --demo
.\.venv\Scripts\python.exe -m pytest -q tests/test_fusion_contracts.py tests/test_fusion_engine.py tests/test_fusion_adapter.py tests/test_fusion_integration.py
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

CLI báo HYPOTHETICAL_SOFTWARE_CHECK_ONLY, case_count13, models_trained_or_loaded=false, artifacts_created=false, hybrid_qc_active=false. Thiếu/lỗi branch hoặc invalid/unchecked capture không trả NORMAL.

Trong một dev sandbox có existing valid anomaly reference, có thể dùng config riêng từ fusion_example để quan sát audit, không activate real A/B registry:

```powershell
$env:FABRIC_FUSION_ENABLED = 'true'
$env:FABRIC_FUSION_CONFIG_PATH = 'C:\path\to\reviewed_fusion_policy.json'
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Đây là conditional workflow, chưa thực hiện trên production. Enabled fusion không tự bật supervised/quality gate; nếu thiếu các inputs đó, phải thấy fused NEEDS_REVIEW/review_required=true thay vì NORMAL. HTTP/QC vẫn legacy, log defect_evidence có exact policy/rule trace cùng legacy_qc_label và hybrid_qc_active=false. A/B hiện inactive nên API vẫn409. Synthetic opt-in chỉ fixture/dev được khai báo rõ, không production approval.

Rollback toggle (restart app để refresh cached settings):

```powershell
$env:FABRIC_FUSION_ENABLED = 'false'
$env:FABRIC_FUSION_CONFIG_PATH = ''
```

Explicit per-fabric fusion.enabled=true vẫn thắng globalOFF; disable override trong config phát triển được kiểm soát nếu có. Code rollback chỉ revert Phase10 additions trong bốn files mục5 và backup đúng12 files mới; không reset/clean/revert toàn worktree hoặc prior phases/user files. Chưa xóa/rollback dữ liệu nào.

## 15. Git diff summary

Phase10: **12 files mới, 4 existing files modified**, chưa stage/commit. Contracts file đã untracked từ phase trước nên không nằm trong tracked git diffstat.

Cumulative tracked `git diff --stat`: **5 files changed, 326 insertions, 53 deletions**; gồm earlier phase changes và user tempCodeRunnerFile.exe. Diffstat insertion delta từ Phase9 là72 (326 vs254), không phải toàn bộ cumulative changes đều Phase10. FabricRegistry code không sửa trong phase này. Staged diff rỗng.

## 16. Recommendation

**SAFE TO APPROVE** cho FusionEngine evidence infrastructure, observe-only integration và backward compatibility. Không approve hybrid production QC, model accuracy hoặc starting thresholds. Chờ review của người dùng trước khi bắt đầu Phase11 business/QC separation.

============================================================
PHASE 10 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
