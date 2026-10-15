# Phase 11 — QC Decision Layer Clean Separation

## 1. Goal

Tách evidence AI khỏi quy tắc chấp nhận nghiệp vụ. CNN/FAISS tạo class, confidence, mask và anomaly score; QC quyết định GOOD / NEEDS_REVIEW / BAD bằng policy riêng. Chỉ triển khai Phase 11, ngày 2026-10-15 (Asia/Bangkok); không bắt đầu Phase 12.

## 2. What I inspected before coding

- Yêu cầu Phase 11 trong pasted request và hồ sơ Phase 0–10; xác nhận Phase 10 chỉ sinh evidence observe-only.
- InspectionService, internal contracts, HTTP response/OpenAPI, SQLite persistence, registry/config resolution và quality gate.
- Dense/patch legacy decision functions, thresholds/score methods, golden responses và các test cũ.
- Fusion contracts/engine/adapters: capture validity, identity, geometry, incomplete branches, provenance, confidence calibration, model-grid mask counts và uncertainty trace.
- Shadow worker/failure statuses và safe regression runner. Không có AGENTS.md áp dụng trong repo/parent đã kiểm tra.
- Branch main, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; worktree có prior phases và user changes. Fingerprint 304 existing files trước sửa; không stage/commit/reset/clean.
- Registry A/B thực vẫn INACTIVE, active reference None. Model cache hiện tại là ResNet18 layer2+3, adapter v4, 128 dimensions, stride 8, CPU. Hệ thống hiện tại vốn đã CNN-based.

## 3. Implementation performed

Tạo frozen QCPolicy, KnownSeverityRule, QCRuleInput, ClassSeverity và QCDecision. Rule input chứa fabric scope, fused evidence và class area count khớp chính xác mask geometry. Từ chối unit mm2 giả, nonnumeric thresholds, duplicate class/rules, sai pixel count và tổng class counts vượt mask grid.

Tạo QCDecisionEngine thuần quy tắc, không gọi neural network hoặc FAISS. Luồng ưu tiên:

1. Thiếu/lỗi evidence, capture không hợp lệ/không kiểm chứng, scope mismatch, uncertainties, branch incomplete hoặc trace thiếu/bất nhất → NEEDS_REVIEW.
2. Synthetic/unknown provenance hoặc confidence chưa calibrated bị chặn mặc định. Chỉ fixture/dev opt-in mới bỏ hai gate này.
3. Known defect với severity đạt `reject_min_fraction` → BAD; severity không quá `allow_max_fraction` có thể GOOD, nhưng chỉ khi evidence đầy đủ và anomaly không yêu cầu review.
4. Unknown high anomaly → BAD hoặc NEEDS_REVIEW theo policy, tuyệt đối không GOOD. Known allowable không che được anomaly cao đồng thời.
5. Khoảng giữa hai ngưỡng, class chưa có allowance hoặc anomaly borderline/localized → NEEDS_REVIEW. Complete NORMAL evidence có thể GOOD sau tất cả các gate.

Severity hiện dùng `area_px / (mask_height * mask_width)`, coordinate frame `model_grid`. Hai biên inclusive; cần khoảng trống giữa allow/reject để review. Known rejectable có ưu tiên hơn localized review nếu evidence không có uncertainty. Đây là quy tắc minh họa tách biệt, không phải production thresholds.

Chuyển logic dense/patch legacy sang QC service; giữ wrapper, signature và return types cũ tại ai/fabric_decision.py và ai/classifier.py. Không đổi legacy threshold values, reason strings, casing hoặc patch confidence formula.

InspectionService hỗ trợ OFF, observe_only và explicit enforce. Default `enabled=false`, mode observe_only, config path rỗng. Global settings dùng `FABRIC_HYBRID_QC_*`; per-fabric `qc` override có `enabled`, `mode` và policy fields. Runtime mode không nằm trong business policy file. Enforce bắt buộc scope fabric không rỗng; không tự bật quality gate, fusion hoặc supervised branch.

Observe-only giữ final legacy decision, ghi QC đề xuất vào internal result/audit. Enforce thay nhãn/reason theo QC, lưu qua HTTP/SQLite path có sẵn; lý do thêm prefix `QC_`. Config/engine/evidence lỗi không âm thầm fallback sang legacy GOOD. Enabled/mode sai kiểu được xử lý fail-closed; logger lỗi không đảo ngược quyết định. Audit tách legacy, proposed và selected labels, policy version, assessments, trace và applied flag.

Không sửa .env thật, registry thật, thresholds thật, model weights, references, datasets, database schema hoặc frontend.

## 4. Files created

- backend/app/core/qc.py
- backend/app/services/qc_decision.py
- config/qc_policy_example.json
- scripts/check_phase11_qc.py
- tests/qc_fixtures.py
- tests/test_qc_contracts.py
- tests/test_qc_decision.py
- tests/test_qc_legacy_compatibility.py
- tests/test_qc_integration.py
- docs/implementation/phase_11_qc_separation.md
- docs/implementation/phase_11_qc_report.json

## 5. Files modified

- .env.example: thêm ba settings QC, mặc định OFF/observe-only.
- backend/app/core/config.py: settings typed cho enable/mode/config path.
- backend/app/core/inspection_contracts.py: optional policy/mode/error và QC/legacy results nội bộ.
- backend/app/services/inspection_service.py: policy resolution, evaluation, optional enforcement và audit.
- backend/app/ai/fabric_decision.py: dense compatibility wrapper.
- backend/app/ai/classifier.py: patch compatibility wrapper.

FabricRegistry, existing tests, prior phase documents, Revision.md và user binary không được sửa trong phase này.

## 6. Architecture before

API → InspectionService → anomaly/localization → legacy decision logic trong ai modules → response/SQLite. Optional shadow/fusion chỉ ghi evidence; chưa có business policy consumer riêng.

## 7. Architecture after

API → InspectionService → anomaly/localization → legacy QC service (qua compatibility wrapper).

Optional supervised/anomaly fusion → explicit QCRuleInput → QCDecisionEngine → proposed business decision. OFF/observe-only chọn legacy; explicit enforce chọn QC. Response/SQLite schema vẫn như cũ. CNN/FAISS weights và feature geometry không bị thay đổi bởi business policy.

Không thực hiện physical mapping, learned fusion, line-scan/PLC hoặc model training.

## 8. Backward compatibility

- OFF/observe-only giữ đúng năm golden HTTP/SQLite cases, OpenAPI schema, confidence, identity, reason, scores, boxes và annotations.
- 49 legacy parity cases giữ dense/patch labels, reasons, confidence và threshold error.
- Enforce là opt-in có chủ đích thay final_result/reason; không tuyên bố inference output unchanged khi bật chế độ này.
- Existing invalid capture vẫn HTTP400 trước inference, không bị gán BAD fabric. Pure QC INVALID_CAPTURE trả NEEDS_REVIEW.
- 304 pre-existing files kiểm tra: đúng sáu files mục 5 thay đổi, 298 unrelated files không đổi. 55 protected artifacts không có hash mismatch. Scope fingerprints storage311, Data643, user_files10, torch checkpoints3 giữ nguyên.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| test_qc_contracts.py — 36 cases | Policy/input/output validation, units, disjoint class counts | PASS |
| test_qc_decision.py — 38 cases | Severity boundaries, unknown, review, incomplete/invalid/provenance/trace gates | PASS |
| test_qc_legacy_compatibility.py — 49 cases | Exact dense/patch decision parity | PASS |
| test_qc_integration.py — 35 cases | OFF/observe/enforce, HTTP/SQLite/schema, config/failure, logging, image hashes, CLI | PASS |
| Phase 11 CLI — 9 hypothetical cases | Read-only independent QC demonstration | PASS |
| Full suite — 1181 cases | All existing and new regressions | PASS |
| Frontend build — 22 modules | TypeScript/Vite compatibility | PASS |
| Baseline smoke — 77 assertions | Cached actual backbone, existing indexes/registry, isolated synthetic API/storage/SQLite | PASS |
| compileall, git diff --check, fingerprints | Syntax/whitespace and preservation | PASS |

New tests ban đầu thất bại: fixture Fusion dùng top_k_mean trong khi golden runtime dùng max; sau đó assertions dùng sai HTTP keys decision_reason/defects thay vì reason/boxes và kỳ vọng reason/precedence chưa đúng. Đã sửa new fixtures/assertions theo schema và evidence thực tế, không bỏ guard hoặc sửa old tests. Bỏ wrapped test engine có thể che gate thực, bổ sung exact-reason assertions và kiểm tra trace/mask counts. Chuẩn hóa prefix lý do lỗi để không xuất QC_QC_*. Không có existing test bị hỏng trong full run cuối.

## 10. Full regression test result

Actual command tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Actual result: **1181 passed, 1 warning in 227.16s (0:03:47)**; frontend build PASS, 22 modules; 77 smoke assertions PASS; process exit code 0. Previous suite 1023 + 158 new Phase 11 cases = 1181.

Warning là FAISS loader dùng deprecated NumPy private namespace đã có từ baseline, không phải lỗi QC. Regression/API sử dụng DB/storage temporary; smoke chỉ đọc cached real backbone/references, không train model, không activate fabric. Actual A/B vẫn INACTIVE và upload vẫn409.

Targeted command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qc_contracts.py tests/test_qc_decision.py tests/test_qc_legacy_compatibility.py tests/test_qc_integration.py -q
```

Actual result: **158 passed, 1 warning in 12.59s**.

## 11. Hypothetical / synthetic cases tested

CLI: known allowable/rejectable/borderline, unknown high, anomaly borderline, complete normal, no evidence, missing branch, invalid capture. Fixtures thêm exact boundaries, simultaneous known allowance/high anomaly, wrong scope/count/units, synthetic/uncalibrated provenance, trace thiếu/duplicate/method mismatch, timeout/busy/worker error/config error, logger error và invalid capture HTTP400.

Integration dùng fake encoder + synthetic mask, nhưng FAISS/HTTP/SQLite thật trong temporary sandbox. CLI dùng declared synthetic counts/checksums, không chứng minh một real mask hay model quality. Demo thresholds .05/.2 và test thresholds .2/.4 chỉ là giả định. Không có real-data QC accuracy, FPR/recall, customer SLA hoặc production acceptance evaluation.

## 12. Known limitations

- Area fraction không phải mm2, source-image area, component length hoặc physical severity; Phase 12 chưa triển khai.
- Policy example không invent known allowances; cần business approval/calibration mới đưa ra thresholds thật.
- Current supervised runtime chưa qualified; confidence chưa calibrated và synthetic evidence bị chặn mặc định. Enforce hiện chỉ là infrastructure cho trusted development.
- Known-defect QC có thể BAD trong khi public boxes/count/area/annotated vẫn anomaly-only. Chưa đưa known masks vào HTTP hoặc annotation; không nên dùng cho operator/PLC production.
- Fusion path hash chỉ correlation invocation, không content/provenance proof. Caller-declared validity trong pure demo không chứng minh acquisition quality.
- Audit best-effort logging, chưa lưu durable policy/evidence audit vào DB. Không measured industrial throughput hoặc streaming/camera SDK.
- OFF/observe-only cố ý giữ legacy quyết định và những hạn chế cũ; không phải chế độ fail-closed hybrid.

## 13. Risks introduced

Sai business policy hoặc opt-in overrides có thể đổi tỷ lệ review/reject; enforce còn cần đủ quality/fusion/supervised dependencies. Thiếu/lỗi inputs sẽ tăng review thay vì tự chấp nhận. False allow/reject khi tự hiệu chuẩn sai vẫn là rủi ro; software tests không chứng minh detection accuracy.

Diện tích model-grid phụ thuộc preprocessing/resolution và là aggregate của high-confidence known classes, không đo từng connected component. Audit có overhead/log-volume và không durable; lý do QC và annotation anomaly-only có thể gây nhầm nếu bật production trước validation. AI wrappers phụ thuộc QC service nhưng service không import các wrapper, không tạo import cycle trong regression.

## 14. Manual verification steps for me

Tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase11_qc.py --demo
.\.venv\Scripts\python.exe -m pytest tests/test_qc_contracts.py tests/test_qc_decision.py tests/test_qc_legacy_compatibility.py tests/test_qc_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

CLI cần case_count9, HYPOTHETICAL_SOFTWARE_CHECK_ONLY, production_policy_activated=false, models_trained_or_loaded=false, writes_performed=false. Integration tests tự tạo temporary registry/reference/DB và kiểm chứng observe-only/enforce, không cần bật real registry. JSON report lưu kết quả lần chạy này; không thay thế lần chạy verification của bạn.

Để rollback một dev sandbox đã bật QC, tắt global và restart app để refresh cached settings:

```powershell
$env:FABRIC_HYBRID_QC_ENABLED = 'false'
$env:FABRIC_HYBRID_QC_MODE = 'observe_only'
$env:FABRIC_HYBRID_QC_CONFIG_PATH = ''
```

Nếu có per-fabric qc.enabled=true thì globalOFF không thắng override: đặt qc.enabled=false trong registry sandbox đã kiểm soát, rồi restart. Không đưa demo synthetic/calibration opt-ins vào production. Code rollback chỉ revert sáu Phase11 edits và backup/remove đúng11 files mới; giữ prior phases/user work, không git reset/clean toàn bộ. Chưa thực hiện deletion hoặc rollback dữ liệu nào.

## 15. Git diff summary

Phase11: **11 files mới, 6 pre-existing files modified**, staged diff rỗng. inspection_contracts.py đã untracked từ prior phase nên không xuất hiện trong tracked diffstat.

Cumulative tracked git diff --stat: **7 files changed, 432 insertions, 101 deletions**, gồm prior phases và preexisting user tempCodeRunnerFile.exe. Trước Phase11 là 5 files/326 insertions/53 deletions; cumulative stats không phải patch riêng Phase11. FabricRegistry và user binary không được sửa trong phase này. Không stage/commit hoặc tự động tiến phase.

## 16. Recommendation

**SAFE TO APPROVE** cho software QC separation, default-OFF backward compatibility và explicit development enforcement. Không approve production QC policy, model accuracy, synthetic/calibration waivers hoặc vận hành PLC. Chờ user review; Phase 12 chưa bắt đầu.

============================================================
PHASE 11 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
