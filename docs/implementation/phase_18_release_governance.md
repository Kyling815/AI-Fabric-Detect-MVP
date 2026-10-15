# Phase 18 — Acceptance Gate and Manual Release Governance

Ngày: 2026-10-15, Asia/Bangkok. Phạm vi: **governance/software infrastructure only**. Release status: **BLOCKED BY DATA / TRAINED CANDIDATE**. Chưa có candidate production được chấp nhận/đăng ký/kích hoạt. Full regression: **1920 passed, 1 warning**, frontend và77 smoke assertions PASS. [Báo cáo máy đọc](phase_18_release_report.json).

## 1. Goal

Triển khai luồng candidate artifact → validation → fresh source/lot/camera-disjoint locked test → acceptance report → **manual** registry activation. Không train model, đặt customer SLA, tự điền threshold hay tạo acceptance evidence. Giữ nguyên runtime, API, FAISS, thresholds, registry data, hai fabric INACTIVE và scope approval giới hạn của Phase17.

## 2. What I inspected before coding

Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`, worktree dirty từ các phase trước/người dùng. Chụp SHA381 file trước sửa; kiểm tra AGENTS.md trong repo/các thư mục cha liên quan, không có file áp dụng. Đọc yêu cầu Phase18 và approval Phase17, FabricRegistry, segmentation data/split/metrics, CLI evaluation/reference build, báo cáo dữ liệu và TensorRT. Không stage/commit/reset.

Phát hiện `scripts/evaluate_fabric_demo.py --activate-if-accepted` có thể ACTIVE reference theo tiêu chí hard-code của evaluation cũ. Đường này không kiểm tra fresh holdout hoặc manual approval và phải bị chặn. Không thay hành vi inference để chặn đường CLI quản trị này.

Read-only preflight hiện tại:0 masked Company row real,120 masked row synthetic; thiếu roll_id/lot_id/camera_id. Không có trained candidate hoặc fresh locked dataset được cung cấp để acceptance. Không đánh giá lại ảnh historical locked/postmortem. Không diễn giải các artifact adapter/FAISS thành trained segmentation candidate.

## 3. Implementation performed

### Contracts và provenance

Schema strict/frozen, extra fields bị từ chối, finite numeric values và timezone-aware dates. Artifact dùng absolute path + lowercase SHA256, bounded same-open-stream reads, không load/execute model. Hash ràng buộc bytes chứ không chứng minh nguồn an toàn, model đã train hoặc attestation là thật.

Release bundle pin6 JSON input: candidate, approved policy, complete development/historical exposure inventory, locked test, validation evidence và full-hybrid evaluation evidence. Candidate pin model/training report cùng reference metadata, memory bank, FAISS index, thresholds, deployment engine và runtime recipe; version/fabric/backbone/runtime phải rõ ràng. Recipe ràng buộc model/engine/reference hashes, FP32 và không có FP16 fallback. Validation phải gắn đúng candidate/inventory và hoàn thành functional/numerical checks trước locked evaluation.

### Fresh locked test

- Policy được duyệt trước candidate freeze; validation trước lock; candidate/policy SHA được bind vào lock; evaluation sau lock, không có timestamp tương lai. Policy có thời hạn.
- Từ chối lock đã mở trước, historical/postmortem hoặc không đúng candidate/policy. Không tự sinh nhãn fresh từ tên thư mục/split cũ.
- Bắt buộc real origin, explicit source/root hash/ancestor lineage, lot/roll/camera IDs và human-verified ground truth; segmentation cần mask đã pin. Provenance/custody được người chịu trách nhiệm attestate, không suy từ folder name.
- So source ID/hash, ancestor ID/hash, image hash/path, lot, roll, camera với toàn development/historical inventory. Bổ sung deny known Company exposure từ CSV cũ bằng metadata/hash only, không chạy inference trên historical images. Trùng bất kỳ chiều nào thì block; ảnh lặp bytes trong locked set cũng bị block.
- Evaluation phải bao phủ đúng locked sample IDs; không bỏ mẫu lỗi/khó. Full-hybrid evidence là bắt buộc: số đo segmentation-only Phase17 không được dùng thay factory-pipeline latency.

### Configurable acceptance, không có ngưỡng mặc định

Template `config/release_policy_unapproved_example.json` có toàn bộ threshold=`null`, approval trống và per-class criteria trống. Không có ngưỡng customer/SLA được suy từ legacy evaluation hoặc benchmark Phase17.

Hỗ trợ defect recall, false-negative event count, underkill, FPR/overkill, precision, F1, foreground IoU/Dice, unknown-defect recall, false alarms/missed defects mỗi100m, P50/P95 latency, throughput và per-class recall. Tính từ imported event/pixel counts/timing samples đã pin, không nhận một boolean accepted do caller đưa vào làm đủ bằng chứng.

Đơn vị rõ ràng: recall/precision/F1/underkill theo one-to-one defect-event matching; FPR/overkill theo normal inspection-unit; IoU/Dice micro foreground pixels. Unknown population hoặc denominator vắng → `null`, không thành1/0 giả. Matching protocol phải explicit. Hiện observation là một lớp foreground/unknown trên mỗi inspection unit; evaluator đa lớp phải cung cấp đơn vị đánh giá phù hợp, không gán nhiều class count vào một field.

Per100m cần measured intervals và calibration artifact xác nhận đúng dataset/camera/roll, đơn vịm; từ chối interval chồng nhau trên cùng physical roll kể cả khác camera. Không có distance/calibration thì không dựng rate. Latency sample phải dương/finite, khai báo warmup excluded và CUDA synchronization nếu dùng; throughput có count/wall-time/protocol riêng. IoU/Dice chỉ được NOT_APPLICABLE khi candidate không áp dụng segmentation và có lý do đã duyệt; các metric khác không được skip.

Thiếu criterion/evidence → BLOCKED; vượt ngưỡng → REJECTED; đạt toàn bộ → AWAITING_MANUAL_APPROVAL, **không tự ACTIVE**. Nếu provenance sai, headline metrics giữ null, không được trình bày là production evidence.

### Manual activation và legacy bypass

Activation là function/CLI operator action riêng, không được gọi từ validation, report generation hoặc runtime. Trước khi ghi: rehash/recompute toàn acceptance report và evidence, kiểm tra policy còn hạn, approval/reviewer khác evaluator, confirm token đúng candidate/report, expected registry SHA đúng và target reference đã được chuẩn bị CANDIDATE/INACTIVE với đầy đủ provenance. Không tự đăng ký candidate, không tự sửa artifact, thresholds hoặc thay reference đang active khác.

Registry phải bind đúng candidate/model/reference/backbone/runtime versions, model/engine/recipe/metadata/bank/index paths và frozen threshold bytes. Ghi có exclusive cooperative lock, backup bytes gốc, atomic replace và audit record chứa approval/report/candidate hashes. Replay/stale registry/tamper/forged report bị từ chối. Failure trước replace giữ registry nguyên; backup được giữ lại nếu đã tạo. Không có activation nào trên production trong phase này; chỉ test registry tạm.

Chặn cả legacy CLI flag (exit2 trước evaluation/writes) và direct legacy `activate()` (raise). Non-activation legacy evaluation vẫn dùng thuật toán cũ, thêm nhãn `LEGACY_HISTORICAL_DIAGNOSTIC_NOT_RELEASE_APPROVAL`, `release_accepted=false` để không coi kết quả cũ là release approval.

Hard deny FP16/FP16_MIXED và13 known Phase17 research artifact hashes, kể cả đổi tên/gắn nhãn trained. FP32 Phase17 chỉ là baseline kỹ thuật; probe synthetic/untrained không được đăng ký hay promote. Không sửa hoặc chạy lại benchmark Phase17.

## 4. Files created

- `backend/app/core/release_governance.py`
- `backend/app/services/release_governance.py`
- `scripts/check_phase18_release.py`
- `config/release_policy_unapproved_example.json`
- `tests/release_governance_fixtures.py`
- `tests/test_release_governance.py`
- `tests/test_release_governance_integration.py`
- `docs/implementation/phase_18_release_governance.md`
- `docs/implementation/phase_18_release_report.json`

Không tạo trained checkpoint, engine, dataset hoặc release candidate thật. Không chạy report export/activation vào production storage. Test artifacts/registry chỉ trong pytest temp scope.

## 5. Files modified

Chỉ `scripts/evaluate_fabric_demo.py`:5 insertions/17 deletions, chặn legacy activation và phân biệt diagnostic với release approval. Đây là thay đổi có chủ ý của CLI quản trị; không phải thay đổi inspection API.380 file hiện hữu khác giữ SHA.

Không sửa `.env`, requirements, `.venv`, `.venv-tensorrt`, backend runtime/registry service, thresholds, FAISS/reference/model artifacts, frontend, DB hoặc báo cáo phase trước. Không cài dependencies hoặc đổi driver.

## 6. Architecture before

Reference registry runtime read-only, builder tạo references INACTIVE; có legacy evaluation/activation CLI dựa historical locked và hard-coded acceptance. Chưa có unified, hash-bound, fresh-test/manual governance.

## 7. Architecture after

Thêm offline fail-closed governance độc lập. Candidate/validation/fresh evidence được đánh giá theo owner-approved policy; report không mutate registry. Operator phải gọi riêng manual activation với approval và exact pins. API tiếp tục dùng FabricRegistry/inference cũ; không auto migrate GPU/full hybrid hoặc đăng ký research probe.

## 8. Backward compatibility

55 protected artifact,311 original storage file,643 Data file,10 user file và3 cache checkpoint giữ fingerprint. Candidate/benchmark Phase17 giữ nguyên. Dependency CPU freeze không đổi; pip check CPU/GPU PASS; driver610.62 không đổi. Hai fabric vẫn INACTIVE. Những thay đổi dirty từ trước và user binary được giữ nguyên.

Legacy activation không còn backward-compatible theo thiết kế vì không đáp ứng gate mới; evaluation không kích hoạt vẫn tồn tại và được đánh dấu historical diagnostic. Không thay public HTTP schema/status, model behavior hoặc thresholds.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| Strict schema/policy/metric unit tests | Null/NaN/bool, bounds, rates, missing populations, all criteria | PASS |
| Disjointness/history/annotation tests | Source/hash/lineage/lot/roll/camera leak, history relabel, duplicate image, real ground truth | PASS |
| Provenance/FP16/research tests | Invalid candidate/recipe/evidence, known research hash, segmentation-only latency | PASS |
| Manual integration | TEMP registry transition, recompute, approval, CAS/replay/tamper, backup/atomic failure | PASS |
| Legacy bypass tests | Direct activate and flag refused before writes | PASS |
| Current audit CLI | Actual project blocked, metrics null, both INACTIVE | PASS |

**113 passed,1 warning in23.53s** cho test Phase18. Warning FAISS NumPy namespace có từ trước khi import legacy CLI. Không dùng số liệu fixture làm real accuracy hoặc acceptance evidence. Compileall/pip check/git diff --check PASS.

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Clean run trên source đã freeze: **1920 passed,1 warning in269.83s**, exit0. Tăng113 test so với1807 cuốiPhase17. Frontend build PASS,22 modules (461ms);77 smoke assertions PASS; DB/storage kiểm thử cô lập. Warning duy nhất là FAISS NumPy namespace có từ trước. Không có test cũ bị hỏng và không sửa test cũ.

Development notes: reader ban đầu đọc theo toàn bộ size cap1GiB cho mỗi file nhỏ, gây overhead; đã sửa dùng fstat/read đúng size trên cùng stream, giữ size/hash guard và restart đúng hai process test của lượt này. Một intermediate full run có1 test CLI mới fail/1912 pass vì process fixture đã nạp source cũ trong khi CLI subprocess đọc gate engine/recipe mới; không coi lượt này là kết quả cuối. Fixture và checks đã đồng bộ;113 focused tests và clean full1920-test run đều pass. Không có production mutation do các lượt lỗi này.

## 11. Hypothetical / synthetic cases tested

Toàn bộ happy-path metrics/criteria/approval trong tests là **metadata/count fixture giả định, TEMP ONLY**; model/engine bytes là stub không thực thi. Các declaration real/trained trong fixture chỉ để exercise contract branch cho một kịch bản giả định, không phải xác nhận đã train model. Không đưa report accepted của fixture vào storage/report vận hành. Mọi activation test chỉ ghi registry tạm, không trỏ production.

Đã thử thiếu từng criterion; reject recall/underkill; population rỗng; missing meters/calibration; intervals chồng giữa camera; synthetic/unreviewed masks; timestamps/lock binding sai; data leakage/ancestor reuse; evidence model/policy/locked hashes sai; segmentation-only latency; candidate FP16/research; forged accepted report; changed artifacts/registry; stale lock/replay; wrong engine/version/threshold; atomic filesystem failure. Một unit no-data result không được gọi production accuracy.

Kết quả thật của project: **BLOCKED BY DATA / TRAINED CANDIDATE**, accepted=false, mọi metric quality/latency/throughput acceptance=`null`. Preflight metadata-only không train, không predict, không đánh giá historical locked ảnh hoặc tạo fresh dataset.

## 12. Known limitations

- Không có trained candidate, real verified pixel masks, fresh source/lot/camera-disjoint test hoặc owner-approved thresholds. Không có model/reference production được chấp nhận.
- Framework nhập evidence từ trusted evaluator, không phải model runner/event matcher/annotation verifier. Tính đúng counts/matching, capture lineage, blind custody và training attestation cần được xác minh độc lập. Hash pin không chống người có quyền sửa toàn bộ input rồi khai báo sai.
- Không có PKI/SSO/RBAC hoặc chữ ký điện tử; reviewer/custodian là identity attestation. OS permissions, người review thật và data custody là trust boundary bên ngoài. Known Company history denylist chỉ là lower bound; inventory phải bao phủ cả lịch sử bên ngoài repo.
- Lock là manifest/custody assertion gắn candidate/policy, không phải blind evaluation server/access-control vault hoặc universal cross-system test-consumption ledger. Không dùng việc đổi manifest version làm bằng chứng freshness.
- Atomic publication dùng cooperative lock và expected-hash checks; legacy builder không dùng lock này. Manual activation phải trong maintenance window, không đồng thời với writer khác. Backup phục hồi được; chưa triển khai auto rollback hay automatic promotion.
- Manual adapter cập nhật registry governance metadata/pointer, không triển khai runtime GPU/hybrid mới. Candidate thực phải được validation trên đúng runtime recipe tương thích trước activation; current legacy references thiếu release provenance nên không thể đi tắt qua adapter mới.
- Per100m chỉ đúng khi interval/calibration/event matching đã được xác minh và không double-count. Coverage/sample size/statistical-confidence và SLA thực tế vẫn cần chủ sở hữu quyết định, không tự đặt trong template.

## 13. Risks introduced

Nhầm imported/fixture evidence thành production evidence, khai báo lineage không đúng, policy quá lỏng do owner chọn hoặc writer không tuân lock đều là rủi ro còn lại. Giảm bằng trusted local acknowledgement, hash/policy/chronology/disjointness checks, explicit approval, backup và maintenance window. Tool có khả năng manual activation nhưng **không được gọi trên production trong phase này**. FP16 vẫn bị chặn, không fallback.

## 14. Manual verification steps for me

An toàn, không activation:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase18_release.py --audit-current
.\.venv\Scripts\python.exe -m pytest tests/test_release_governance.py tests/test_release_governance_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
.\.venv\Scripts\python.exe -m pip check
.\.venv-tensorrt\Scripts\python.exe -m pip check
```

Audit exit0 nghĩa là lệnh audit chạy thành công, **không** nghĩa candidate accepted. Kỳ vọng status BLOCKED, metrics null, accepted=false, FP16_REJECTED và hai fabric INACTIVE. Legacy flag `--activate-if-accepted` phải exit2 trước evaluation.

Khi thực sự có đủ evidence, tự tạo/pin bundle theo các schema trong `backend/app/core/release_governance.py`; ArtifactPin path phải absolute. Các placeholder dưới đây **không chạy được cho đến khi thay bằng input hợp lệ**, không phải evidence hoặc threshold mẫu được phê duyệt:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase18_release.py --bundle 'C:\path\to\release_bundle.json' --bundle-sha256 '<BUNDLE_SHA256>' --trusted-local --output-dir storage/models/release_governance/NEW_VERSION
```

Chỉ ghi report vào output version chưa tồn tại; lệnh này không activation. Exit2 nếu rejected/blocked, exit1 nếu malformed/hash/file error. Đạt criteria chỉ trả AWAITING_MANUAL_APPROVAL.

Manual invocation riêng chỉ được thực hiện sau review độc lập và maintenance window. Không dùng current project/probe cho bước này. Request JSON có đúng3 keys `report` (ArtifactPin), `registry_path` và `approval` (ManualApproval). Approval pin candidate/report/expected registry, có reviewer/reviewed_at/reason và confirm=`ACTIVATE:<candidate_id>:<report_sha256>`. Reference phải đã được chuẩn bị CANDIDATE/INACTIVE với các trường provenance/paths theo adapter; không tự register.

```powershell
.\.venv\Scripts\python.exe scripts/check_phase18_release.py --bundle 'C:\path\to\release_bundle.json' --bundle-sha256 '<BUNDLE_SHA256>' --trusted-local --manual-activation-request 'C:\path\to\reviewed_request.json' --request-sha256 '<REQUEST_SHA256>'
```

Không chạy manual command cho current project khi status vẫn BLOCKED. Chưa có lý do rollback production: không production mutation. Nếu muốn hoàn tác phần mềm, archive9 file mới và backup/revert chính phần thay đổi legacy CLI sau khi review; không reset worktree hoặc bật lại legacy activation bypass. Khi dùng adapter trong tương lai, backup path được trả về; restore có kiểm tra hash trong maintenance window, không tự rollback một registry đã thay đổi bởi người khác.

## 15. Git diff summary

Phase18: **9 file mới,1 file cũ sửa** (`scripts/evaluate_fabric_demo.py` +5/-17);380 file hiện hữu khác giữ SHA, không xóa file cũ. Tracked cumulative diff: **11 files changed,516 insertions(+),149 deletions(-)**, gồm các phase trước/user binary. Không stage/commit. Không ghi current production candidate/registry hoặc report acceptance giả.

## 16. Recommendation

**BLOCKED BY DATA / TRAINED CANDIDATE** cho acceptance/release. Phần mềm governance và manual adapter sẵn sàng để review sau khi toàn bộ test cuối pass; không phải production approval. Giữ nguyên Phase17 FP32 research baseline, FP16 REJECTED, synthetic/untrained probe không eligible. Không có production activation.

## 17. Tiến độ so với yêu cầu

| Yêu cầu | Trạng thái |
|---|---|
| Candidate/validation/fresh-test/report/manual flow | Đã triển khai offline, fail-closed |
| Configurable toàn bộ acceptance metrics + per-class | Đã có, template threshold null/chưa duyệt |
| Source/lineage/lot/roll/camera-disjoint và history protection | Đã kiểm tra metadata/hash và trusted custodian inventory |
| Manual approval, provenance, CAS/backup/audit | Đã triển khai và test TEMP, không chạy production |
| Chặn legacy bypass/FP16/research promotion | Đã triển khai; không thay runtime/artifacts |
| Trained candidate + fresh real test + SLA/criteria | Chưa có; BLOCKED |
| Production acceptance hoặc activation | KHÔNG thực hiện |
| Unit/integration/full regression/report |113 focused tests +1920 full tests +frontend/77 smoke PASS; báo cáo đã lưu |

Đã có nền tảng phần mềm đến Phase18 nhưng không đồng nghĩa18/18 production readiness. Để mở block: bổ sung verified real labels/acquisition lineage, trained/versioned candidate và full runtime recipe; chủ sở hữu duyệt criteria; custodian chuẩn bị fresh test độc lập với toàn training/calibration/reference/postmortem history; trusted evaluator cung cấp validation/full-hybrid evidence; sau đó mới xét report và manual approval. Không triển khai thêm phase sau báo cáo này.

============================================================
PHASE 18 IMPLEMENTATION COMPLETE — SOFTWARE INFRASTRUCTURE ONLY
STATUS: WAITING FOR USER REVIEW
RELEASE STATUS: BLOCKED BY DATA / TRAINED CANDIDATE
NO PRODUCTION CANDIDATE HAS BEEN ACTIVATED
NEXT PHASE HAS NOT BEEN STARTED
============================================================
