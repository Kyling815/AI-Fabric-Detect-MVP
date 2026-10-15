# Phase 5 — Fabric + Lot Memory Hierarchy

Ngày: 2026-10-14, Asia/Bangkok. Chỉ Phase 5; dừng để người dùng review, chưa bắt đầu Phase 6.

## 1. Goal

Biểu diễn memory Good toàn fabric và memory Good của current lot, hỗ trợ truy vấn fabric-global ∪ current-lot trong đường experimental. Không tự bật lot memory cho production; không tuyên bố cải thiện domain shift khi chưa có evaluation độc lập theo lot.

## 2. What I inspected before coding

- Đọc lại phạm vi Phase 5 và test/data/Git/report policy trong pasted request. Phase 4 đã được người dùng duyệt để tiếp tục.
- Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`. Các thay đổi Phase 0–4, executable, Revision.md, research report và test_image còn chưa stage/commit; được giữ nguyên.
- Backend FastAPI/service/SQLite, CurrentAnomalyDetector, Phase 4 ExperimentalAnomalyMemory, identity/config/checksum loaders và source metadata; frontend React/TypeScript/Vite/public schemas.
- Runtime chỉ truy vấn fabric-specific reference; acquisition lot_id không route memory. Phase 4 lot_id chỉ là nullable provenance.
- Registry A/B vẫn INACTIVE. V5 có bank 12,000 vectors, encoder/adapter v4, 128 dimensions, stride 8; các tệp này chỉ được đọc.
- Header của curated manifests và reference metadata có đường dẫn `lot_001`, nhưng manifest runtime không có bộ capture/evaluation độc lập có current-lot provenance đã xác minh. Không suy ra lot thật chỉ từ tên folder. Historic locked/postmortem data không được dùng hay gọi là fresh locked test.
- Không có AGENTS.md áp dụng trong workspace/ancestors đã kiểm tra. Snapshot 222 pre-existing source/document/tracked files; các Phase 5 additions không nằm trong scope snapshot này.

## 3. Implementation performed

### Experimental hierarchy

`LotMemoryConfig` frozen, default `enabled=False`, mode `global_only`, missing_lot_policy `error`. Example config chọn global_plus_lot nhưng vẫn enabled=false. Feature OFF luôn chọn global-only, không đọc/use lot artifact; thiếu global không tự activate lot.

`FabricLotMemoryManifest` schema 1, `EXPERIMENT_ONLY`, ghi fabric_id, current lot_id, policy và hai `MemoryArtifactSpec` tùy chọn. Mỗi child spec chứa directory, exact expected MemoryIdentity và MemoryExperimentConfig của Phase 4. Global child phải có lot_id=None; lot child phải khớp current lot. Các child reference versions độc lập, được pin/check với artifact riêng, không buộc global và lot phải cùng version string.

`FabricLotAnomalyMemory` hỗ trợ:

| Mode | Selected memory | Hành vi |
| --- | --- | --- |
| OFF/global_only | Global Good | Reuse exact child FAISS index; bỏ qua lot |
| lot_only, ON | Current lot Good | Reuse lot index; không cần/read global |
| global_plus_lot, ON | Global + current lot | Concatenate stored vectors, build compatible research index |

Union không chọn lại coreset, không normalize lại stored vectors, không weighting/fusion scores và không sinh training data. Duplicate vectors được giữ; không ảnh hưởng nearest distance nhưng có thể tăng memory. Query chuẩn hóa theo child vector policy đã chọn; current/cosine/squared-L2 vẫn giữ semantics Phase 4.

Kết quả frozen gồm distances/raw FAISS scores, nearest scope global/lot, local bank index, original selected source index, reference version, effective mode/fallback reason. Index cần đi kèm scope; không hiểu local index của lot là index trong global bank.

### Safety and fallback

- Chặn selected child khác fabric, lot scope, encoder_version, adapter_version, metric, vector policy hoặc feature dimension. Không trộn memory của vải khác.
- Fabric/lot hierarchy IDs dùng 1–128 ASCII letters/digits/._-, bắt đầu bằng letter/digit; blank/traversal/path separators/control characters bị reject. Current lot bắt buộc khi lot mode được chọn.
- Phase 4 loader kiểm tra exact expected identity/config, schema, hashes, dimension/count và FAISS stored vectors trước khi union.
- Missing lot mặc định lỗi. Chỉ global_plus_lot với `missing_lot_policy="global_only"` cho fallback rõ `missing_lot_memory`. Lot-only không fallback sang global.
- Directory không tồn tại hoặc spec=None được coi là missing; artifact hiện hữu nhưng corrupt/incomplete/version mismatch không bị che bằng fallback. Wrong identity bị reject trước missing-path handling.
- Không auto-infer Good, lot ID, approval hay current-lot provenance từ anomaly score/filename/curated folder. Không tự ghi/activate production reference.

### Evaluation script

`scripts/evaluate_fabric_lot_memory.py` so sánh fabric-only với fabric+lot trên prepared query embeddings. CLI default PLAN_ONLY; cần explicit `--enable-lot-memory` để so sánh. `--demo` là explicit synthetic-only fixture trong TemporaryDirectory.

`LotEvaluationManifest` ghi dataset/fabric/lot, encoder/adapter versions, research_evaluation split, independence_basis, aggregation policy và samples. Sample có id/image path+hash, embedding path+hash, normal/defect label, origin real/synthetic và optional parent_source_sha256.

Evaluator:

- Kiểm tra query fabric/lot/embedding-space version; source hashes và dimension/finite vectors.
- Reject reference/query overlap theo declared image id/hash/path và known parent hash, duplicate query identities/image hashes/paths; không dùng artifact bank file làm independent queries.
- Không nhận locked_test split. Đây không phải cơ chế biến historical test thành fresh holdout; source lineage chưa khai báo vẫn là hạn chế.
- Dùng cùng explicit `aggregate_image_score` cho cả hai nhánh. Trả per-image scores, nearest-lot patch count và AUROC tie-aware nếu có hai classes; one-class trả None. Không tune thresholds, emit production GOOD/REVIEW/BAD hay calibrated confidence.
- Missing-lot fallback được ghi NOT_COMPARED_MISSING_LOT_MEMORY, không báo completed comparison/AUROC benefit cho nhánh lot.
- Ghi origin/effective mode, fresh_locked_test=False và improvement_status=UNVERIFIED_PRODUCTION_BENEFIT; metrics chỉ mô tả tập research caller cung cấp.

Nearest patch distance của union là min(global, lot), trong floating-point tolerance. Điều đó không đảm bảo image score/accuracy tốt hơn: median-contrast aggregation có thể tăng dù mọi patch distances giảm; defect cũng có thể trở nên ít anomalous hơn. Có unit test phản ví dụ.

## 4. Files created

- `backend/app/core/lot_memory.py` — hierarchy config/spec/manifest và ID validation.
- `backend/app/ai/lot_memory.py` — strict selected scopes, load, union search và provenance result.
- `config/lot_memory_example.json` — default-OFF example.
- `backend/app/core/lot_evaluation.py` — query dataset/sample/aggregation contracts.
- `backend/app/ai/lot_evaluation.py` — paired comparison, overlap checks, AUROC reporting.
- `scripts/evaluate_fabric_lot_memory.py` — plan/opt-in comparison và synthetic demo.
- `scripts/check_phase5_memory_hierarchy.py` — cached real-model/global-bank diagnostics, synthetic lot in temp.
- `tests/lot_memory_fixtures.py` — deterministic research memories.
- `tests/test_lot_memory.py` — 55 hierarchy cases.
- `tests/test_lot_evaluation.py` — 35 evaluation/CLI cases.
- `tests/test_lot_memory_integration.py` — 7 pipeline/API integration cases.
- `docs/implementation/phase_5_memory_report.json` — diagnostic/demo result snapshots.
- `docs/implementation/phase_05_fabric_lot_memory.md` — báo cáo này.

## 5. Files modified

Không sửa tệp nào đã có trước Phase 5. Tất cả 222 pre-existing hashes giữ nguyên, gồm runtime service/detector/registry, tests/fixtures/docs Phase 0–4, frontend/dependencies và user documents.

Các cập nhật trong lượt này chỉ nằm trên Phase 5 additions. Không đổi private .env, Settings/API/DB contracts, CNN/adapter, embedding geometry, FAISS/threshold/checkpoint artifacts cũ hay dataset.

## 6. Architecture before

Production: fabric context → CurrentAnomalyDetector → cached CNN/adapter → fabric-specific FAISS → existing localization/QC → DB/API. Quality Gate và period observer tùy chọn của các phase trước giữ nguyên.

Research: Phase 4 có một standalone memory với strict provenance, optional metric/coreset và save/load; chưa chọn/union theo lot.

## 7. Architecture after

Production không thay đổi. Research có hierarchy manifest chọn global-only, lot-only hoặc compatible global+lot; output là distances và nearest-source provenance, không QC.

Evaluation chạy hai research configurations trên cùng approved query set, so sánh scores/AUROC theo cùng aggregation. Không có đường nối research result vào production registry/thresholds/API. Không triển khai period-aware pooling hoặc bất kỳ Phase 6 behavior nào.

## 8. Backward compatibility

Default OFF chọn exact global index; production không import/call hierarchy. Năm frozen POST/GET QC responses và OpenAPI vẫn pass, integration test cố tình khiến hierarchy.load ném lỗi nếu production gọi nó.

Các test mới truy vấn experimental lot có thể đổi research scores, nhưng không mutate original embeddings/evidence/candidates/QC. Không thêm HTTP lot selector hoặc DB fields.

Sáu cached CPU CNN/adapter/v5-global-bank diagnostics ở A/B xác nhận global-only/OFF patch scores exact baseline, union theo min rule trong 1e-6 tolerance và combined reload exact. Lot của diagnostic chỉ là 32 CNN vectors từ synthetic checker, không real approved current lot hay independent evaluation.

Protected verification:

| Scope | Result |
| --- | --- |
| 55 Phase 0 artifact/config/cache hashes | 55/55 khớp |
| storage: 311 files, gồm production DB/models/indexes/registries | Fingerprint khớp Phase 0 |
| Data: 643 files | Fingerprint khớp Phase 0 |
| User files: 10 files | Fingerprint khớp Phase 0, cùng scope path ordering |
| Cached weights: 3 checkpoint files | Fingerprint khớp Phase 0 |
| 222 pre-existing source/document/tracked files | Tất cả giữ nguyên hash |

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| Hierarchy unit tests | 3 metrics × 3 modes, default OFF, provenance mapping, missing/invalid lot, incompatible fabric/version/metric, reload | PASS — 55 cases |
| Evaluation/CLI tests | AUROC/ties/single-class, query identity/version/hash, leakage rejection, explicit opt-in, read-only workflow | PASS — 35 cases |
| Pipeline/API integration | Exact global anomaly parity, experimental isolation, five golden response/OpenAPI cases | PASS — 7 cases |
| Full regression | 389 prior + 97 new cases | PASS — 486 cases |
| Real-artifact diagnostics | CPU CNN/adapter + v5 bank + synthetic lot, no production activation | PASS — 6 cases |
| Synthetic evaluation demo | Fabric A-tagged toy drift, scores and honest origin/status reporting | PASS; no real benefit claim |
| Baseline smoke | Real artifacts, ten bank/index pairs, API/storage/error paths | PASS — 77 assertions |
| Frontend build | TypeScript/Vite compatibility | PASS — 22 modules |
| Syntax/JSON/Git/hash checks | New code/report validity and preservation | PASS |

## 10. Full regression test result

Lệnh thực tế từ workspace root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Kết quả cuối: `486 passed, 1 warning in 17.36s`. Frontend `tsc -b && vite build` PASS; smoke PASS 77 assertions. Warning FAISS/NumPy private namespace đã có từ baseline. Không cài/nâng cấp dependency, không sửa pre-existing tests.

Targeted trước bốn defensive cases bổ sung: `93 passed, 1 warning in 6.94s`. Tất cả 97 cases cuối được chạy trong full suite. Không có test failure cần đổi/weakening legacy expectations.

Diagnostic và demo commands:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase5_memory_hierarchy.py
.\.venv\Scripts\python.exe scripts/evaluate_fabric_lot_memory.py --demo
```

## 11. Hypothetical / synthetic cases tested

Unit hierarchy dùng normalized/raw 2D vectors với global/lot sources riêng; test các metrics current/cosine/squared L2, mode selection, invalid identifiers, corrupt/missing/schema-version artifacts và k/dimension errors. Union giữ scope/local/source indices và không renormalize/reselect bank.

Demo có global vector [1,0] và synthetic lot vector [0.8,0.6], bốn toy query images/vectors khác source image IDs/hashes. Cả fabric-only và fabric+lot AUROC đều 1.0, difference=0.0 trên fixture này; đây chỉ là output thuật toán trên dữ liệu dựng, tuyệt đối không phải model accuracy 100% hay gain thực tế. Một số defect scores cũng giảm khi thêm lot.

Real-model diagnostic đọc real global banks 12,000×128 và dùng synthetic lot 32×128. Checker self-retrieval và existing reference Good không độc lập với nguồn memory; diagnostic không gọi chúng là evaluation và không tính real-lot AUROC.

Không chạy historical locked/postmortem evaluation, không relabel/merge datasets. Chưa có independent current-lot Good + held-out query dataset được xác minh để đo domain-shift benefit của Fabric A. Infrastructure hoạt động, cải thiện thực tế vẫn UNVERIFIED_NO_INDEPENDENT_REAL_LOT_DATA.

## 12. Known limitations

- Chỉ research/offline hierarchy; chưa nối acquisition lot_id vào production inference, UI hay PLC. Không auto-create/update lot memory từ QC output.
- Folder `lot_001` trong dữ liệu curated không được coi tự động là actual production lot hoặc independent sample boundary. Cần lineage/approval/lot metadata đáng tin trước evaluation thực tế.
- Metadata/hashes kiểm tra expected identity và byte consistency, không chứng minh image thật sự Good, query label chính xác hay embeddings thật sự tạo từ encoder khai báo. Source independence vẫn một phần caller-declared.
- Overlap checks chỉ bao phủ IDs/hashes/paths/parent lineage đã khai báo. Không phát hiện mọi near-duplicate/crop descendant khi thiếu lineage. Nên dùng absolute source paths; source path tương đối được giải theo child artifact directory, query paths theo dataset manifest directory.
- Chỉ compatible encoder/adapter/metric/vector-policy/dim được union; coreset ratios/source threshold-profile tags của từng child có thể khác. Reference versions global/lot được pin riêng; không yêu cầu version string bằng nhau.
- Thêm Good memory có thể che defects/contamination; không đảm bảo improvement. Contrast aggregation không monotonic theo patch-distance reduction.
- Không có lot-specific calibrated threshold, QC decision, recall/FPR acceptance hoặc production throughput benchmark. AUROC cần hai classes; không đánh giá pixel masks/localization quality trong script này.
- Concatenation giữ duplicates và rebuild research index trong RAM; chưa có cache/eviction/large-bank optimization. Tie ordering phụ thuộc FAISS/environment, không hứa bitwise cross-device behavior.

## 13. Risks introduced

Production risk thấp vì zero existing-file edits và không route lot engine vào API. Caller phải opt-in research config/CLI; output vẫn EXPERIMENT_ONLY/uncalibrated.

Lot Good sai/contaminated hoặc sparse có thể giảm distance của cả defects; không promote experiment chỉ dựa trên score decrease. Cần independent evaluation và calibration trước bất kỳ bước production activation nào.

Union thêm RAM/index-build/search cost; hash/full-vector checks của Phase 4 loader cũng có overhead. Missing fallback nếu được cho phép có thể làm người dùng tưởng lot đang bật, nên result/evaluation luôn ghi effective_mode và fallback_reason.

Rollback: bỏ riêng 13 Phase 5 additions, giữ nguyên Phase 0–4 và user work. Không cần rollback production DB/model/index/registry vì không đổi. Không dùng git reset/clean hoặc delete rộng.

## 14. Manual verification steps for me

1. Chạy full regression/frontend/smoke và hai commands ở mục 10. Registry A/B thật vẫn INACTIVE; không activate candidate để review.
2. Xem `docs/implementation/phase_5_memory_report.json`: sáu software diagnostics, synthetic comparison và explicit unverified real-lot benefit.
3. Xem CLI:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_fabric_lot_memory.py --help
```

4. Với strict Phase 4 global/lot research artifact directories và prepared independent queries đã được duyệt, tạo hierarchy/evaluation JSON theo models ở `backend/app/core/lot_memory.py` và `lot_evaluation.py`.

Hierarchy cần fabric_id/current lot_id, config và global_memory/lot_memory specs. Mỗi spec có path, exact MemoryIdentity và explicit-enabled MemoryExperimentConfig. Global identity lot_id=None; lot identity lot_id khớp current lot. Child directories có bốn Phase 4 files; không dùng raw legacy v5 directory làm strict experiment artifact.

Evaluation JSON cần dataset_id, same fabric/lot, encoder/adapter versions, split=research_evaluation, independence_basis, samples với source/query hashes, label và origin. Pin same aggregation cho hai nhánh. Không đưa reference images/known descendants hoặc historical locked set vào đây rồi gọi chúng là fresh evaluation.

Default read-only plan:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_fabric_lot_memory.py --hierarchy path/to/hierarchy.json --dataset path/to/evaluation.json
```

Explicit research comparison, chỉ đọc artifact/query và in JSON stdout:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_fabric_lot_memory.py --hierarchy path/to/hierarchy.json --dataset path/to/evaluation.json --enable-lot-memory
```

`path/to/...` là placeholder phải thay bằng manifest thật. Flag comparison chọn enabled global_plus_lot, vẫn không production activation. Config sample ở `config/lot_memory_example.json` default OFF. Không sửa private .env hay production registry để review.

## 15. Git diff summary

Phase 5 riêng: 13 files created, 0 pre-existing files modified. 222 pre-existing hashes giữ nguyên; không stage/commit.

Actual tracked `git diff --stat` không tăng so với Phase 4 vì additions còn untracked; cumulative against HEAD gồm các phase trước và executable user:

```text
.env.example                               |   6 +
backend/app/core/config.py                 |   4 +
backend/app/services/fabric_registry.py    |   2 +
backend/app/services/inspection_service.py | 216 changed lines
tempCodeRunnerFile.exe                     | Bin 44395 -> 44626 bytes
5 files changed, 175 insertions(+), 53 deletions(-)
```

`git diff --check` không có whitespace errors, chỉ LF/CRLF warnings từ các existing dirty files. Staged diff trống. New research memory artifacts trong kiểm tra chỉ ở temp, được dọn khi diagnostic/demo kết thúc; không xóa material user data.

## 16. Recommendation

SAFE TO APPROVE — cho Phase 5 experimental hierarchy/evaluation infrastructure và compatibility đã kiểm tra. Benefit/domain-shift improvement của lot memory vẫn chưa được xác minh bằng independent real-lot evaluation; chưa approval cho production lot memory.

Dừng để review; chưa bắt đầu Phase 6.

```text
============================================================
PHASE 5 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
```
