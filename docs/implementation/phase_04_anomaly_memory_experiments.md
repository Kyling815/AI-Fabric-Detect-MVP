# Phase 4 — Anomaly Branch Abstraction + Memory Experiment Support

Ngày: 2026-10-14, Asia/Bangkok. Phạm vi: Phase 4 duy nhất; chờ review, chưa bắt đầu Phase 5.

## 1. Goal

Làm nhánh anomaly research-ready mà không thay thế ResNet18 layer2+3, projection adapter, FAISS hoặc memory riêng theo fabric. Thử metric/coreset trong đường offline, không chuyển production sang metric/bank chưa hiệu chuẩn.

## 2. What I inspected before coding

- Đọc lại phạm vi Phase 4 và các quy tắc test/data/Git/report trong pasted request; tiếp tục từ Phase 3 đã được người dùng duyệt.
- Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; Phase 0–3 và user files còn chưa stage/commit. Không có AGENTS.md áp dụng trong workspace/ancestors đã kiểm tra.
- Backend: FastAPI/service/SQLite, CurrentAnomalyDetector, dense scoring/localization, FAISS wrapper, FabricRegistry và reference builders. Frontend: React/TypeScript/Vite, public response còn dùng cosine cho nhánh dense.
- `DenseReferenceDetector` dùng inner product trên unit vectors rồi tính `max(0, 1-nearest)`. Không được áp công thức đó cho một index L2.
- `FaissIndexStore` đã hỗ trợ IndexFlatIP và IndexFlatL2, nhưng không tự normalize cosine vectors và không kiểm tra metric khi load. Không sửa wrapper cũ vì legacy callers còn phụ thuộc hành vi đó.
- Builder hiện dùng seeded random sampling với cap số vector; đây không phải k-center. Giữ nguyên builder/registries/artifacts cũ.
- 281 regression cases sau Phase 3; cached CPU CNN/adapter v4, 128 dimensions, stride 8; A/B thật vẫn INACTIVE. Snapshot 208 tệp hiện có trước khi sửa Phase 4.

## 3. Implementation performed

### Internal anomaly boundary

Thêm runtime-checkable `AnomalyDetector` Protocol với `inspect(InspectionInput, InspectionContext) -> DenseAnomalyEvidence`. Service dependency mang kiểu interface; implementation mặc định vẫn là CurrentAnomalyDetector. QC tiếp tục xử lý evidence riêng như trước.

### Production compatibility guards

CurrentAnomalyDetector kiểm tra trong reference-error scope, sau model extraction và trước scoring:

- Requested fabric phải khớp reference; encoder version trong context phải khớp shared backbone.
- Nếu metadata khai báo fabric/reference/encoder/adapter/version/schema/dimension, các giá trị phải khớp runtime/registry. Metadata schema hiện hỗ trợ version 1.
- Registry record khai báo `fabric_id` khác fabric được chọn bị từ chối ngay trước upload/AI.
- Metadata index/bank paths phải khớp registry paths khi được khai báo; actual index phải là inner-product compatible và metadata metric phải cosine hoặc không khai báo như legacy.
- Artifact `memory_experiment`/`EXPERIMENT_ONLY` không được dùng cho production, dù ai đó đặt registry record ACTIVE trong fixture.

Mismatch mới trả domain error `InvalidFabricReferenceError`, mapped HTTP 409 và giữ cleanup; native/model errors giữ scope cũ. Đây là hardening chủ đích cho dữ liệu không tương thích, không phải thay scoring trên reference hợp lệ.

Legacy metadata thiếu identity fields vẫn được chấp nhận để giữ baseline. Không tự đoán fabric/version/coreset ratio cho artifacts cũ; đây là hạn chế provenance được nêu ở mục 12.

### Offline memory experiments

`MemoryExperimentConfig` default `enabled=False`, metric `current`, ratio 1.0, seeded_random, seed 42. API build/load yêu cầu enabled=True; CLI cần flag `--enable-experiment` riêng, kể cả JSON có enabled=true. Không có flag chuyển production metric/coreset; không route experiment vào `/api/inspect`.

| Config metric | FAISS | Vector policy | Returned anomaly distance |
| --- | --- | --- | --- |
| current | IndexFlatIP | Require unit vectors; preserve bytes, không normalize lại | max(0, 1 - inner product) |
| cosine | IndexFlatIP | Normalize cả bank và query; reject zero/near-zero vectors | max(0, 1 - cosine similarity) |
| l2 | IndexFlatL2 | Giữ vector raw, kiểm tra finite/range | Squared Euclidean distance |

L2 ở đây là squared L2 theo output FAISS, không phải sqrt L2 và không phải Manhattan. L1/Manhattan, metric aliases và selector chưa implement bị reject; không fallback ngầm.

`CoresetSelector` Protocol và `SeededRandomCoresetSelector` lấy subset không replacement, indices sorted, số lượng `ceil(N × ratio)` tối thiểu 1. Ratio=100% giữ nguyên toàn bộ thứ tự/vector. Persist requested/actual ratio, seed, method và selected original indices. Các ratios 100/25/10/5% đều hỗ trợ, không tuyên bố ratio tối ưu hoặc selector là k-center.

`ExperimentalAnomalyMemory` chỉ search distance/nearest indices; không có QC, threshold calibration hay activation. Có score definition rõ ràng để tránh dùng lại threshold cosine cho L2/subset mới.

### Reproducible experimental reference metadata

Artifacts mới gồm `memory_bank.npy`, `memory_bank.faiss`, `selected_indices.npy`, `reference.json`. Metadata frozen có:

- `encoder_version`, `adapter_version`, `distance_metric`, `coreset_ratio`;
- `source_good_images`: id/path/SHA-256/fabric/lot/normal label/reference split;
- `fabric_id`, `lot_id`, `threshold_profile_version`, `reference_version`;
- requested metric, vector policy, score definition, actual ratio, seed/method, counts/dimension, canonical source embedding hash và ba artifact hashes;
- schema 1, `EXPERIMENT_ONLY`, `NOT_CALIBRATED`.

`lot_id` chỉ là provenance nullable trong Phase 4, không có lot-memory routing/merge/hierarchy. `threshold_profile_version` là tag provenance của profile nguồn hoặc null; không biến experiment thành calibrated. CLI kiểm tra source hashes; schema chỉ cho normal/reference hoặc research_reference declarations, không nhận validation/locked_test splits.

Loader bắt buộc expected identity/config, reject fabric/encoder/adapter/reference/lot/profile/config mismatch, unknown schema, path escape, checksum mismatch, metric/count/dimension mismatch và stored vectors khác bank. Saved unit vectors không normalize lại khi reload. Selection indices được kiểm tra theo seed/ratio đã khai báo.

`save_new` chỉ tạo directory mới, không overwrite cả directory rỗng đã tồn tại. Output dưới Data/production storage bị chặn; có thể dùng `storage/experiments/<new-name>` hoặc directory research ngoài các scope đó. Manifest ghi cuối; lỗi ghi giữa chừng có thể để lại artifact chưa hoàn chỉnh, không load/register/activate được.

## 4. Files created

- `backend/app/ai/anomaly_branch.py` — internal anomaly interface.
- `backend/app/ai/reference_compatibility.py` — production reference compatibility guard.
- `backend/app/ai/coreset.py` — selector protocol và seeded random implementation.
- `backend/app/ai/experimental_memory.py` — opt-in research build/search/save/load.
- `backend/app/core/memory_experiment.py` — config/provenance/reference metadata schemas.
- `config/memory_experiment_example.json` — default-disabled example.
- `scripts/experiment_anomaly_memory.py` — plan-only by default; explicit research build/query.
- `scripts/check_phase4_memory_support.py` — cached real-model/bank verification, writes only temp.
- `tests/memory_experiment_fixtures.py` — deterministic synthetic matrix/provenance fixtures.
- `tests/test_memory_experiment.py` — 78 unit cases.
- `tests/test_anomaly_branch_integration.py` — 23 pipeline/HTTP integration cases.
- `tests/test_memory_experiment_cli.py` — 7 CLI workflow cases.
- `docs/implementation/phase_4_memory_report.json` — real-artifact software verification results.
- `docs/implementation/phase_04_anomaly_memory_experiments.md` — báo cáo này.

## 5. Files modified

- `backend/app/ai/current_anomaly.py` — gọi guard trước scoring, giữ encoder/scoring/error scope.
- `backend/app/services/fabric_registry.py` — reject declared wrong-fabric record.
- `backend/app/services/inspection_service.py` — dùng Protocol annotation, vẫn default CurrentAnomalyDetector.

Không sửa FAISS wrapper, dense scoring/localization, CNN/adapter, reference builder, thresholds, contracts Phase 1–3, tests cũ, private .env, dependency files, frontend, API/DB schemas hay user documents.

## 6. Architecture before

InspectionService phụ thuộc CurrentAnomalyDetector cụ thể. Detector dùng cached ResNet18/adapter, reference FAISS cosine và dense localization; service áp QC, lưu DB/ảnh rồi trả API. Builder cũ chọn seeded random subset theo cap và ghi artifact/registry theo workflow riêng.

## 7. Architecture after

Service phụ thuộc AnomalyDetector interface; default production implementation và luồng CNN → FAISS → QC giữ nguyên, có thêm compatibility checks cho reference đã chọn. Quality Gate và period observer Phase 2/3 không thay đổi.

Đường research độc lập: supplied approved-Good embedding matrix + strict provenance + opt-in config → selector → compatible FAISS → offline distances → fresh experiment artifact. Không có cạnh nối output research vào production registry, thresholds hay QC. Phase 5 lot-memory hierarchy chưa implement.

## 8. Backward compatibility

Năm QC golden snapshots POST/GET/OpenAPI giữ nguyên; 281 case cũ đều pass. L2/coreset không được bật trong runtime; public anomaly_summary vẫn mô tả cosine đúng với production path.

Sáu real CNN/adapter-v4/v5-bank cases cho hai fabrics có patch_scores, anomaly_map, localized_score và boxes bằng exact baseline scorer. Với current+100%, bank bytes/order và sample nearest distances khớp baseline. Các giá trị source image_score trùng báo cáo Phase 3.

24 research configurations (2 fabrics × 3 metric choices × 4 ratios) save/reload reproduces exact nearest indices/distances. New artifacts nằm trong TemporaryDirectory và được dọn sau kiểm tra. Registry A/B thật không được activate hay sửa.

Hash verification:

| Protected scope | Result |
| --- | --- |
| 55 Phase 0 artifact/config/cache hashes | 55/55 khớp |
| storage: 311 files, gồm DB/checkpoints/indexes/registries | Fingerprint khớp Phase 0 |
| Data: 643 files | Fingerprint khớp Phase 0 |
| user files: 10 files | Fingerprint khớp Phase 0, đúng thứ tự scope ban đầu |
| cached weights: 3 checkpoint files | Fingerprint khớp Phase 0 |
| 208 unique pre-existing source/document/tracked files | Chỉ ba tệp ở mục 5 thay đổi; 205 tệp khác giữ nguyên |

Legacy service từ `create_placeholder_inspection` trở đi giữ nguyên byte. Không sửa các pre-existing tests hoặc fixture snapshots.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| Experimental memory/coreset/config | NN determinism, reload, normalization, 100% parity, ratios, provenance, checksum/metric failures | PASS — 78 cases |
| Branch/runtime integration | Protocol default, five frozen QC responses, wrong-fabric/version/index rejection, research artifact rejection | PASS — 23 cases |
| CLI workflow | Read-only plan gate, explicit builds, source hash rejection, no overwrite | PASS — 7 cases |
| Full regression | 281 prior cases + 108 new cases | PASS — 389 cases |
| Real production component parity | Cached CPU CNN/adapter and original v5 FAISS | PASS — 6 cases |
| Real research memory configurations | current/cosine/l2 × 100/25/10/5% × A/B, save/reload | PASS — 24 configurations |
| Phase 0 smoke | Model, 10 bank/index pairs, API/storage/error compatibility | PASS — 77 assertions |
| Frontend build | TypeScript/Vite | PASS — 22 modules |
| Syntax/JSON/Git/hash checks | New source syntax, report schema, no whitespace errors, protected state | PASS |

## 10. Full regression test result

Lệnh thực tế từ workspace root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Full suite: `389 passed, 1 warning in 18.44s`. Frontend `tsc -b && vite build` PASS; smoke PASS 77 assertions. Warning FAISS/NumPy private namespace đã có từ baseline. Không cài/nâng cấp package.

Targeted lần đầu sau integration/CLI: 98 passed; sau đó bổ sung 10 defensive config/selector/schema cases, tất cả được chạy trong full suite. Không có test failure cần sửa/weakening legacy tests.

Real verification command:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase4_memory_support.py
```

Script này không import/init production DB, không sửa registry; đọc cached model/v5 references và tạo artifacts tạm. JSON kết quả lưu trong `phase_4_memory_report.json`.

## 11. Hypothetical / synthetic cases tested

Unit tests dùng deterministic 100×8 normalized random matrices, duplicated vectors/ties, unnormalized vectors, zero/NaN/Inf/complex/empty inputs và known L2 query (3,4) so với (0,0), squared distance 25. Thử ratios 100/25/10/5%, small bank với ceil/minimum-one và k không hợp lệ.

Integration/CLI dùng fake encoder nhưng FAISS/SQLite/HTTP thật trong temp; mô phỏng wrong fabric/encoder/adapter/reference/schema/dimension, index L2 giả cosine, path mismatch, corrupt/swapped artifacts, source hashes sai, disabled gate và existing output.

Real cached CNN verification dùng checker, scratch synthetic và ảnh Good đã nằm trong reference sources. Đây là software parity, không phải evaluation trên unseen/fresh locked test. Ratio thực nghiệm được áp dụng lên 12,000-vector v5 input bank đã là historical subset: 100%=12,000; 25%=3,000; 10%=1,200; 5%=600. Không được gọi những con số này là ratio so với toàn bộ feature gốc hoặc accuracy improvement.

## 12. Known limitations

- Selector duy nhất hiện tại là seeded random, không k-center/coverage optimization. Subset có thể mất rare normal modes; không có ratio tối ưu đã được xác nhận.
- Metric hoặc subset mới cần calibration/evaluation riêng; không reuse threshold gốc để ra QC. L2 squared distances không cùng scale với cosine distances.
- Legacy reference thiếu identity metadata vẫn được chấp nhận theo registry binding; không thể xác minh identity bị lược bỏ chỉ bằng shape/index. Guard kiểm tra declared provenance, không retrofit hay cryptographically authenticate legacy artifacts.
- Metadata/hash bảo đảm artifact consistency và expected identity/config, không chứng minh declared image thật sự Good hay supplied embeddings thực sự được tạo từ source images bằng encoder khai báo. CLI xác minh source bytes; caller vẫn phải cung cấp approved reference provenance trung thực.
- `source_embeddings_sha256` hash canonical float32 input matrix, không hash toàn bộ .npy file. Version tags không thay thế training/config manifest hay checkpoint hash của pipeline tạo embeddings; Phase 0 đã ghi hash model hiện có.
- `threshold_profile_version` có thể chỉ là tag của source profile; mọi output vẫn NOT_CALIBRATED. `lot_id` là metadata, không lot-memory hierarchy hoặc fusion.
- Chưa benchmark production latency/throughput, peak memory hoặc unseen detection accuracy. Query results là distances/indices, không calibrated confidence.
- Exact repeatability được kiểm tra CPU/FAISS/environment hiện tại; không hứa tie ordering bitwise giống nhau trên GPU/library khác. Persisted indices/hashes giúp phát hiện thay đổi.

## 13. Risks introduced

Production guards có thể từ chối artifacts trước đây load được nhưng có declared provenance sai hoặc index không phù hợp; đây là fail-safe chủ đích. Không auto-correct field/index/threshold để lách validation.

Offline build/load cần RAM và disk; có copy matrix, hash files và kiểm tra tất cả stored vectors, chưa tối ưu cho bank rất lớn. Artifact ghi dở khi IO failure vẫn có thể tồn tại ở output mới; không tự delete để tránh mất dữ liệu phục hồi, không publish/register nó.

Research CLI khi được explicit opt-in sẽ ghi bốn artifact files trong directory mới. Trong lượt triển khai này, mọi new memory artifacts chỉ ở temp; production storage/data/user scopes không đổi. Không có rollback DB/index/model cần thực hiện.

Rollback source: bỏ đúng Phase 4 hunks trong ba modified files và 14 additions ở mục 4, giữ nguyên Phase 0–3/user work. Bỏ guard làm mất protection với wrong-fabric/metric mismatches; không dùng reset/clean toàn repo.

## 14. Manual verification steps for me

1. Chạy full runner ở mục 10 và cached real-artifact checker. A/B thật vẫn INACTIVE; không activate candidates để review.
2. Đọc `docs/implementation/phase_4_memory_report.json` để xem sáu parity cases và 24 configurations.
3. Demo unit-only với synthetic vectors và temp artifacts, không DB/camera/model:

```powershell
@'
from pathlib import Path
from tempfile import TemporaryDirectory
from backend.app.ai.experimental_memory import ExperimentalAnomalyMemory
from tests.memory_experiment_fixtures import bank, config, identity
with TemporaryDirectory(prefix='fabric-phase4-manual-') as directory:
    for metric in ('current', 'cosine', 'l2'):
        for ratio in (1.0, 0.25, 0.10, 0.05):
            options, provenance = config(metric, ratio), identity()
            memory = ExperimentalAnomalyMemory.build(bank(), identity=provenance, config=options)
            path = Path(directory) / f'{metric}_{ratio}'
            metadata = memory.save_new(path)
            restored = ExperimentalAnomalyMemory.load(path, expected_identity=provenance, config=options)
            print(metric, ratio, metadata.memory_bank_count, restored.search(bank()[:1]).anomaly_distances.tolist())
'@ | .\.venv\Scripts\python.exe -
```

4. CLI help:

```powershell
.\.venv\Scripts\python.exe scripts/experiment_anomaly_memory.py --help
```

Với embedding/provenance JSON đã chuẩn bị và nguồn Good được duyệt, truyền `--embeddings`, `--provenance`, `--config`, `--output-dir`; mặc định chỉ PLAN_ONLY, không ghi. Chỉ thêm `--enable-experiment` khi chủ động muốn tạo một experiment directory mới. Có thể truyền `--query-embeddings` để nhận distances. Chưa có migration/build từ raw images trong phase này.

Example config mặc định OFF ở `config/memory_experiment_example.json`. Khi API build/load được dùng trực tiếp, caller phải set enabled=True rõ ràng. Không cần sửa private .env, production Settings hay reference registry.

## 15. Git diff summary

Phase 4 riêng: ba existing files modified và 14 files created; 205/208 unique pre-existing hashes khác giữ nguyên. Trong snapshot raw có path separator duplicates; số unique đã được normalize/deduplicate trước khi báo cáo.

Actual tracked `git diff --stat` cumulative against HEAD, bao gồm Phase 1–3 và executable user có sẵn:

```text
.env.example                               |   6 +
backend/app/core/config.py                 |   4 +
backend/app/services/fabric_registry.py    |   2 +
backend/app/services/inspection_service.py | 216 changed lines
tempCodeRunnerFile.exe                     | Bin 44395 -> 44626 bytes
5 files changed, 175 insertions(+), 53 deletions(-)
```

CurrentAnomalyDetector và Phase 4 additions còn untracked nên chưa được tính trong plain tracked diff. `git diff --check` không có whitespace errors, chỉ LF/CRLF warnings. Staged diff trống; không commit/stage/delete material data.

## 16. Recommendation

SAFE TO APPROVE — cho Phase 4 anomaly abstraction, compatibility guards và offline memory experiment infrastructure. Không phải phê duyệt metric/coreset ratio mới cho production hoặc xác nhận accuracy improvement.

Dừng tại đây, chờ review và explicit approval trước Phase 5.

```text
============================================================
PHASE 4 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
```
