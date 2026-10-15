# Phase 1 — Runtime Contracts and Inspection Service Refactor

Date: 2026-10-13, Asia/Bangkok. Phase 0 was approved by the user before this work. Phase 2 has not been started.

## 1. Goal

Tách orchestration và CNN/FAISS inference thành các thành phần có contract nội bộ, giữ nguyên hành vi bên ngoài, model artifacts và kết quả inference hiện tại. Không thêm thuật toán AI, QC rule hoặc preprocessing mới.

## 2. What I inspected before coding

- Nhánh `main`, baseline commit `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`.
- Các thay đổi có sẵn: `tempCodeRunnerFile.exe`; `Revision.md`, `deep-research-report.md`, `test_image/`; tài liệu/runner Phase 0 chưa được stage. Không ghi đè các phần này.
- Đọc lại actual `/api/inspect`, `InspectionService.create_fabric_inspection`, model loaders, dense detector, registry, decisions, bbox mapping, storage/CRUD/schema và các fixture/test hiện có.
- Xác nhận không có `AGENTS.md` mới; không có commit/stage tự động được yêu cầu.
- Chụp snapshot của orchestration cũ trước khi sửa: năm decision cases, POST/GET responses và OpenAPI request/response schema. Snapshot dùng fake encoder cố định, real FAISS, SQLite và API hiện có.
- Lưu checksum 176 tracked files trước thay đổi; đối chiếu artifact/config/cache với freeze Phase 0.

Các điểm baseline cần giữ: confidence null; Good vẫn có thể chứa bbox; GET persisted inspection không khôi phục transient embedding/anomaly summaries; legacy patch path không phải public dense route.

## 3. Implementation performed

### Internal contracts

Thêm `backend/app/core/inspection_contracts.py` với:

- `InspectionInput`: đường dẫn ảnh đã lưu và fabric ID.
- `InspectionContext`: selected FabricReferenceEntry, backbone configuration, device và resize fallback.
- `DefectCandidate`: alias của `MergedBox` hiện có, giữ nguyên geometry/score/patch_count/area.
- `DenseAnomalyEvidence`: dense features, extractor info và existing DenseAnomalyResult.
- `InspectionResult`: evidence, mapped candidates, existing FabricDecision và area ratio.

Contract dùng frozen dataclasses ở mức field assignment; không deep-freeze NumPy arrays hay threshold dictionaries. Không thêm validation policy hoặc public schema.

### Callable anomaly component

Thêm `CurrentAnomalyDetector.inspect(input, context)` tại `backend/app/ai/current_anomaly.py`. Component tiếp tục gọi cached ResNet18/adapter và DenseReferenceDetector, với cùng arguments, scale, index metric, thresholds và return data.

Component không import FastAPI hoặc HTTPException. Các reference-only lỗi `OSError`, `KeyError`, `ValueError` được bọc vào `InvalidFabricReferenceError`; service chuyển thành HTTP 409 với cùng error prefix. Model-loading/native FAISS failures vẫn nằm ngoài catch này để giữ phạm vi xử lý lỗi baseline.

### Existing service orchestration

`InspectionService` vẫn ở vị trí hiện tại, với entrypoint cũ. Thêm:

- `_resolve_fabric_context`: gọi FabricRegistry và giữ nguyên HTTP error mapping.
- `_inspect_fabric`: gọi anomaly component, dùng nguyên box mapping/decision/area helpers.
- optional keyword injection `anomaly_detector` để kiểm thử orchestration riêng; cách gọi cũ `InspectionService(db)` vẫn hợp lệ.

Public method tiếp tục xử lý upload/decode, HTTP translation, annotation, SQLite persistence, response formatting và cleanup. Không tạo song song một runtime/service package mới.

Giả định: FabricRegistry hiện tại vẫn là nguồn authority cho selected reference; internal context được tạo bởi service. Type annotations không thay thế validation của registry. Phase 1 không cố chữa metadata identity checking hoặc error policy đã được ghi nhận ở baseline.

## 4. Files created

| File | Purpose |
| --- | --- |
| `backend/app/core/inspection_contracts.py` | Internal contracts, reuse existing candidate/evidence/decision types |
| `backend/app/ai/current_anomaly.py` | HTTP-independent current CNN/FAISS component |
| `tests/phase1_fixtures.py` | Temporary fake-encoder/real-FAISS/SQLite/API fixtures |
| `tests/test_inspection_contracts.py` | Contract unit tests |
| `tests/test_current_anomaly.py` | Component unit/integration and exception-boundary tests |
| `tests/test_inspection_service_phase1.py` | Service/API parity, persistence and failure regression |
| `tests/fixtures/phase_1_dense_baseline.json` | Pre-refactor response and schema snapshot |
| `docs/implementation/phase_1_parity_report.json` | Six actual encoder/index old-vs-new parity cases |
| `docs/implementation/phase_01_runtime_refactor.md` | This phase report |

## 5. Files modified

Only `backend/app/services/inspection_service.py`: orchestration now delegates CNN/FAISS and carries typed internal data. The tracked diff has 59 insertions and 52 deletions.

No pre-existing test file, API route, public schema, frontend, settings, registry, database schema, AI algorithm, checkpoint, FAISS index, threshold profile or Phase 0 artifact/document was modified. Exact source comparison confirms `create_placeholder_inspection` and all subsequent service methods remained unchanged.

## 6. Architecture before

```text
API -> InspectionService.create_fabric_inspection
        -> FabricRegistry
        -> upload/decode
        -> extractor loading + feature extraction
        -> DenseReferenceDetector loading + scoring
        -> box mapping + decision + area
        -> annotation + SQLite + response
```

CNN/FAISS calls and reference HTTP translation were inline in the same public method.

## 7. Architecture after

```text
API (unchanged)
 -> InspectionService.create_fabric_inspection
     -> _resolve_fabric_context -> InspectionContext
     -> upload/decode -> InspectionInput
     -> _inspect_fabric
         -> CurrentAnomalyDetector.inspect
             -> existing encoder + DenseReferenceDetector
             -> DenseAnomalyEvidence
         -> existing box mapping + QC rules
         -> InspectionResult
     -> existing annotation + SQLite + response
```

Diff-level summary:

```diff
- CNN/FAISS calls inline in public service method
+ callable CurrentAnomalyDetector with internal input/context/evidence
+ InspectionResult around existing mapped boxes and QC decision
  /api/inspect + response schema + algorithms + artifacts: unchanged
  legacy service methods: unchanged
```

## 8. Backward compatibility

Preserved:

- multipart `image + required fabric_id`, HTTP 201 success and all original response fields;
- HTTP 400 unknown fabric/decode, 409 inactive/missing/incompatible reference, 413 upload limit, 422 missing fields;
- `GOOD`, `NEEDS_REVIEW`, `BAD`, existing reason strings and dense confidence null;
- model factory arguments, CPU/CUDA resolution, 384-dimensional base and existing projection behavior;
- cosine/IP FAISS scoring, image/localized aggregation, thresholding, morphology, bbox mapping and QC rules;
- model/reference identities, SQLite writes, images, cleanup and legacy service behavior;
- native/model failure status behavior, including existing HTTP 500 paths.

Five complete POST/GET response snapshots and OpenAPI schemas match the pre-refactor baseline exactly after normalizing only IDs, timestamps and generated image URLs. Six additional real-model comparisons match all other response fields exactly and produce byte-identical annotated PNG hashes.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| 4 contract unit cases | Frozen fields, existing candidate reuse, reference/configuration, separate evidence/decision | PASS |
| 8 component cases | Existing extractor args, resize fallback, real FAISS, typed errors, preserved exception scope | PASS |
| 5 response snapshot cases | Good, review without box, Bad, review below bad, strong localized review | PASS |
| OpenAPI snapshot | Exact existing inspect request/response contract | PASS |
| Service injection unit case | Orchestration can consume component evidence without DB/HTTP | PASS |
| 9 registry rejection cases | Unknown/inactive/no-active/inactive-ref/version/missing artifacts/thresholds; reject before upload/model | PASS |
| 6 failure/cleanup cases | Decode, metadata, dimensions, native FAISS, encoder, persistence | PASS |
| Original 95 tests | Existing behavior regression | PASS |
| Frontend build | Existing TypeScript/Vite compatibility | PASS |
| Phase 0 smoke | Existing model/artifacts and full API persistence | PASS — 77 assertions |
| Actual model parity | Old/new service with actual v4 encoder/adapter and A/B v5 indexes | PASS — 6 cases |
| Baseline checksum checks | Artifact/config/cache and non-target source preservation | PASS |

First targeted run: 5 failed, 29 passed. Classification: implementation error in the new test fixture. Static mounts were bound to the initial app directory before per-test settings changed, producing image-route 404s. Fix: construct each fixture app after storage overrides. The static-serving assertions were retained. No legacy test was modified or weakened; production code was not changed to resolve this fixture issue.

## 10. Full regression test result

Focused Phase 1 execution ran `python -m pytest -q` for the three new test modules in a subprocess with temp DB/storage environment:

```text
34 passed, 1 warning in 4.52s
```

Full regression/build/smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py --regression --frontend
```

Actual final results:

```text
129 passed, 1 warning in 6.11s
Frontend: vite v8.0.16; 22 modules transformed; build PASS
Smoke: 77 assertions PASS
Exit code: 0
```

129 = 95 original tests + 34 new Phase 1 cases. No original tests broke, changed, skipped or were removed. The warning is the baseline FAISS/NumPy deprecation. Environment/dependencies remained the Phase 0 CPU environment.

## 11. Hypothetical / synthetic cases tested

Golden fixture: RGB 48×32 image, deterministic six 2-dimensional embeddings, processed geometry 24×16 at scale 0.5, and a real FAISS index with one normalized reference vector. Thresholds exercise all existing decision branches; they are fixture values, not proposed production thresholds.

The actual-model parity run loaded the old service from Git commit `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572` into an in-memory module, and ran old/new services against the same inputs with cached pretrained ResNet18, adapter v4 and existing v5 FAISS indexes. DB/storage/registry copies were temporary. The production registry stayed INACTIVE.

| Fabric | Input | Full response equality | Annotated SHA-256 equality |
| --- | --- | --- | --- |
| A | Synthetic checker | PASS | PASS |
| A | Synthetic scratch | PASS | PASS |
| A | Existing reference Good image | PASS | PASS |
| B | Synthetic checker | PASS | PASS |
| B | Synthetic scratch | PASS | PASS |
| B | Existing reference Good image | PASS | PASS |

Existing Good images belong to the reference source set. These checks verify parity, not generalization, accuracy or localization quality. No locked-test images were evaluated. Recorded results and scope are in [phase_1_parity_report.json](phase_1_parity_report.json).

## 12. Known limitations

- Golden encoder is synthetic; actual-model parity covers six CPU cases, not every possible input/device.
- Tests deliberately retain baseline native FAISS/model errors returning HTTP 500. Improving that policy would change behavior and is outside this refactor.
- Internal frozen contracts are shallow. Existing threshold dictionaries/arrays are reused; no shared configuration mutation is introduced.
- Registry metadata identity checks, documentation/model-info drift, domain shift and pair-level training leakage risks remain as baseline issues.
- No segmentation, quality gate, pattern estimation, camera/encoder acquisition, fusion or physical coordinates were implemented.
- Fabric A/B are still INACTIVE and no production-ready model is established.

## 13. Risks introduced

The new import boundary and context/evidence handoff are additional integration points. Unit tests, original regression, complete response/schema snapshots, real-artifact smoke and old/new parity cover the exercised paths. There is no change to model weights, feature computation or threshold values.

The optional injected detector is internal; normal callers retain the default existing algorithm. The alias for DefectCandidate avoids divergent bbox representations.

Rollback notes: restore only the Phase 1 hunks in `inspection_service.py` to the inspected baseline, then remove the nine Phase 1 additions if needed. Keep Phase 0 files and unrelated user changes. Do not reset/clean the whole repository; no DB/model/index rollback or migration is needed.

## 14. Manual verification steps for me

From repository root:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py --regression --frontend
```

Expect 129 test passes, frontend build pass, 77 smoke assertions and exit code 0. This command isolates DB/storage for tests and does not activate production references.

Inspect the pre-refactor snapshot and new regression cases:

```powershell
Get-Content tests/fixtures/phase_1_dense_baseline.json -Encoding UTF8
Get-Content docs/implementation/phase_1_parity_report.json -Encoding UTF8
git diff -- backend/app/services/inspection_service.py
```

Verify artifacts/config/cache still match Phase 0:

```powershell
$baseline = Get-Content docs/implementation/phase_0_baseline_manifest.json -Raw -Encoding UTF8 | ConvertFrom-Json
$baseline.artifacts | ForEach-Object {
    $actual = (Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash
    [PSCustomObject]@{Path=$_.path;Matches=($actual -eq $_.sha256)}
}
```

Expect all `Matches=True`. On the normal UI, A/B remain disabled because their production registry is INACTIVE. Successful inspection branches are exercised in the temporary runner/test fixtures; do not activate a candidate to manually verify this phase.

## 15. Git diff summary

Actual tracked diff at Phase 1 completion:

```text
backend/app/services/inspection_service.py | 111 changes
tempCodeRunnerFile.exe                     | Bin 44395 -> 44626 bytes
2 files changed, 59 insertions(+), 52 deletions(-)
```

The executable change predates this work. Phase 1 modifies one tracked file and adds nine untracked files listed above; plain `git diff --stat` excludes additions until staged. Existing 175 other tracked-file hashes match the pre-Phase-1 snapshot, including all original tests. No stage or commit was performed.

## 16. Recommendation

SAFE TO APPROVE — Phase 1 runtime contract/orchestration refactor has zero intentional inference changes in the exercised paths and passes the full regression/parity checks. This does not approve a production model/reference.

Await explicit user approval to continue to Phase 2.

```text
============================================================
PHASE 1 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
```
