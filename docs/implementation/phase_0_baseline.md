# Phase 0 — Baseline Audit and Reproducibility Freeze

Date: 2026-10-13, Asia/Bangkok. Status: waiting for user review.

## 1. Goal

Hiểu và ghi lại hệ thống hiện tại trước khi tiến hóa sang PAH-CNN; kiểm tra khả năng tái lập bằng môi trường, checksum và smoke test. Phase 0 chỉ bổ sung tài liệu và công cụ chẩn đoán. Chưa triển khai Phase 1.

## 2. What I inspected before coding

- Nhánh `main`, commit `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`, remote GitHub đúng dự án được yêu cầu.
- Thay đổi sẵn: ` M tempCodeRunnerFile.exe`; `?? Revision.md`; `?? deep-research-report.md`; `?? test_image/`. Không có thay đổi runtime chưa commit.
- Inventory 176 tracked paths; parse/inventory 115 Python files, 13.548 dòng; kiểm tra API, services, AI dense/legacy, preprocessing, model loaders, DB/schema, scripts, frontend và config/docs.
- Nạp metadata của 10 reference v1-v5, hai registry, checkpoint và các threshold profile. Kiểm tra 311 file storage, 643 file Data bằng checksum; xem DB read-only.
- Entry point thực tế: `backend.main:app` → `routes_inspection.inspect_image` → `InspectionService.create_fabric_inspection`.
- Không tìm thấy `AGENTS.md` áp dụng. Không có lệnh tự động commit.
- Các điểm mơ hồ: README lẫn hợp đồng `reference_id` cũ; registry legacy có active pointer nhưng Fabric A/B đều INACTIVE; model-info chứa thông số legacy; không có frontend test runner.

Architecture map và phân tích chi tiết nằm tại [current_baseline.md](../architecture/current_baseline.md).

## 3. Implementation performed

1. Tạo tài liệu baseline mô tả đúng runtime, API, model-loading, feature extraction, projection, FAISS, localization, decision, DB và frontend.
2. Tạo manifest ghi môi trường, dependency snapshot, 55 checksum artifact/config/cache, commit và trạng thái Git ban đầu.
3. Tạo runner chẩn đoán `scripts/check_phase0_baseline.py` để tái chạy regression/frontend và smoke test.
4. Runner dùng DB/storage/registry fixture trong `TemporaryDirectory`. Nó nạp model/index thật read-only, chặn download weights và không gọi builder, calibration, training hay activation hiện có.
5. Kiểm tra đầy đủ ba nhánh quyết định bằng fixture synthetic và threshold thử nghiệm chỉ trong registry tạm; xác minh persistence và ảnh qua API.

Giả định: chạy từ root repo bằng `.venv` hiện có; pretrained weights và các local reference artifact có sẵn; CPU là thiết bị đã được xác minh. Registry fixture ACTIVE chỉ phục vụ kiểm tra orchestration, không phải reference production mới.

## 4. Files created

- `docs/architecture/current_baseline.md` — bản đồ kiến trúc và các giới hạn audit.
- `docs/implementation/phase_0_baseline.md` — báo cáo Phase 0 này.
- `docs/implementation/phase_0_baseline_manifest.json` — môi trường và checksum.
- `scripts/check_phase0_baseline.py` — công cụ smoke/regression không được gọi bởi production runtime.

## 5. Files modified

Không sửa file hiện có. Không sửa test hiện có, `.env`, dependency declarations, lockfile, frontend, API/service/AI runtime, database, checkpoint, FAISS index hoặc registry production. Các thay đổi người dùng có sẵn được giữ nguyên.

## 6. Architecture before

```text
React -> POST /api/inspect(image, fabric_id)
 -> FabricRegistry -> cached ResNet18 layer2+3
 -> adapter v4 -> per-fabric FAISS cosine
 -> anomaly map -> localization -> rule decision
 -> SQLite + upload/annotated images -> response
```

Legacy patch service, preprocessing v1/v2 và ReferenceRegistry cùng tồn tại; public inspect dùng dense branch.

## 7. Architecture after

```diff
  Production architecture: unchanged
+ Architecture baseline document
+ Environment / artifact SHA-256 manifest
+ Offline diagnostic runner
  API / model / thresholds / registry / DB / frontend: unchanged
```

Không thêm contract, quality gate, period estimator, segmentation hoặc fusion vào runtime ở Phase 0.

## 8. Backward compatibility

Giữ nguyên multipart `image + fabric_id`, HTTP status codes, response schema, `GOOD / NEEDS_REVIEW / BAD`, confidence null của dense branch, behavior từ chối fabric INACTIVE và không fallback sang fabric khác.

SHA-256 trước/sau khớp trên toàn bộ original tracked files (176), storage (311), Data (643), user files (10) và torch cache (3). Kiểm tra bao gồm DB, checkpoint, index, threshold, manifests và dữ liệu locked có trong các scope; không có thay đổi byte.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| Existing pytest unit tests | Scoring, FAISS, decisions, preprocessing, registries, geometry, preparation | PASS |
| Existing integration tests | Reference build, preprocessing batch, upload errors, feedback | PASS |
| Entire original backend suite | Regression trên current checkout | PASS — 95 tests |
| Frontend `npm run build` | TypeScript compile và Vite production bundle | PASS — 22 modules |
| `pip check` | Tính tương thích dependency đã cài | PASS — no broken requirements |
| Registry load | Fabric registry và legacy registry thực tế | PASS |
| Real model load/forward | Cached pretrained ResNet18 + adapter v4, shape/finite/L2/repeatability | PASS — CPU |
| Ten existing bank/index pairs | Shape, stored-vector parity trên 16 vector đầu, normalized self-query | PASS |
| Compatible v4/v5 indexes | Encoder hiện tại chấm synthetic query qua index thật | PASS |
| Existing A/B API behavior | Copy registry thực tế vào sandbox; cả hai fabric trả 409 | PASS |
| Synthetic successful inspect | GOOD, NEEDS_REVIEW, BAD qua API/model/FAISS/DB | PASS — 201 |
| Persistence/static routes | GET saved inspection, upload image, annotated image | PASS |
| Error smoke | 422 missing field; 400 unknown/MIME/decode; 409 inactive/missing/invalid metadata/dimension | PASS |
| Failed decode cleanup | Không để lại upload lỗi trong sandbox | PASS |
| Protected scope SHA-256 | Xác minh không thay đổi dữ liệu/runtime/artifact | PASS |

Runner có 77 assertion PASS, tách biệt với 95 pytest cases. Không dùng tổng này làm metric chất lượng mô hình.

## 10. Full regression test result

Lệnh chạy sau khi hoàn thiện diagnostic runner:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py --regression --frontend
```

Runner thực thi đúng `python -m pytest -q` từ root repo với biến môi trường DB/storage tạm, rồi `npm run build` trong `frontend`, rồi smoke tests.

Kết quả thực tế lần cuối:

```text
95 passed, 1 warning in 4.65s
vite v8.0.16: 22 modules transformed; build passed
77 smoke assertions PASS
exit code: 0
```

Warning: FAISS 1.9.0 dùng `numpy.core._multiarray_umath` đã deprecated ở NumPy 2.2.1. Không có test cũ bị hỏng, bỏ qua, xóa hoặc sửa để vượt qua test. Không phát hiện failure cần phân loại theo failure policy.

Môi trường xác minh:

| Component | Actual version/state |
| --- | --- |
| Python | 3.11.9 |
| PyTorch | 2.5.1+cpu |
| torchvision | 0.20.1+cpu |
| FAISS | module 1.9.0; distribution 1.9.0.post1 |
| CUDA | unavailable; torch CUDA build null |
| Node / npm | 22.19.0 / 10.9.3 |
| React / ReactDOM | 19.2.7 / 19.2.7 |
| Vite / TypeScript | 8.0.16 / 6.0.3 |

## 11. Hypothetical / synthetic cases tested

Fixture là ảnh checker RGB 128×96; memory bank tạm chứa embedding của chính fixture. Review/bad/pixel thresholds được chọn để kiểm tra ba nhánh QC, không được xem là calibrated production thresholds.

Các trường hợp synthetic/error gồm ảnh không decode được, MIME sai, missing field, fabric không tồn tại, reference không ACTIVE, metadata JSON hỏng, index thiếu và embedding dimension sai. Artifact thật được dùng cho load/self-query, không dùng lại locked test để đánh giá accuracy hoặc cải thiện model.

## 12. Known limitations

- Model/index smoke chạy trên CPU; không có CUDA/GPU/industrial throughput benchmark.
- TestClient xác minh API in-process; không phải browser E2E hay camera/PLC integration.
- Không có frontend test runner; đã xác minh build, chưa xác minh tương tác bằng browser automation.
- Khả năng load artifact không chứng minh production quality. V5 là development/postmortem evidence, không phải fresh locked test; A/B vẫn INACTIVE.
- 33/40 ảnh reference v4/v5 chỉ có 17/14 giá trị source_original khác nhau; không suy diễn chúng là 33/40 nguồn độc lập.
- README/model-info drift, pair-level source leakage risk, exception handling cho native FAISS/model loading và metadata identity checking vẫn là vấn đề baseline; Phase 0 không sửa chúng.
- Freeze là checksum/dependency record; không đóng gói dữ liệu, checkpoint hoặc private environment. Absolute Windows paths có giới hạn portability.

## 13. Risks introduced

Diagnostic runner đọc các artifact hiện có và sẽ thất bại nếu môi trường/cache/reference không còn tương thích; failure này là tín hiệu audit. Các lần chạy build ghi `frontend/dist` và cache build bị Git ignore. Tests tạo fixture tạm và cache kiểm thử thông thường.

Không có thay đổi import/call graph của production. Các thư mục temp do runner tạo được dọn sau khi đóng client/DB. Hai lần chạy pytest ban đầu dùng temp storage riêng ngoài repo; không ghi vào DB thực tế.

Rollback: bỏ riêng bốn file mới của Phase 0 nếu cần; không rollback checkpoint, index hay database vì chúng không thay đổi. Không dùng reset/clean toàn repo vì có user work chưa commit.

## 14. Manual verification steps for me

Chạy tại root repo:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py --regression --frontend
```

Mong đợi 95 pytest PASS, frontend build PASS, toàn bộ smoke PASS và exit code 0. Chỉ chạy smoke nếu cần:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py
```

Kiểm tra artifact checksum theo manifest (bao gồm `.env` hash, không in nội dung private):

```powershell
$baseline = Get-Content docs/implementation/phase_0_baseline_manifest.json -Raw -Encoding UTF8 | ConvertFrom-Json
$baseline.artifacts | ForEach-Object {
    $actual = (Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash
    [PSCustomObject]@{Path=$_.path;Matches=($actual -eq $_.sha256)}
}
```

Mong đợi mọi `Matches=True`. Các path cache tuyệt đối là của máy đã audit, cần có đúng cache để tái lập trên máy này.

Kiểm tra UI bằng hai terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

```powershell
Set-Location frontend
npm run dev
```

Mở URL Vite được in ra. Mong đợi health ONLINE, Fabric A/B INACTIVE và không chọn được để scan. Không activate reference để làm bước kiểm tra này. Kiểm tra API tại `/docs`; muốn xem ba nhánh inspect thành công hãy dùng sandbox runner, vì production registry hiện không có ACTIVE reference.

## 15. Git diff summary

`git diff --stat` cho tracked files vẫn chỉ hiện thay đổi người dùng có sẵn:

```text
tempCodeRunnerFile.exe | Bin 44395 -> 44626 bytes
1 file changed, 0 insertions(+), 0 deletions(-)
```

Bốn file Phase 0 mới chưa được stage, nên không hiện trong plain `git diff --stat`. Xem chúng bằng `git status --short` và `git diff --no-index --stat -- NUL <new-file>`; exit 1 của diff no-index là bình thường khi có nội dung mới. Không stage hoặc commit tự động.

## 16. Recommendation

SAFE TO APPROVE — đối với Phase 0 baseline audit và diagnostic scaffolding. Kết luận này không phải phê duyệt model/reference production.

Chờ user gửi rõ `APPROVE PHASE 0 — CONTINUE TO PHASE 1` trước khi viết Phase 1.

```text
============================================================
PHASE 0 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
```
