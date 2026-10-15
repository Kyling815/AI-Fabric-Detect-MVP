# Phase 7 — Supervised segmentation data pipeline

Ngày kiểm tra: 2026-10-14, Asia/Bangkok. Chỉ thực hiện Phase 7; chưa bắt đầu Phase 8.

## 1. Goal

Xây dựng nền tảng dữ liệu cho supervised segmentation: taxonomy, manifest, loader/validation image–mask, split chống leakage, augmentation đồng bộ, cấu hình huấn luyện và pixel metrics. Không huấn luyện model, không thay encoder anomaly, không đưa segmentation vào `/api/inspect`.

## 2. What I inspected before coding

- Branch `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; worktree đã có thay đổi từ các phase trước và file người dùng. Không có AGENTS.md áp dụng được trong repo/các thư mục cha đã kiểm tra.
- Tài liệu yêu cầu và báo cáo Phase 0–6; runtime FastAPI, inspection/QC contracts, encoder/FAISS, manifests Company, script adapter training và fixture regression hiện hữu.
- Baseline trước Phase 7: 564 tests. Registry production của hai fabric vẫn `INACTIVE`, không có active dense reference. Encoder đăng ký là CPU ResNet18 layer2+3 + adapter v4, 128 chiều, stride 8; không thay trạng thái này.
- Legacy adapter trainer có `random_split` trên pairs. Chỉ đọc; không sửa và không tái sử dụng cách split đó cho segmentation.
- DME/WFDD/Lusitano/MVTec tại các đường dẫn đang cấu hình không tồn tại; không coi chúng là dữ liệu sẵn sàng.
- Audit đọc metadata toàn bộ hai runtime CSV; chỉ decode bốn image–mask pairs thuộc `adapter_train` (hai/fabric). Không decode/evaluate ảnh locked holdout.

| Fabric | Tổng rows | Real | Synthetic | Rows có mask | Metadata capture thiếu |
| --- | ---: | ---: | ---: | ---: | --- |
| fabric_001 | 121 | 61 | 60 | 60, đều synthetic | roll_id, lot_id, camera_id |
| fabric_002 | 129 | 69 | 60 | 60, đều synthetic | roll_id, lot_id, camera_id |

Bốn spot-check có geometry 960×1280 và mask 0/255 hợp lệ khi mapping rõ sang `defect`. Không suy ra semantic ground truth tám lớp từ tên lỗi legacy. Folder/nhãn `lot_001` không phải chứng cứ provenance thu nhận đã xác thực. Những crop chung `source_original` phải được giữ chung nhóm.

## 3. Implementation performed

1. Contracts Pydantic frozen, extra fields bị từ chối: taxonomy subset có background ID 0, manifest version 1, SHA-256 image/mask, provenance thực/synthetic, source lineage, roll/lot/camera và split status. Defect bắt buộc có mask và declared foreground classes; Good không khai báo lỗi.
2. Taxonomy hỗ trợ `background`, `stripe`, `scratch`, `missing_fabric`, `wrong_color`, `wrinkle`, `spot`, `stain`, thêm lớp generic `defect`. Mặc định chỉ `background/defect`; IDs lấy theo thứ tự subset. Binary 0/255 bắt buộc mapping explicit, không tự đổi tên lỗi legacy.
3. Split bằng connected components: source/ancestor IDs và hashes, duplicate image hash, roll luôn được liên kết khi có metadata; lot và camera liên kết khi được bật. Roll/lot scoped theo fabric, camera ID global. Seed deterministic, không chia ngẫu nhiên các patch trong cùng nhóm. Thiếu metadata bắt buộc hoặc thiếu nhóm độc lập thì fail.
4. Giữ safe split pins; conflicting pins bị từ chối. Mẫu historical_locked giữ trạng thái cũ; các hậu duệ/nhóm liên quan bị quarantine `excluded`, không tái gắn thành fresh test. Explicit train/evaluation pin trong nhóm protected gây lỗi.
5. Loader kiểm tra checksum, image decode, geometry trước resize, integer class IDs, active labels và annotation consistency. Palette giữ indices; hỗ trợ raw/oriented mask coordinates theo EXIF; không chấp nhận EXIF mask độc lập. Good có thể có mask trống hoặc không có mask. All-ignore không được xem là annotation hợp lệ.
6. Paired flip/quarter-turn; brightness/contrast chỉ ảnh; resize image bilinear, mask nearest. Transform không sửa input. Dataset trả float32 CHW [0,1], int64 HW, sample_id; augmentation chỉ train, deterministic theo seed/epoch/sample ID. Validation/test không đổi theo epoch.
7. Training config là specification duy nhất: `enabled=false`, U-Net/ResNet18 dự kiến, AdamW, encoder/decoder LR, CE+Dice, early stopping, AMP-if-CUDA. Các model/loss/optimizer/training loop này chưa được triển khai. `share_anomaly_encoder=false` bắt buộc; synthetic mặc định bị từ chối.
8. Streaming confusion matrix, IoU/Dice/precision/recall từng lớp và mean foreground/present classes. Undefined metrics là null và loại khỏi macro mean; false-positive ở lớp vắng ground truth vẫn bị tính điểm 0, không biến Good-only thành accuracy foreground 100%.
9. CLI read-only preflight, schema, Company audit và temporary synthetic demo. Không có training endpoint/command, không tạo checkpoint/index mới.

## 4. Files created

12 files mới:

- `backend/app/core/segmentation_data.py`
- `backend/app/ai/segmentation_split.py`
- `backend/app/ai/segmentation_dataset.py`
- `backend/app/ai/segmentation_metrics.py`
- `config/segmentation_pipeline_example.json`
- `scripts/check_segmentation_pipeline.py`
- `tests/test_segmentation_data.py`
- `tests/test_segmentation_split.py`
- `tests/test_segmentation_metrics.py`
- `tests/test_segmentation_pipeline_integration.py`
- `docs/implementation/phase_7_data_report.json`
- `docs/implementation/phase_07_supervised_data_pipeline.md`

## 5. Files modified

Không sửa file có sẵn trong Phase 7. Fingerprint của 248 file trước khi triển khai không đổi. Không sửa Revision.md, research report, ảnh test, dữ liệu, database, registry, FAISS, adapter hoặc cached weights.

## 6. Architecture before

Runtime: image → inspection/QC pipeline Phase 1–6 → dense anomaly encoder/memory → QC decision/localization → response/persistence.

Company manifests phục vụ reference/anomaly/adapter và synthetic masks legacy; chưa có supervised manifest contract, source-disjoint segmentation loader hoặc pixel metric layer độc lập.

## 7. Architecture after

Runtime cũ giữ nguyên. Thêm nhánh offline độc lập:

`manifest + taxonomy → connected-component grouped split → verified image/mask loader → train-only paired transforms → Dataset/DataLoader`

`prediction IDs + target IDs → segmentation confusion matrix / IoU / Dice`

Training config là contract cho phase sau, không phải trainer. Hai nhánh không chia sẻ encoder; production không import/call Dataset này.

## 8. Backward compatibility

- Năm golden QC/API cases, GET endpoints và OpenAPI giữ nguyên; integration test cố tình khiến SegmentationDataset constructor ném lỗi nếu bị production gọi.
- Full baseline runner dùng DB/storage tạm cho regression/synthetic API checks. Cached artifacts chỉ đọc; registry A/B vẫn inactive. Không activation, không calibration hoặc thay threshold.
- 55 protected artifact hashes không đổi. Scope hashes không đổi: storage 311 files, Data 643 files, user_files 10 files, torch checkpoints 3 files.
- Không thêm/sửa dependency, không tải model/dataset, không stage/commit. Các synthetic fixtures chỉ tồn tại trong temporary directories.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| test_segmentation_data.py, 42 cases | Geometry, IDs, Good/defect masks, checksum, palette, EXIF, ignore, augmentation, strict config | PASS |
| test_segmentation_split.py, 21 cases | Source/roll/lot/camera disjoint, ancestor transitivity, duplicate content, locked descendants, pins, seed, fail-closed | PASS |
| test_segmentation_metrics.py, 9 cases | Known confusion matrix, absent classes, false positives, ignore pixels, invalid IDs | PASS |
| test_segmentation_pipeline_integration.py, 15 cases | DataLoader contracts, eval isolation, five golden API cases, CLI read-only/synthetic gates | PASS |
| Existing 564 regression tests | Không làm hỏng Phase 0–6/runtime hiện hữu | PASS |
| Frontend production build | TypeScript/Vite compatibility | PASS, 22 modules |
| Baseline artifact/API smoke | Real cached model/index checks và isolated synthetic API cases | PASS, 77 assertions |
| compileall / git diff --check | Syntax và whitespace tracked diff | PASS |
| Hash verification | 248 source files và 55 protected artifacts; bốn protected scope hashes | Không đổi |

## 10. Full regression test result

Command thực tế, chạy từ project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Kết quả cuối: **651 passed, 1 warning in 39.23s**, frontend build PASS, artifact/API smoke PASS (77 assertions), process exit 0. Có 87 tests mới so với baseline 564. Warning là FAISS dùng NumPy private namespace đã có sẵn; không phải lỗi segmentation mới.

## 11. Hypothetical / synthetic cases tested

- Tiny demo chín pairs/ba source groups, mỗi source ba crop; split 3/3/3 để giữ nguyên nhóm. Không đạt chính xác 70/15/15 vì chỉ ba nhóm độc lập; phát warning, không phá group boundaries.
- Batch demo image `[2,3,32,32]` float32, mask `[2,32,32]` int64. Fixture origin được ghi synthetic và cần opt-in.
- Mask mismatch, unsupported IDs, fractional/RGB masks, empty defect, foreground trên Good, checksum sai, missing capture fields, one-camera-only split, overlapping lots/rolls và locked ancestor bị chặn đúng kỳ vọng.
- Oracle demo so target với chính target để kiểm tra số học IoU/Dice. Điểm 1.0 này không phải inference, không chứng minh model accuracy; không có model đã train.

## 12. Known limitations

- Production training readiness: `UNVERIFIED_REAL_PIXEL_LABELS_AND_CAPTURE_GROUP_METADATA`. Chưa có real human-verified pixel labels/capture groups đủ để duyệt training; không coi 120 masked synthetic rows là dữ liệu thật.
- Audit pixel chỉ bốn adapter_train pairs, không phải validation toàn bộ mask files. Locked metadata không bị sửa và locked pixels không dùng đánh giá.
- Split chỉ đảm bảo disjoint theo provenance/hash được cung cấp; không tự phát hiện near-duplicates, lineage thất lạc hoặc roll/lot/camera khai sai. Root source hashes còn thiếu sẽ phát warning.
- Groups không thể chia nhỏ: tỷ lệ gần đúng, không đảm bảo class balance. Một camera/lot duy nhất có thể khiến yêu cầu ba partitions không khả thi.
- Dataset output chưa có ImageNet mean/std normalization; phase trainer phải quy định normalization rõ ràng. Không elastic/crop augmentation, không color-based auto-labeling.
- Chưa triển khai CNN segmentation model, loss, optimizer, training loop, overfit check, checkpoints, latency/real accuracy evaluation hoặc segmentation fusion. Đây không phải lỗi thiếu của Phase 7.
- Test môi trường CPU hiện hữu; chưa kiểm chứng CUDA hoặc distributed/multiworker persistent-loader training.

## 13. Risks introduced

- Taxonomy ordering quyết định ID: đổi subset cần version manifest mới, không tái diễn giải mask cũ.
- Resize có thể làm mất lỗi rất nhỏ; cần lựa chọn resolution/patching bằng validation thực ở phase training.
- Flip/rotation có thể không phù hợp hướng weave; augmentation mặc định OFF, chỉ bật khi xác minh domain.
- Content hashes được kiểm mỗi lần load làm tăng I/O; cần benchmark/caching có integrity policy riêng về sau.
- IDs/provenance caller-declared không thay thế audit thu nhận. Không bỏ require_roll_id chỉ để dataset hiện tại chạy được; override source-only không phải chứng cứ roll-disjoint.
- CE+Dice/lr/etc hiện là cấu hình dự kiến, không phải hyperparameters đã tối ưu hoặc bảo đảm convergence.

## 14. Manual verification steps for me

Chạy tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/check_segmentation_pipeline.py --schema
.\.venv\Scripts\python.exe scripts/check_segmentation_pipeline.py --audit-existing
.\.venv\Scripts\python.exe scripts/check_segmentation_pipeline.py --demo
.\.venv\Scripts\python.exe -m pytest -q tests/test_segmentation_data.py tests/test_segmentation_split.py tests/test_segmentation_metrics.py tests/test_segmentation_pipeline_integration.py
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

Kỳ vọng: demo báo `SYNTHETIC_SOFTWARE_CHECK_ONLY`, trained_model=false, production_accuracy_evaluated=false; audit báo readiness chưa xác minh. Các lệnh không train hoặc export artifacts.

Với manifest riêng đã được người dùng chuẩn bị và kiểm chứng, chỉ preflight (đổi đường dẫn theo file thực tế):

```powershell
.\.venv\Scripts\python.exe scripts/check_segmentation_pipeline.py --manifest "C:\path\to\manifest.json" --config config/segmentation_pipeline_example.json
```

Không thêm `--allow-synthetic` cho dữ liệu production; flag chỉ dành cho fixture/experiment synthetic được khai báo rõ. Bật training.enabled trong JSON cũng không tạo trainer trong phase này.

Rollback: production không dùng nhánh mới nên không cần restore registry/model. Có thể chuyển chính 12 file mới liệt kê ở mục 4 sang một thư mục backup riêng sau review. Không dùng reset/clean toàn repo, không động vào thay đổi Phase 0–6 hoặc file người dùng. Chưa thực hiện rollback/xóa file nào.

## 15. Git diff summary

Phase 7 riêng: **12 files mới, 0 files có sẵn bị sửa**; các file mới vẫn untracked, chưa stage/commit.

`git diff --stat` chỉ phản ánh cumulative tracked changes có trước phase: 5 files, 175 insertions, 53 deletions; bao gồm `.env.example`, core/config.py, services/fabric_registry.py, services/inspection_service.py và binary `tempCodeRunnerFile.exe` của người dùng. Không quy các thay đổi đó cho Phase 7. Staged diff rỗng.

## 16. Recommendation

**SAFE TO APPROVE** cho nền tảng dữ liệu và backward compatibility của Phase 7. Không phải approval training/model quality hoặc production activation. Trước training supervised thực cần pixel labels thật đã xác minh, roll/lot/camera/source provenance, class coverage và fresh untouched holdout. Chờ người dùng review trước khi bắt đầu Phase 8.

============================================================
PHASE 7 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
