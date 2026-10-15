# Phase 16 — ONNX Export

## 1. Goal

Export riêng supervised CNN sang ONNX khi hành vi PyTorch ổn định ở mức phần mềm; kiểm tra export, reload, inference, numerical parity, fixed shapes và error handling. Không chuyển đổi CNN/FAISS anomaly, không triển khai Phase 17/18. Theo yêu cầu mới của người dùng, báo cáo thêm section 17 “Tiến độ so với yêu cầu”.

## 2. What I inspected before coding

Đọc yêu cầu Phase 16 trong lộ trình gốc; kiểm tra ResNet18UNet, normalization/predict, checkpoint save/load và provenance, shadow worker, fixture Phase 8/9, audit dữ liệu Phase 7/8, báo cáo Phase 15 và script hồi quy an toàn. Hành vi PyTorch đã có kiểm thử forward/backward, overfit fixture, deterministic prediction và 1617 test hồi quy trước Phase 16; đây không phải xác nhận model production đã tốt.

Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; working tree có thay đổi từ các phase trước và binary người dùng. Không có AGENTS.md áp dụng trong repo/các thư mục cha đã kiểm tra. Chụp SHA của 357 file hiện hữu trước khi sửa. Không stage/commit/reset.

Môi trường Python 3.11.9, PyTorch 2.5.1+cpu, NumPy 2.2.1, không có CUDA. Trước phase này chưa cài ONNX/ORT; storage chỉ có adapter/projection checkpoints, không có checkpoint segmentation production đã xác nhận. Kiểm tra signature exporter ngay trong Torch được cài; dùng `dynamo=False`, opset 17, không nâng Torch. Tham khảo [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html), [ONNX checker](https://onnx.ai/onnx/api/checker.html) và [tài liệu các lựa chọn exporter PyTorch](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html?highlight=transformer). Tài liệu online hiện mới hơn stack pinned; kết quả chạy thực trên bản cài đặt là bằng chứng tương thích của phase này.

## 3. Implementation performed

- Contracts `OnnxExportConfig`/`OnnxManifest`: NCHW cố định, RGB float32 [0,1], opset 17, taxonomy theo đúng thứ tự, source/version/hash/synthetic flag và `production_approved=false`. Default tolerance `atol=1e-4`, `rtol=1e-4`; không tự nới tolerance khi lỗi.
- Export wrapper gọi đúng model PyTorch hiện có; xuất logits, probabilities, mask int64 và confidence. ImageNet normalization nằm trong graph; decoding/EXIF/resize ngoài graph và thuộc caller.
- Chỉ nạp checkpoint local được xác nhận tin cậy, pin SHA-256 và model version; hash và deserialize cùng một stream. Synthetic checkpoint bị chặn nếu không opt-in rõ ràng. Không coi `weights_only` hoặc tự tính hash là sandbox/xác thực nguồn tin cậy.
- Thực thi ONNX checker và ONNX Runtime với `CPUExecutionProvider` được chọn rõ ràng; kiểm tra graph/opset/input/output, không dùng external tensor initializers. Chạy 7 probe xác định cho MỖI export; float outputs phải allclose, mask phải bằng nhau tuyệt đối. Argmax flip vẫn là parity failure dù sai số float nhỏ.
- Chỉ publish bundle mới sau khi parity pass: `model.onnx`, `manifest.json`, `parity_report.json`; exclusive directory/files, không overwrite. Dùng lại guard Phase 8 bảo vệ Data/source/cache/production storage; trong storage chỉ cho phép new version dưới `storage/models/segmentation_candidates`.
- Runner reload yêu cầu cả model hash và manifest hash do caller tin cậy cung cấp, ràng buộc cả taxonomy/shape với graph; checker và runtime dùng đúng bytes đã xác minh. Host guard chặn sai shape/dtype/range/NaN/Inf trước khi gọi ORT, không tự resize/coerce input.
- CLI có chế độ export checkpoint thật và `--demo` cô lập. Demo thực sự export/reload/infer trên checkpoint SYNTHETIC có pretrained encoder + standard decoder chưa huấn luyện; tất cả model/checkpoint demo nằm trong thư mục tạm, được xóa khi kết thúc. Không train/download/activate.

## 4. Files created

- `requirements-onnx.txt`
- `backend/app/core/segmentation_onnx.py`
- `backend/app/ai/segmentation_onnx.py`
- `scripts/export_segmentation_onnx.py`
- `tests/segmentation_onnx_fixtures.py`
- `tests/test_segmentation_onnx_contracts.py`
- `tests/test_segmentation_onnx_export.py`
- `tests/test_segmentation_onnx_integration.py`
- `docs/implementation/phase_16_onnx_export.md`
- `docs/implementation/phase_16_onnx_report.json`

## 5. Files modified

Không sửa file hiện hữu nào; giữ nguyên SHA của toàn bộ 357 file trước phase. Không đổi `requirements.txt`, `.env`, API/frontend/config, model/anomaly, QC, registry hoặc DB.

Môi trường `.venv` được bổ sung đúng 7 package: onnx 1.17.0, onnxruntime 1.20.1, coloredlogs 15.0.1, flatbuffers 25.12.19, humanfriendly 10.0, protobuf 7.36.2, pyreadline3 3.5.6. So sánh pip freeze trước/sau: không package cũ nào bị gỡ/đổi phiên bản; `pip check` PASS. Hai dependency chính nằm trong file optional riêng, không tự import ONNX/ORT ở production API.

## 6. Architecture before

Supervised ResNet18+U-Net độc lập chạy PyTorch offline/shadow. API vẫn dựa trên CNN dense features + FAISS và QC hiện có. Chưa có ONNX exporter/loader/parity contract.

## 7. Architecture after

Luồng offline mới: trusted pinned checkpoint → PyTorch supervised CNN → fixed-shape FP32 ONNX → checker → ORT CPU → 7-case parity → NEW bundle. Runner đọc bundle với model+manifest pins → input guard → CPU inference → output contract validation.

API, shadow subprocess, FAISS và QC không tự chuyển sang ONNX. Mỗi shape là một artifact riêng; thay shape phải export và kiểm tra parity lại. Không có dynamic axes, TensorRT, FP16/INT8, registry activation hoặc release gate Phase 18.

## 8. Backward compatibility

Golden HTTP và OpenAPI giữ nguyên; full smoke vẫn kiểm tra pretrained anomaly backbone/adapter/cache, FAISS, HTTP/SQLite, fail-safe, persistence và cleanup. Hai fabric vẫn INACTIVE và trả 409 đúng baseline.

55 artifact gốc khớp SHA. Fingerprint `storage` 311 file, `Data` 643 file, 10 file người dùng và cache trọng số 3 file không đổi so với Phase 15; cụ thể lưu trong JSON. Không chạm locked data, threshold, encoder anomaly hoặc memory geometry. Không giữ ONNX demo làm candidate production.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| 58 contract tests | Shape/tolerance/schema/taxonomy/input/output/parity guards | PASS |
| 40 export/runner tests | Export thật, reload, 3-class ordering, batch/odd shape, checksum/version, corrupt graph, no overwrite, protected output, cleanup | PASS |
| 9 integration tests | Golden API, demo CLI 3 exports/21 cases, CLI failure | PASS |
| Artifact/fingerprint checks | Giữ nguyên data/model/storage/user files | PASS |
| Compileall, git diff --check, pip check | Syntax/whitespace/dependencies | PASS |

Lệnh Phase 16:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_onnx_contracts.py tests/test_segmentation_onnx_export.py tests/test_segmentation_onnx_integration.py -q
```

Kết quả thực tế: **107 passed, 1 warning in 29.83s**, không skip. Không sửa/giảm độ chặt của test cũ. Nếu môi trường khác chưa cài optional dependencies thì một số test export/integration sẽ skip; phải cài prerequisites để xác minh Phase 16 đầy đủ như đợt này.

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Lần cuối với đủ 107 test mới: **1724 passed, 1 warning in 146.01s**, exit 0. Phase 15 có 1617 test; tăng đúng 107. Frontend TypeScript/Vite build PASS, 22 modules; smoke PASS **77 assertions**. DB/storage kiểm thử cô lập, không ghi vào registry/DB production. Không có test cũ bị hỏng.

Warning suite duy nhất là FAISS loader dùng NumPy private namespace, đã có từ trước. Export có TracerWarning được capture và ghi rõ vào parity report: Python boolean input checks trở thành hằng trong trace, KHÔNG phải guard trong ONNX graph; host runner đảm nhiệm các guard đó.

## 11. Hypothetical / synthetic cases tested

Demo đo parity thực sau hồi quy, ghi nhận lúc `2026-10-15 05:48:27 UTC`. Không phải đánh giá accuracy, latency hoặc fresh locked test. Chi tiết từng case/hash/taxonomy/version trong [báo cáo JSON](phase_16_onnx_report.json).

| Fixed input NCHW | Output logits/probabilities | Output mask/confidence | Cases |
|---|---|---|---:|
| 1×3×64×64 | 1×2×64×64 | 1×64×64 | 7 PASS |
| 1×3×35×49 | 1×2×35×49 | 1×35×49 | 7 PASS |
| 2×3×32×48 | 2×2×32×48 | 2×32×48 | 7 PASS |

Probe: black, white, uniform, checker, scratch, RGB channel differences và seeded noise; batch=2 có noise khác nhau theo item. Tổng 3 export/21 case, reload và inference thực đều pass. Test riêng còn kiểm tra taxonomy 3 lớp và prediction tương đương checkpoint PyTorch reload.

Sai số tuyệt đối lớn nhất trên 21 case: logits **3.9935112e-6**, probabilities **1.3709068e-6**, confidence **1.3709068e-6**; đều đạt `abs(actual-expected) <= 1e-4 + 1e-4 * abs(expected)`. Mask có **0 pixel sai khác**, yêu cầu equality tuyệt đối. Mỗi ONNX demo khoảng 49.86 MB; không giữ binary vào repo/storage.

Checkpoint demo dùng encoder pretrained từ cache đã xác minh và decoder ngẫu nhiên standard, không train. Epoch/loss trong metadata fixture chỉ là giá trị phục vụ contract, không phải kết quả training/validation. Không dùng confidence chưa calibrated làm QC confidence.

## 12. Known limitations

- Chỉ xác minh CPU FP32 trên stack pinned hiện tại; chưa GPU, FP16, TensorRT, VRAM hoặc performance optimization.
- Chưa có supervised checkpoint production đã được huấn luyện/đánh giá hợp lệ. Parity fixture không chứng minh khả năng phát hiện lỗi, giảm FPR hoặc generalization. Candidate thật phải export và chạy parity riêng.
- Các shape được kiểm thử là artifact RIÊNG, không phải một dynamic model. Không bảo đảm mọi shape/hardware ngoài case kiểm tra; exporter kiểm tra lại parity cho từng artifact được yêu cầu.
- Numerical parity trên 7 probe không phải chứng minh toán học cho mọi input. Cần thêm ảnh/tensor đại diện và kiểm tra gần ranh giới argmax của candidate thật; exporter fail khi mask khác, không tự bỏ qua.
- Raw ONNX graph không có Python input validation; bypass runner phải cung cấp guard tương đương. EXIF/resize/pipeline image-file end-to-end nằm ngoài export này.
- Chỉ trusted local checkpoints/graphs. Hash pin phát hiện thay đổi bytes, không chứng minh nguồn an toàn hay thay thế sandbox/approval/signature governance. Parity không tự cấp production approval.

## 13. Risks introduced

Optional native dependencies tăng kích thước môi trường và có thể khác hành vi theo phiên bản/provider. Legacy exporter đang dùng để giữ tương thích Torch 2.5.1, cần kiểm tra migration riêng nếu nâng stack. Export tốn RAM/disk và tạm thay đổi RNG/thread/backend flags rồi restore: chỉ chạy offline, tuần tự, không chạy đồng thời trong production server. Process chết giữa publish có thể để lại thư mục mới chưa hoàn chỉnh; không tái dùng/ghi đè thư mục đó. Strict exact-mask rule có thể từ chối candidate có argmax gần hòa dù float error nhỏ; đây là failure rõ ràng, không tự nới điều kiện.

## 14. Manual verification steps for me

Tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-onnx.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/export_segmentation_onnx.py --demo
.\.venv\Scripts\python.exe -m pytest tests/test_segmentation_onnx_contracts.py tests/test_segmentation_onnx_export.py tests/test_segmentation_onnx_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Mong đợi: demo `all_passed=true`, 3 fixed shapes, 21 parity cases, 0 mask mismatch, CPUExecutionProvider, synthetic=true, production_approved=false, không giữ model binary. Test riêng 107 passed, toàn bộ 1724 passed, frontend PASS và 77 smoke assertions. Sai số có thể dao động giữa môi trường, nhưng phải đạt tolerance, không bắt buộc giống số đã lưu.

Chỉ khi có checkpoint nguồn đã được xác nhận tin cậy, có provenance và đúng model version, dùng mẫu dưới đây; tự tính hash không làm file không rõ nguồn trở nên an toàn. Thay placeholder trước khi chạy, chọn một output version CHƯA tồn tại:

```powershell
$taskCheckpoint = 'C:\path\to\trusted_candidate\best_epoch_0001.pt'
$taskCheckpointSha = (Get-FileHash -LiteralPath $taskCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
.\.venv\Scripts\python.exe scripts/export_segmentation_onnx.py --checkpoint $taskCheckpoint --expected-sha256 $taskCheckpointSha --model-version 'EXACT_SOURCE_MODEL_VERSION' --trusted-local --output-dir 'storage/models/segmentation_candidates/onnx_export_v1' --shape 1 3 64 64
```

Checkpoint synthetic cần thêm `--allow-synthetic`, chỉ dùng nghiên cứu. Khi reload, cung cấp cả `model_sha256` và `manifest_sha256` từ kết quả export cho `OnnxSegmentationRunner(..., trusted_local=True)`; gọi `.predict()` với đúng ndarray contract. Không gọi export trong request/API, không tự activate registry.

Rollback: ngừng dùng CLI/runner ONNX là đủ để trở về runtime cũ vì API chưa thay đổi. Nếu hoàn tác source, backup/archive đúng 10 file mới trong section 4; không reset/checkout các file cũ. Optional packages chỉ được bổ sung, không nâng package cũ; chưa tự gỡ để tránh ảnh hưởng môi trường dùng chung. Không cần rollback data/model/registry/DB.

## 15. Git diff summary

Phase 16: **10 file mới, 0 file hiện hữu bị sửa**; 357 file hiện hữu giữ nguyên SHA. `git diff --stat` tracked tích lũy vẫn **9 files changed, 510 insertions(+), 132 deletions(-)**, không đổi so với Phase 15; bao gồm thay đổi trước đó và binary người dùng. Toàn bộ Phase 16 là file mới untracked nên không xuất hiện trong stat mặc định. Không stage/commit. `.venv` có thêm 7 dependencies như section 5.

## 16. Recommendation

SAFE TO APPROVE — cho công cụ export/reload/parity ONNX CPU offline, bảo toàn pipeline hiện tại. Không phải phê duyệt quality, production deployment hoặc GPU/TensorRT. Muốn chuyển candidate thật sang production cần dữ liệu/model hợp lệ và acceptance riêng. Phase 17 chưa bắt đầu.

## 17. Tiến độ so với yêu cầu

Đã triển khai phần mềm đến **Phase 16 trên lộ trình Phase 1–18**, kèm Phase 0 audit. Phase 17 và 18 chưa bắt đầu. Không diễn giải 16/18 thành phần trăm sẵn sàng sản xuất: nhiều module hiện là nghiên cứu, observe-only hoặc mô phỏng.

Đối với Phase 16, đủ yêu cầu export, reload, inference CPU, numerical parity với tolerance xác định, fixed shapes và error handling đều có test chạy thực; pipeline anomaly FAISS không chuyển đổi. Điều kiện “PyTorch stable” đạt ở mức phần mềm/checkpoint behavior đã hồi quy, không đồng nghĩa đã có model sản xuất tốt.

| Nhóm yêu cầu gốc | Đã làm được | Chưa chứng minh/hoàn thành và phương hướng |
|---|---|---|
| Phase 1–4: nền tảng migration an toàn | Contracts, quality/pattern scaffolds, anomaly abstraction/experiments; giữ API | Quan sát/nghiên cứu chưa chứng minh cải thiện detection; giữ feature gating và đánh giá độc lập |
| Phase 5–6: lot/pattern awareness | Memory hierarchy, pattern pooling và ablation tools | Lợi ích trên real lot độc lập chưa xác nhận; cần dữ liệu nhiều lot/camera và evaluation chống leakage |
| Phase 7–8: supervised CNN | Pipeline dữ liệu/split/metrics, model/trainer/checkpoint độc lập | Chưa có production model hợp lệ; cần real pixel masks, roll/lot/camera provenance và training/validation thật |
| Phase 9–11: hybrid evidence/QC | Shadow inference, fusion và QC tách biệt, opt-in policies | Chưa có hybrid candidate được chấp nhận/kích hoạt production; phải kiểm tra từng branch và calibrate policy trên dữ liệu độc lập |
| Phase 12–14: machine vision | Physical mapping, simulated line-scan, seam grouping có contract/calibration | Chưa xác minh calibration/camera/encoder/PLC thực tế và full factory loop; cần pilot có đo đạc |
| Phase 15: profiling/scheduling | CPU stage profiler, bounded offline scheduler, measured CPU probes | Chưa GPU contention/VRAM/industrial throughput; cần hardware và workload thật |
| Phase 16: ONNX | Export/check/reload/infer/parity thực trên CPU, 3 fixed-shape demo artifacts | Chưa export/evaluate checkpoint production; chạy lại exporter trên candidate thật và tensor/ảnh đại diện |
| Phase 17: TensorRT/optimization | Chưa bắt đầu | Chỉ thực hiện khi có môi trường CUDA/TensorRT phù hợp và người dùng duyệt phase; không giả lập benchmark |
| Phase 18: acceptance/release governance | Chưa bắt đầu | Cần tiêu chí configurable, fresh locked test tách historical postmortem, acceptance report và explicit activation |

Khoảng cách quan trọng nhất với mục tiêu ban đầu vẫn là **bằng chứng trên dữ liệu thật**: chưa chứng minh giảm false positive Fabric A, tăng localization hoặc đạt SLA dây chuyền. Audit Phase 7/8 ghi nhận toàn bộ 120 Company masked rows là synthetic và thiếu provenance roll/lot/camera xác thực; Phase 16 không biến chúng thành real ground truth hay tái sử dụng historical locked set làm fresh test. Hướng đi ưu tiên: hoàn thiện dữ liệu/provenance → train/evaluate candidate thật → hardware profiling → acceptance độc lập. Đây là khuyến nghị, không phải bắt đầu phase tiếp theo.

============================================================
PHASE 16 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
