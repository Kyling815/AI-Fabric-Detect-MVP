# Phase 17 — TensorRT Full-Architecture Research Benchmark

Ngày: 2026-10-15, Asia/Bangkok. Trạng thái: chờ người dùng review. Hoàn thành phần mềm và benchmark offline trong phạm vi research; **không phải production model/release approval**. FP32 được chọn, FP16 bị loại. Phase 18 chưa bắt đầu.

## 1. Goal

Tiếp tục sau [setup và 5 smoke checks](phase_17_environment_setup.md): xác định candidate/workload cụ thể, build/reload/chạy TensorRT cho kiến trúc đầy đủ, kiểm tra parity, đo P50/P95/throughput và VRAM, rồi dừng review. Không thay CPU `.venv`, driver, API, registry hoặc artifact hiện hữu.

## 2. What I inspected before coding

Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; worktree đã dirty từ các phase trước/người dùng. Chụp SHA-256 của 373 file hiện hữu, đọc Phase 16/17 và yêu cầu gốc Phase 17–18; không có AGENTS.md áp dụng trong các đường dẫn đã kiểm tra. Không stage/commit/reset.

Backend vẫn FastAPI, frontend React/TypeScript/Vite, runtime anomaly vẫn CNN feature extractor + projection adapter + FAISS + rules. ResNet18UNet độc lập dùng decoder chuẩn `(128,64,32,16,16)`. Cấu hình dự án chọn ảnh 256×256; API hiện tại xử lý ảnh đơn nên batch1 là workload liên quan trực tiếp.

**Không tìm thấy supervised checkpoint production hợp lệ.** Các `.pt` hiện hữu trong storage là projection/adapter, không được diễn giải thành segmentation checkpoint. Chọn full architecture research probe, không tạo lịch sử train/epoch/loss giả. Encoder pretrained từ cache đã xác minh SHA `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; decoder/head ngẫu nhiên seed17, không train. Không đọc historical locked test hoặc dữ liệu ảnh thật cho phép đo này.

Tham chiếu [TensorRT10 Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html), [precision control/inspector](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/advanced.html) và [PyTorch2.8 CUDA synchronization/memory](https://docs.pytorch.org/docs/2.8/notes/cuda.html). Dùng API thực của bản đã cài; không đổi dependency để chạy mẫu tài liệu mới hơn.

## 3. Implementation performed

- CLI CPU chuẩn bị **ResNet18 + U-Net đầy đủ 12.460.898 tham số**, taxonomy background/defect, input `1×3×256×256` RGB float32 `[0,1]`. ImageNet normalization nằm trong graph; output gồm logits, probabilities, mask int64 và confidence.
- Export ONNX opset17, TorchScript reference và 7 probe/reference tensor. Hash pin manifest và cả ba payload; file/graph self-contained, fixed shape, chuẩn ONNX, không custom/external tensor data. TorchScript chỉ cho artifact local được xác nhận tin cậy; hash không biến file không rõ nguồn thành an toàn.
- TorchScript chuyển mô hình từ stack CPU sang GPU mà không cần cài torchvision/Pydantic/FAISS/backend vào GPU venv. Không thay module model/trainer/ONNX cũ.
- CLI GPU kiểm tra đúng isolated `.venv-tensorrt`, provenance research, CPU version bridge, cờ TF32 trong từng convolution đã trace và parity GPU PyTorch trước khi so TensorRT. Không CPU fallback nếu GPU lỗi.
- TensorRT FP32 strongly typed, TF32 tắt; FP16 mixed dùng FP16 builder flag cùng requested FP32 constraints cho normalization và output subgraph. **Requested constraints không tự chứng minh arithmetic FP32:** kiểm tra engine inspector và output thực. FP16 bị loại nếu sai semantics/numerics, dù nhanh hơn.
- Hai engine riêng, workspace128MiB, optimization level2; ghi file mới, đọc lại với SHA pin, deserialize và thực thi CUDA thật. Không load đồng thời source PyTorch và engine hoặc hai precision engine. Workspace limit không phải giới hạn tổng VRAM.
- Tiêu chí đặt trước: FP32 `atol=rtol=1e-4`; FP16 mixed `atol=rtol=1e-2`; **mask phải bằng tuyệt đối**, tổng xác suất sai không quá1e-5. Không nới tolerance khi kết quả fail. Semantic failure được ghi thành candidate rejection; shape/dtype/nonfinite lỗi phải dừng.
- Warmup10; mỗi timing mode GPU có3 block×100 mẫu, đồng bộ CUDA trước/sau từng request. Ghi đủ raw samples và từng block. Throughput là số ảnh/giây của serial single-request, không phải saturated concurrency.
- Tách GPU-resident I/O khỏi CPU-tensor roundtrip H2D→full model→D2H cả4 output. Không đo decode/EXIF/resize, FAISS, fusion, QC, HTTP, persistence hoặc full hybrid factory pipeline.
- Resource pass độc lập50 lượt, requested polling10ms, dùng device-wide `cudaMemGetInfo` qua Torch; thêm Torch allocated/reserved peak và TensorRT context size. **Device-wide high-water lấy mẫu không phải exact physical peak**, bao gồm context/libraries/ứng dụng khác; native TRT không nằm đầy đủ trong Torch allocator counters.

Candidate cuối: `storage/models/segmentation_candidates/phase17_full_architecture_probe_20261015_v3`.

Manifest SHA: `40155972b1021fc13f9e8cd892b3b90194c405204aee65b1cd1c4d9359e42462`.

ONNX SHA: `51decba9681af3d43925db55d3cb322ce8a3813d6a93e466d7eddd0b44dbce5b`.

Run cuối: `storage/models/segmentation_candidates/phase17_trt_benchmark_20261015_v5`; CPU preparation và GPU run cuối được chạy tuần tự sau khi hồi quy đã kết thúc, không chạy đồng thời với agent tests. Không đảm bảo loại bỏ hoạt động hệ thống/người dùng khác. Phép đo GPU kết thúc lúc15:56:29 +07:00. Raw/provenance/results được lưu trong [báo cáo JSON](phase_17_benchmark_report.json).

Ba vấn đề đã điều tra, không che giấu các run cũ:

1. V1 FP16 vi phạm probability contract và abort. Parser có thể phân rã Softmax nên constraints theo LayerType không bao phủ mọi op; bổ sung requested constraints theo tên subgraph. Cấu hình FP16 cuối vẫn sai parity và bị loại, không tuyên bố đã sửa được FP16.
2. Source TorchScript ban đầu trace `aten::_convolution` với `allow_tf32=True`; tắt global flag ở process GPU không ghi đè constant này. GPU baseline ban đầu có34 mask disagreement/7 probe. Sửa trace tắt TF32 ngay từ CPU, restore flags sau đó và kiểm tra từng convolution; candidate mới đạt CPU/GPU parity. ONNX source giữ nguyên hash. Không đổi weights/driver/TLS/DLL.
3. Unit test dùng no-op có thể đo0 do độ phân giải timer Windows. Sửa fixture dùng clock xác định **chỉ trong unit test**, không sửa số liệu benchmark. Suite cuối pass.

Candidate v1/v2 và GPU run v1–v4 là lịch sử/superseded, được giữ nguyên, không dùng làm số liệu chính. Run v1 có `failure_report.json`. Không xóa hay ghi đè các artifact đó.

## 4. Files created

- `backend/app/core/tensorrt_benchmark.py`
- `backend/app/ai/tensorrt_benchmark.py`
- `scripts/prepare_phase17_candidate.py`
- `scripts/benchmark_phase17_tensorrt.py`
- `tests/test_tensorrt_benchmark_contracts.py`
- `tests/test_tensorrt_benchmark_integration.py`
- `docs/implementation/phase_17_tensorrt_optimization.md`
- `docs/implementation/phase_17_benchmark_report.json`

Ngoài Git: các thư mục version mới dưới `storage/models/segmentation_candidates`, đã nằm trong ignore hiện hữu. Có ONNX, TorchScript, probe tensors, manifest, engine, inspector và raw benchmark results. Không đưa vendor/model binaries vào Git.

Tổng46 file candidate/run mới,920.483.228 bytes (~0,92GB), gồm các lần thử lịch sử được giữ lại. Không xóa dữ liệu hoặc artifact hiện hữu.

## 5. Files modified

**0 file hiện hữu trước bước này bị sửa/xóa.** 373 file baseline giữ nguyên SHA, gồm `.gitignore`, `Revision.md`, báo cáo setup và source/test các phase trước. Chỉ sửa các file mới của bước này trong quá trình triển khai.

Không cài/nâng/hạ dependencies trong CPU hoặc GPU venv. CPU freeze trước/sau khớp hoàn toàn; `pip check` cả hai PASS. Driver vẫn610.62. Không thay requirements, `.env`, registry, database schema hoặc frontend.

## 6. Architecture before

Pipeline anomaly CPU và supervised ONNX offline đã có; GPU venv chỉ xác nhận CUDA/TensorRT bằng mạng Conv+ReLU nhỏ. Chưa có benchmark full segmentation architecture hoặc native engine đầy đủ.

## 7. Architecture after

Production/API vẫn nguyên. Đường offline opt-in mới: full source model trên CPU → pinned research ONNX/TorchScript/probes → isolated GPU parity → TensorRT FP32/FP16 build/reload → timing/resource reports → lựa chọn cấu hình research theo parity trước, P95 sau. Không registry activation hoặc production runtime switching.

## 8. Backward compatibility

55 protected artifact khớp SHA,0 mismatch. 311 storage file gốc giữ fingerprint; các candidate mới được kiểm kê riêng, không giả vờ toàn storage không tăng. Data643 file, user files10 file, cache checkpoints3 file giữ fingerprint như Phase0/16. Hai fabric vẫn INACTIVE; API phản hồi409 tương ứng. Không thay behavior HTTP, thresholds, FAISS indexes hoặc reference activation.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| New unit/CPU integration | Bounds, hash/overwrite/path/trust guards, real full-model export/reload, trace TF32, rejected-selection policy | 66 PASS |
| CPU source→ONNX/TorchScript | 7 synthetic probes,4 outputs, exact masks | PASS |
| Torch2.5 CPU→Torch2.8 CPU bridge | Same pinned TorchScript/probes | PASS |
| PyTorch CUDA FP32 | Full graph vs CPU source reference,7 probes | PASS |
| TensorRT FP32 | Build, pinned reload, actual GPU inference,7 probes | PASS |
| TensorRT FP16 mixed | Build/reload/infer and inspect real FP16 formats | Execution PASS; parity REJECTED |
| Protection/environment checks | Old SHA/freeze/driver and both pip checks | PASS |

66 tests passed in6.03s,0 warning. Unit test clock fixtures are not GPU performance evidence; actual native measurements are recorded separately. No mocked TensorRT measurements. Compileall và git diff --check PASS.

TensorRT parser có cảnh báo kiểm tra binding Int64 của output mask; runner đã xác minh dtype thực `int64` trước inference. Không bỏ qua sai dtype hoặc coi cảnh báo là bằng chứng parity.

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

**1807 passed,1 warning in157.51s**, exit0. Tăng66 so với1741 sau setup, hoặc83 so với1724 cuốiPhase16. Frontend22 modules build PASS;77 smoke assertions PASS. DB/storage tests cô lập. Warning FAISS NumPy private namespace đã có từ trước. Không sửa hoặc làm hỏng test cũ. Hai lỗi phát triển nêu trên đã được phân loại/sửa; FP16 numerical rejection không bị biến thành test PASS cho độ đúng model.

## 11. Hypothetical / synthetic cases tested

Chỉ7 probe tổng hợp: black, white, uniform, checker, scratch, color_channels và seeded_noise. Kiến trúc đầy đủ nhưng decoder/head chưa train; các kết quả sau **không phải accuracy/recall/FPR, fresh locked test hoặc factory SLA**.

FP32 đạt7/7 case,0/458.752 mask pixel khác reference. Max absolute error: logits8.4638596e-6; probabilities2.9206276e-6. FP16 mixed có178 descriptor FP16 trong inspector, nhưng0/7 case đạt parity,155 mask pixel khác reference, logits error tối đa0.02184391; tổng xác suất có sai lệch2.44140625e-4. 155 pixel này là numerical disagreement, không được gọi là false alarms hay missed defects.

Timing với CPU tensors H2D→model→D2H (CPU baselines không có H2D/D2H):

| Backend | P50 ms | P95 ms | Serial images/s | Parity |
|---|---:|---:|---:|---|
| PyTorch2.5 CPU eager FP32,2 threads | 66.3163 | 70.7745 | 15.00 | PASS |
| ONNX Runtime CPU FP32,2 threads | 41.2214 | 44.4237 | 24.03 | PASS |
| PyTorch2.8 CUDA TorchScript FP32 | 6.8858 | 7.0956 | 144.92 | PASS |
| TensorRT FP32 | 3.5678 | 4.1092 | 276.25 | PASS, selected research configuration |
| TensorRT FP16 mixed | 2.4994 | 3.1624 | 386.23 | REJECTED |

GPU-resident I/O, synchronized host wall, không D2H:

| Backend | P50 ms | P95 ms | Serial images/s |
|---|---:|---:|---:|
| PyTorch CUDA FP32 | 6.0761 | 6.4206 | 162.74 |
| TensorRT FP32 | 2.4663 | 6.4402 | 344.64 |
| TensorRT FP16 mixed, rejected | 1.6700 | 2.1683 | 573.16 |

Các mode là cửa sổ đo riêng, không lấy P95 roundtrip trừ P95 resident làm transfer cost. FP32 resident block đầu có P95=6.5540ms, hai block sau khoảng2.5022/2.5046ms; giữ đủ3 block trong tổng số liệu. Đây là tail jitter quan sát được, chưa chứng minh nguyên nhân do clocks/thermal hay đạt trạng thái ổn định.

VRAM theo MiB=1.048.576 bytes:

| Engine | Sampled device used high-water khi build | Khi inference | TRT context allocation |
|---|---:|---:|---:|
| FP32 | 1341.8 MiB | 921.8 MiB | 13.188 MiB |
| FP16 mixed | 889.8 MiB | 869.8 MiB | 10.250 MiB |

FP32 min sampled free khi inference khoảng3173.7MiB; không gặp OOM trong workload đã đo. Device-wide values bao gồm context/libraries/process khác, không phải engine-only allocation. Polling thực có gap lớn hơn10ms (chi tiết JSON), có thể bỏ lỡ transient peak. Không dùng Torch buffer peak2.5MiB làm tổng VRAM của TensorRT. Không thử batch2/4 hoặc concurrent anomaly residency; không suy chúng không chạy vừa chỉ từ giới hạn4GB.

## 12. Known limitations

- Chưa có trained production candidate hoặc real-data quality evidence. Công cụ hiện nhận pinned research bundle cụ thể, không tự nhận arbitrary checkpoint là production model.
- FP16 bị loại; mixed precision constraints/float32 bindings chưa đảm bảo semantics. Giữ FP32; muốn thử tiếp FP16 cần graph precision/cast boundaries rõ ràng và parity riêng, không tự nới mask rule.
- Workload256×256/batch1 chưa đại diện mọi defect size, image resolution hoặc line-scan window. Không benchmark đầy đủ anomaly/FAISS/fusion/QC hay việc giữ hai model trên GPU.
- VRAM là sampled high-water cộng component counters, không exact device/process peak. Bộ nhớ hệ thống/driver/context và ứng dụng khác ảnh hưởng phép đo.
- 10 warmup/300 mẫu mỗi mode là benchmark ban đầu, chưa long-running soak/thermal/concurrency test. Clocks/power không khóa; số liệu có jitter, không phải SLA. PyTorch GPU baseline dùng TorchScript còn CPU baseline dùng eager; runtime/buffer-reuse khác nhau đã ghi rõ.
- Candidate thật sau training phải được export/parity/benchmark lại với ảnh/tensor đại diện, không kế thừa approval của probe này. Không tái sử dụng historical postmortem locked set như fresh test.

## 13. Risks introduced

Candidate/engine versions mới tốn dung lượng và phụ thuộc GPU/driver/TensorRT; không giả định portability của engine. TorchScript/ONNX/engine phải từ nguồn tin cậy. FP16 hiện có output drift và không được dùng làm production default. GPU venv riêng bảo vệ CPU runtime nhưng không có nghĩa full backend đã migrate GPU. Không có registry activation hoặc tải weights mới.

## 14. Manual verification steps for me

Kiểm tra báo cáo và kiểm thử phần mềm:

```powershell
Get-Content -Raw -Encoding UTF8 docs/implementation/phase_17_benchmark_report.json | ConvertFrom-Json | Select-Object status,recommendation,fp16_accepted,phase18_started
.\.venv\Scripts\python.exe -m pytest tests/test_tensorrt_benchmark_contracts.py tests/test_tensorrt_benchmark_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
.\.venv\Scripts\python.exe -m pip check
.\.venv-tensorrt\Scripts\python.exe -m pip check
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
```

Chạy lại benchmark trên candidate cuối; output version phải CHƯA tồn tại. Chạy tuần tự, không đồng thời với test/training/API tải nặng. Có thể thay tên output version mới, không xóa/ghi đè version cũ:

```powershell
.\.venv-tensorrt\Scripts\python.exe scripts/benchmark_phase17_tensorrt.py --candidate-dir storage/models/segmentation_candidates/phase17_full_architecture_probe_20261015_v3 --manifest-sha256 40155972b1021fc13f9e8cd892b3b90194c405204aee65b1cd1c4d9359e42462 --output-dir storage/models/segmentation_candidates/phase17_manual_review_v1 --trusted-local --allow-untrained-probe
```

Mong đợi FP32 parity/precision PASS, selected research configuration=FP32; FP16 được đo nhưng không được chọn nếu parity fail. Timing/engine hash và số pixel FP16 mismatch có thể đổi theo tactics/clocks; không đòi giống số đã ghi. Nếu FP32 hoặc GPU baseline fail, CLI phải dừng, không chuyển CPU/TensorRT giả hoặc tự đổi driver.

Tái tạo probe trong **một output version mới** nếu cần:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_phase17_candidate.py --research-probe --output-dir storage/models/segmentation_candidates/phase17_manual_source_v1
```

Dùng manifest SHA do lệnh trả về, không tái sử dụng SHA của v3 cho source mới. Rollback vận hành: ngừng gọi các CLI offline; runtime cũ chưa đổi nên không cần rollback model/registry/driver/venv. Nếu muốn dọn artifact hoặc source mới, archive đúng các version/file đã liệt kê và xác minh target trước; phase này không tự xóa thư mục hoặc dữ liệu người dùng.

## 15. Git diff summary

Riêng bước hoàn thiệnPhase17: **8 file mới,0 file hiện hữu sửa/xóa**,373 file trước bước này giữ nguyên SHA. Binaries/results dưới storage được ignore. `git diff --stat` tracked tích lũy vẫn **10 files changed,511 insertions(+),132 deletions(-)**, gồm thay đổi phase trước/người dùng; không phải delta của bước này. Không stage/commit, không sửa binary người dùng.

## 16. Recommendation

**SAFE TO APPROVE — chỉ phần mềm benchmark offline và TensorRT FP32 trên full research architecture.** FP16 REJECTED, không phải cấu hình dự phòng được tự kích hoạt. Production quality/release vẫn BLOCKED BY DATA/TRAINED CANDIDATE; không có phê duyệt triển khai. Chờ người dùng review trước khi bắt đầu Phase18, không tự chuyển phase vì benchmark đã chạy xong.

## 17. Tiến độ so với yêu cầu

| Yêu cầu bước này | Trạng thái |
|---|---|
| Candidate/workload cụ thể | DONE: full12.46M-parameter architecture, research/untrained,256×256/batch1,hash/provenance |
| TensorRT model đầy đủ | DONE: FP32 và FP16 mixed build/reload/infer thật |
| Parity | DONE: CPU bridge/PyTorch GPU/FP32 đạt; FP16 bị loại có bằng chứng |
| P50/P95/throughput | DONE: actual raw samples, tách resident và CPU-tensor roundtrip |
| VRAM | DONE: sampled device-wide high-water, Torch peaks và TRT context; không tuyên bố exact peak |
| Regression và báo cáo kết thúc | DONE:1807 tests/frontend77 smoke, chờ review |
| Batch2/4 hoặc hybrid concurrent residency | CHƯA THỬ; không nằm trong single-image benchmark đã chốt |
| Trained model/real masks/provenance/fresh acceptance | CHƯA CÓ bằng chứng production mới; vẫn cần dữ liệu và train/evaluate hợp lệ |
| Phase18 acceptance/release governance | CHƯA BẮT ĐẦU; chỉ thực hiện sau explicit approval |

Đây là hoàn thành engineering/research scope của Phase17, không phải17/18 mức độ sẵn sàng sản xuất. Hướng chuẩn bị Phase18: giữ FP32 làm technical baseline; chuẩn bị trained candidate, tiêu chí acceptance configurable và fresh locked test độc lập; không tự đặt customer SLA, không tự chuyển registry ACTIVE. Không viết code Phase18 trong bước này.

============================================================
PHASE 17 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
