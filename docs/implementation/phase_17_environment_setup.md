# Phase 17 — Isolated GPU/TensorRT Setup and Smoke Test

## 1. Goal

Theo yêu cầu bổ sung của người dùng: chỉ thiết lập môi trường GPU/TensorRT riêng, giữ nguyên `.venv` CPU, driver và model artifacts; chạy 5 smoke test thực rồi **dừng để review môi trường trước khi tiếp tục Phase 17**. Đây KHÔNG phải hoàn thành toàn bộ Phase 17 benchmark/production optimization. Không bắt đầu Phase 18.

## 2. What I inspected before coding

Đọc yêu cầu Phase 17 và điều kiện “ONLY if the actual environment supports TensorRT”, sau đó yêu cầu mới về isolated setup/review gate. Kiểm tra nhánh `main`, worktree dirty từ các phase trước, `.gitignore`, Phase 16 contracts/report và môi trường thực. Không có AGENTS.md áp dụng trong repo/các thư mục cha đã kiểm tra. Chụp SHA-256 của 367 file hiện hữu trước khi sửa; không stage/commit/reset.

Thông tin đã kiểm tra/báo cáo trước cài đặt:

| Hạng mục | Kết quả thực |
|---|---|
| GPU vật lý | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB nominal |
| Compute capability | 8.6 |
| NVIDIA driver | 610.62; `nvidia-smi` chạy được |
| CUDA driver API/UMD | 13.3; `cuInit(0)=0`, `cuDriverGetVersion=13030` |
| Toolkit/runtime trước setup | Không tìm thấy nvcc trên PATH/vị trí chuẩn; chưa xác nhận Toolkit. Driver API/UMD không phải phiên bản Toolkit đã cài |
| Python | 3.11.9, Windows x64 |
| PyTorch trong `.venv` CPU | 2.5.1+cpu, CUDA build=None, cuda_available=False |
| ONNX / ONNX Runtime trong `.venv` | 1.17.0 / 1.20.1; không có CUDA/TensorRT EP |
| TensorRT/cuda-python ban đầu | Không có trong interpreter hiện tại |

“CPU-only” ở các báo cáo trước là mô tả Python environment đã dùng, không có nghĩa máy không có NVIDIA GPU. Đối chiếu [TensorRT 10.13.3 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.13.3.html): nhánh CUDA 12 hỗ trợ driver Windows r537+; driver 610.62 đáp ứng. Chọn wheel CUDA 12 rõ ràng thay vì mặc định CUDA mới nhất. Kiểm tra [wheel CUDA PyTorch chính thức](https://pytorch.org/get-started/previous-versions/) và [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html), sau đó xác nhận bằng thực thi GPU thật, không suy ra readiness chỉ từ support matrix.

## 3. Implementation performed

- Tạo `.venv-tensorrt` bằng Python 3.11.9, `include-system-site-packages=false`; không activate/sửa PATH toàn cục và không kế thừa site-packages CPU.
- Cài riêng PyTorch **2.8.0+cu129**, ONNX **1.17.0**, ORT **1.20.1**, TensorRT **10.13.3.9.post1** cùng dependencies cần thiết. NVIDIA wheel kéo `cuda-toolkit[cudart]` 12.9.2.0 và `nvidia-cuda-runtime-cu12` 12.9.79; không cài SDK/nvcc toàn hệ thống. API runtime thực trả **12090 (CUDA 12.9)**, status 0.
- Nâng pip **chỉ trong GPU venv** lên 26.2.1 để hỗ trợ tải tiếp tục/retry sau timeout mạng; pip CPU vẫn 24.0. Không thay driver, không copy/rename DLL, không tắt TLS/checksum, không dùng provider fallback để giả lập GPU.
- Wheel Torch chính thức 3,571,828,172 bytes được tải qua HTTPS và khớp full SHA-256 `8a92b6ac49be932a8e4f70282d0d396a95a0fc877a9fbe0bd36be5f765707c84` từ official index trước khi cài. Đã xóa đúng 16 mảnh tải tạm (~3.57 GB) sau xác minh; wheel hoàn chỉnh giữ trong GPU venv để có thể cài lại. Đây là xử lý transport, không phải workaround tương thích. Một số lượt tải mạng timeout đã được retry thành công.
- Thêm smoke CLI có guard bắt buộc đúng `.venv-tensorrt`, không dùng CPU fallback. Chỉ nhận `--smoke`, không có tham số nạp model chính/benchmark. Mạng test tự tạo Conv+ReLU, batch1, input 1×3×16×16, không checkpoint/dataset/registry.
- Smoke build hai engine riêng: graph/tensor FP32 và FP16 strongly-typed, TF32 tắt, workspace 64 MiB, build level0. Deserialize, bind tensor CUDA/stream, thực thi GPU thật, synchronize và so output với reference. Giải phóng children/owners/buffers theo thứ tự trước probe tiếp theo. Workspace limit không phải giới hạn tổng VRAM.
- Lưu dependency snapshot và báo cáo; ONNX/engine toy chỉ dùng thư mục tạm, không giữ binary engine, không sửa model artifacts hiện hữu.

## 4. Files created

- `requirements-tensorrt.txt`
- `requirements-tensorrt.lock.txt`
- `scripts/check_phase17_environment.py`
- `tests/test_tensorrt_environment_smoke.py`
- `docs/implementation/phase_17_environment_setup.md`
- `docs/implementation/phase_17_environment_report.json`

Ngoài source: `.venv-tensorrt/` mới, được ignore, chứa vendor packages, transport helper và cached wheel đã xác minh. Snapshot khoảng 13.72 GB kể cả cached wheel 3.57 GB; không đưa môi trường/binary vào Git. Lockfile ghi phiên bản thực của 25 packages; không phải chứng nhận release hay lock theo hash cho mọi dependency.

## 5. Files modified

Chỉ `.gitignore`: thêm dòng `.venv-tensorrt/`. 366/367 file hiện hữu ngoài thay đổi này giữ nguyên SHA, không sửa application code, test cũ, `.env`, `.venv`, requirements CPU, driver hoặc model/reference.

So sánh pip freeze CPU trước/sau: **khớp hoàn toàn**, không có dependency cũ bị nâng/hạ/gỡ. `pip check` cả CPU/GPU PASS. GPU venv có dependencies riêng; không phải migration backend sang GPU.

## 6. Architecture before

Backend/AI chạy trong `.venv` CPU, supervised có ONNX export/parity offline. Có GPU NVIDIA vật lý/driver nhưng interpreter dự án chưa có CUDA/TensorRT stack.

## 7. Architecture after

Backend và pipeline cũ vẫn nguyên trong `.venv` CPU. Nhánh độc lập `.venv-tensorrt` cung cấp CUDA runtime/PyTorch GPU/native TensorRT và smoke CLI. ORT trong GPU venv vẫn dùng CPU provider để làm reference; native TensorRT engine mới là phần chạy GPU. Không yêu cầu ORT CUDA/TensorRT EP cho smoke này.

GPU venv chưa phải full backend environment: không chuyển FAISS, torchvision, API/DB hoặc segmentation/anomaly runtime sang đó. Không nạp đồng thời hai model lên GPU; chưa có main-model scheduler/benchmark mới.

## 8. Backward compatibility

55 artifact Phase 0 khớp SHA; fingerprint storage 311 file, Data 643 file, user files 10 file, cache trọng số 3 file không đổi so với Phase 16. `.venv` CPU vẫn Torch2.5.1+cpu/CUDA=None/ONNX1.17.0/ORT1.20.1 và freeze không đổi. Driver sau setup vẫn **610.62**. Registry hai fabric vẫn INACTIVE, không kích hoạt candidate. Không thay schema HTTP/DB, threshold hoặc inference behavior.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| CUDA device detected | CUDA thật thấy RTX3050, capability8.6 | PASS |
| PyTorch CUDA inference | Conv GPU FP32 và FP16 so CPU reference | PASS |
| ONNX model load | Load/check graph và ORT CPU inference | PASS |
| TensorRT import | Import native TensorRT10.13.3.9.post1 | PASS |
| Tiny engine build/reload/inference | Engine FP32 và FP16, batch1, GPU execution thật | PASS |
| 17 CPU safety/fixture tests | Từ chối venv sai, isolation, graph shape/dtype, input/output errors | PASS |
| Freeze/driver/artifact checks | Không phá CPU environment/driver/models | PASS |

17 unit tests chạy trong CPU venv: **17 passed in 0.30s**, không warning. Đây không thay cho GPU evidence; 5 smoke checks phía trên chạy thật, không mock. Compileall, git diff --check, pip check PASS.

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Lần kiểm tra sau setup: **1741 passed, 1 warning in 148.37s**, exit0. Tăng 17 so với 1724 của Phase16; không sửa hoặc làm hỏng test cũ. Frontend build PASS, 22 modules; 77 smoke assertions PASS. DB/storage test được cô lập. Warning duy nhất là FAISS NumPy private namespace có từ trước. Đây là hồi quy phần mềm CPU, không phải benchmark TensorRT model chính.

## 11. Hypothetical / synthetic cases tested

Chỉ mạng Conv+ReLU tổng hợp, không phải Fabric-Detection model. Input batch1 1×3×16×16; output 1×4×14×14. Mọi ONNX/engine toy nằm trong thư mục tạm, bị xóa khi script kết thúc. Raw smoke/provenance/hash lưu trong [JSON](phase_17_environment_report.json).

| Check | Max absolute error | Điều kiện |
|---|---:|---|
| PyTorch CUDA FP32 vs CPU | 2.9802322e-8 | atol=rtol=1e-5, PASS |
| PyTorch CUDA FP16 vs quantized-input CPU reference | 2.8491020e-4 | atol=rtol=1e-3, PASS |
| ONNX/ORT FP32 vs CPU | 5.9604645e-8 | atol=rtol=1e-5, PASS |
| TensorRT engine FP32 vs CPU | 5.9604645e-8 | atol=rtol=1e-5, PASS |
| TensorRT engine FP16 vs quantized-input CPU reference | 3.3199787e-4 | atol=rtol=1e-3, PASS |

Engine đã deserialize và execute trên CUDA, không chỉ serialize hoặc import. Graph/I/O FP16 được kiểm tra type thực, không suy từ một FP16 builder flag. Không coi các tolerance toy này là acceptance criteria cho CNN chính. **Không đo P50/P95, throughput hoặc peak VRAM; tất cả để NOT_MEASURED/null, không điền 0.** Batch2/4 chưa thử.

## 12. Known limitations

- Chỉ chứng minh stack GPU hoạt động với mạng nhỏ; chưa chứng minh segmentation/anomaly chính parse được, đạt parity FP16, chạy vừa 4GB hoặc nhanh hơn CPU.
- Chưa benchmark/optimize model chính, chưa đo contention hai model, latency distribution, throughput hay VRAM peak. Tổng VRAM thiết bị/64MiB workspace không phải số đo peak usage.
- Torch GPU2.8 khác Torch CPU2.5.1; CPU environment giữ nguyên. Candidate chính cần numerical validation riêng khi được duyệt tiếp, không mặc nhiên chuyển weights/runtime.
- CUDA runtime/pip cudart đã có, không phải SDK/nvcc/C++ toolchain toàn cục. Custom CUDA plugins/source builds chưa kiểm tra.
- GPU venv là công cụ setup/smoke/native TensorRT, không thay full backend environment hoặc đồng nghĩa release/production approval.

## 13. Risks introduced

Môi trường riêng và binary CUDA chiếm dung lượng đáng kể. Smoke nhỏ không dự báo khả năng chịu tải/lỗi số học FP16 của CNN. 4GB VRAM dễ đầy khi giữ segmentation và anomaly cùng lúc; chưa thử residency đó. TensorRT engines phụ thuộc stack/GPU nên không suy portability từ toy engine. Giữ isolation và review gate; không sửa driver hay dùng DLL/TLS workaround.

## 14. Manual verification steps for me

Chạy đúng interpreter, không cần activate hoặc sửa ExecutionPolicy/PATH:

```powershell
.\.venv-tensorrt\Scripts\python.exe scripts/check_phase17_environment.py --smoke
.\.venv-tensorrt\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/test_tensorrt_environment_smoke.py -q
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
```

Mong đợi all_passed=true, CUDA available/device_count1, runtime build12.9, TensorRT10.13.3.9.post1, FP32/FP16 tiny GPU engine PASS; main_model_benchmark_performed=false, batch_2_or_4_tested=false, phase18_started=false. Có thể dao động sai số nhỏ nhưng phải đạt tolerance. Engine hashes có thể thay đổi giữa lần build, không cần giống tuyệt đối.

Nếu tái tạo môi trường trên một bản sao dự án, trước hết xác minh target CHƯA tồn tại; không xóa/ghi đè môi trường đang dùng. Chỉ cài vào GPU venv riêng:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-tensorrt
.\.venv-tensorrt\Scripts\python.exe -m pip install pip==26.2.1
.\.venv-tensorrt\Scripts\python.exe -m pip install --timeout 180 --resume-retries 5 -r requirements-tensorrt.lock.txt
.\.venv-tensorrt\Scripts\python.exe scripts/check_phase17_environment.py --smoke
```

Nếu GPU import/build/inference thật thất bại, dừng và lưu lỗi; không tự đổi driver, bỏ guard, rename DLL, tắt TLS hoặc nhận CPU fallback là TensorRT pass. Rollback vận hành chỉ cần ngừng dùng GPU venv; `.venv`/API/model/driver cũ không đổi nên không cần rollback chúng. Không tự xóa môi trường đã tạo; backup/archive nếu muốn hoàn tác file mới. Cached wheel còn nguyên, 16 part downloads tạm đã được dọn, không có dữ liệu người dùng bị xóa.

## 15. Git diff summary

Phần setup này: **6 file mới, 1 file hiện hữu sửa** (`.gitignore` +1); 366 file hiện hữu khác giữ nguyên SHA. GPU venv được ignore. Diff tracked tích lũy: **10 files changed, 511 insertions(+), 132 deletions(-)**, gồm các phase trước và binary người dùng, không phải riêng Phase17. Không stage/commit; không sửa binary.

## 16. Recommendation

SAFE TO APPROVE — chỉ cho isolated setup và 5 smoke checks. **Dừng để người dùng review trước khi benchmark/optimization model chính trong Phase17.** Không coi Phase17 đầy đủ đã hoàn thành, không chuyển Phase18, không cấp production approval.

## 17. Tiến độ so với yêu cầu

| Yêu cầu hiện tại | Trạng thái |
|---|---|
| Kiểm tra/báo cáo GPU, driver, CUDA, Python, Torch/ONNX/ORT trước cài | Hoàn thành |
| Tạo GPU env riêng, không thay CPU dependencies/driver/artifacts | Hoàn thành, có freeze/hash/driver đối chiếu |
| CUDA + PyTorch GPU + ONNX load + TensorRT import + tiny engine | 5/5 smoke thực PASS; batch1 FP32/FP16 |
| Dừng sau setup để review | Đang chờ người dùng review |
| Phase17 gốc: P50/P95/throughput/VRAM model chính | CHƯA BẮT ĐẦU theo yêu cầu mới |
| Batch2/4, giữ segmentation+anomaly đồng thời trên GPU | CHƯA THỬ; chỉ xem xét sau đo VRAM và review |
| Phase18 acceptance/release governance | CHƯA BẮT ĐẦU |

Project vẫn ở mốc phần mềm Phase16 + setup môi trường Phase17, không phải đã xong 17/18 phase. Chưa có bằng chứng mới về accuracy, false-positive reduction, model production hoặc SLA dây chuyền. Các trở ngại dữ liệu thật/provenance/production training và acceptance độc lập đã nêu ở Phase16 vẫn còn. Không tự tiến hành phần benchmark còn lại sau báo cáo này.

============================================================
PHASE 17 SETUP AND SMOKE TEST COMPLETE
STATUS: WAITING FOR USER REVIEW
MAIN MODEL BENCHMARK HAS NOT BEEN STARTED
PHASE 18 HAS NOT BEEN STARTED
============================================================
