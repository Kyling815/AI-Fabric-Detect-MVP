# Phase 15 — Runtime Profiling and Async Scheduler

## 1. Goal

Đo runtime trước khi tối ưu; bổ sung scheduler có kiểm soát. Hoàn thành phần mềm và benchmark CPU trong phạm vi Phase 15. Không bắt đầu ONNX/TensorRT, không huấn luyện hoặc kích hoạt mô hình/reference.

## 2. What I inspected before coding

Đối chiếu yêu cầu Phase 15 trong bản yêu cầu 18 phase; đọc đường chạy InspectionService → CurrentAnomalyDetector → SharedDenseResNet18 → FAISS/localization, shadow subprocess, fusion, QC, contracts/settings, simulator Phase 13 và báo cáo Phase 14. Kiểm tra fixture golden, công cụ hồi quy an toàn và cache model hiện có.

Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`. Working tree đã có thay đổi từ các phase trước và binary người dùng; không stage/commit/reset. Chụp SHA-256 của 349 file hiện hữu trước khi sửa. Máy dùng Python 3.11.9, PyTorch 2.5.1+cpu, không có CUDA khả dụng.

## 3. Implementation performed

- `RuntimeProfiler`: đo `preprocessing_ms`, `segmentation_ms`, `anomaly_ms`, `fusion_ms`, `postprocessing_ms`, `total_ms`, `queue_wait_ms`, thêm `uninstrumented_ms`, số lần gọi và exception từng stage. Dùng `perf_counter_ns`, ContextVar tách worker; span lồng nhau được trừ khỏi parent để không đếm trùng.
- Gắn mốc vào decode/resize/normalize, CNN, FAISS branch, localization, mapping, shadow, fusion và QC. Profile nội bộ và audit logger chỉ bật khi `FABRIC_RUNTIME_PROFILING_ENABLED=true`; mặc định OFF. Không đưa profile vào HTTP hoặc DB schema.
- `RuntimeScheduler`: future-based background workers, pending queue hữu hạn, admission không chặn. Queue đầy ném `SchedulerBusy` trước khi nhận job; caller giữ input để retry hoặc dừng, không tự bỏ tile. Mặc định khóa tài nguyên chung trong một process tuần tự hóa cả job.
- Job lỗi trả `ScheduledResult(succeeded=false, value=None, error_type=...)`; worker tiếp tục chạy. Caller phải kiểm tra `succeeded`, không suy ra GOOD từ kết quả thiếu. Có thống kê accepted/completed/failed/cancelled/rejected/running/pending, hủy trước khi bắt đầu và graceful drain.
- Benchmark thực sự chạy CNN anomaly đã đăng ký + adapter và encoder segmentation độc lập có decoder chưa huấn luyện, FAISS bank tạm và FusionEngine thật trên tile giả lập. Đo hai nhánh riêng, cả cặp tuần tự và cặp chạy đồng thời trên CPU; không dùng công thức `max(T1,T2)`.

Quy ước quan trọng: `total_ms` chỉ bao phủ callable/core inference. `queue_wait_ms` tính từ nhận job đến lấy được khóa tài nguyên, tách khỏi total; không gồm thời gian caller chờ retry khi admission bị từ chối. Upload, capture pre-screening/image-info trước core trong API, annotation, SQLite và HTTP không nằm trong profile này. Pattern observation, nếu bật, được tính vào preprocessing; segmentation shadow bao gồm thời gian chờ subprocess. `stage_calls=0` nghĩa là nhánh không chạy. `stage_errors` chỉ đếm exception được ném, không đếm diagnostic lỗi đã được branch xử lý; xem thêm trạng thái shadow/fusion trong InspectionResult.

## 4. Files created

- `backend/app/core/runtime_profiling.py`
- `backend/app/services/runtime_scheduler.py`
- `scripts/check_phase15_runtime.py`
- `tests/test_runtime_profiling.py`
- `tests/test_runtime_scheduler.py`
- `tests/test_runtime_profiling_integration.py`
- `docs/implementation/phase_15_runtime_profiling.md`
- `docs/implementation/phase_15_benchmark_report.json`

## 5. Files modified

- `.env.example`: thêm flag OFF, không chỉnh `.env` thật.
- `backend/app/core/config.py`: setting profiling mặc định OFF.
- `backend/app/core/inspection_contracts.py`: thêm optional profile nội bộ.
- `backend/app/ai/current_anomaly.py`: scope anomaly và giữ nguyên exception translation.
- `backend/app/ai/dense_feature_extractor.py`: phân tách timing preprocessing/CNN/spatial cells, không đổi tensor/embedding.
- `backend/app/ai/dense_anomaly.py`: span localization, giữ nguyên phép score/ngưỡng/morphology.
- `backend/app/services/inspection_service.py`: wrapper profiler, span orchestration và audit best-effort, không đổi schema hoặc decision policy.

Không sửa test cũ, frontend, registry, checkpoint, threshold, calibration, dataset hoặc SQLite thật.

## 6. Architecture before

API đồng bộ → service → anomaly → shadow tùy chọn → fusion/QC. Các layer stream/coordinates/seam riêng biệt; chưa có stage ledger hay inference job queue có giới hạn.

## 7. Architecture after

API vẫn đồng bộ như trước; core có profiler opt-in và giữ nguyên thứ tự nhánh. Luồng offline chủ động: caller → bounded admission → worker → khóa tài nguyên mặc định → profiled callable → future chứa kết quả/timing hoặc lỗi. Scheduler chưa tự nối với camera/simulator, API hoặc DB.

Với nhiều worker: pending queue lấy theo FIFO, nhưng khóa tài nguyên không bảo đảm thứ tự thực thi/hoàn tất; consumer cần đọc future theo thứ tự nguồn nếu cần giữ thứ tự tile. Giới hạn tổng job được sở hữu là `max_pending + workers`; payload tính theo số job, không phải byte. Bất biến: accepted = completed + cancelled + running + pending; failed là tập con của completed.

## 8. Backward compatibility

Golden POST/GET và OpenAPI không đổi khi profiling OFF/ON; nguyên nhãn GOOD/NEEDS_REVIEW/BAD, reference identity, boxes, embedding, persistence và cleanup. Shadow failure vẫn diagnostic, anomaly vẫn khả dụng, không tự bật hybrid QC. Khi đã có profiler ngoài của scheduler, profiler ngoài sở hữu kết quả hoàn chỉnh; không tạo session lồng trong service.

342/349 file có sẵn không đổi; đúng 7 file trong phạm vi được sửa. 55 artifact Phase 0 khớp SHA; fingerprint storage 311 file, Data 643 file, file người dùng 10 file và cache trọng số 3 file không đổi so với Phase 14. A/B vẫn INACTIVE; API trả 409 như trước, không kích hoạt để lấy số liệu.

## 9. Test cases executed

| Test | Purpose | Result |
|---|---|---|
| Profiler deterministic clock, nested spans | Không cộng trùng, tách queue wait, exception cleanup | PASS |
| CPU/auto/CUDA fallback, CUDA fence mocks | Không gọi CUDA trên CPU; đường CUDA chỉ được kiểm bằng mock | PASS |
| Bounded queue, cancellation, drain, self-join | Không silent drop; bảo toàn accounting và vòng đời | PASS |
| Resource gate, explicit overlap, setup failure | Kiểm soát concurrency, không strand future/worker | PASS |
| Batch=1 actual CNN, multiple service tiles | Embedding parity và profile riêng mỗi job | PASS |
| Golden HTTP/GET/schema, branch/persistence/log failure | Giữ nguyên output, diagnostic, cleanup, persistence | PASS |
| Benchmark CLI subprocess | CNN thật, toy bank, queue pressure, branch failure | PASS |
| 55 protected artifacts và 4 scope fingerprints | Không đổi model/data/registry/file người dùng | PASS |

Lệnh riêng Phase 15: ` .\.venv\Scripts\python.exe -m pytest tests/test_runtime_profiling.py tests/test_runtime_scheduler.py tests/test_runtime_profiling_integration.py -q ` → **64 passed, 1 warning in 9.48s**. Compileall và `git diff --check` PASS.

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Kết quả thực tế: **1617 passed, 1 warning in 180.53s**, exit 0; tăng 64 test so với 1553 của Phase 14, không sửa hoặc làm hỏng test cũ. Frontend TypeScript/Vite build PASS, 22 modules. Smoke PASS 77 assertions, bao gồm forward backbone/cache hiện có, FAISS, HTTP/SQLite và fail-safe A/B inactive. DB/storage kiểm thử được cô lập. Warning duy nhất là FAISS loader dùng NumPy private namespace, đã có trước Phase 15.

## 11. Hypothetical / synthetic cases tested

Đây là kiểm thử phần mềm và latency, không phải đánh giá dữ liệu nhà máy. Benchmark được đo lại sau khi hồi quy kết thúc, lúc `2026-10-15T03:49:05.079163+00:00`; raw samples/provenance lưu trong [báo cáo JSON](phase_15_benchmark_report.json).

CPU, batch=1, RGB 64×64, 3 warmups, 12 mẫu đo, 1 PyTorch thread và 1 FAISS thread. Anomaly dùng backbone v4, pretrained encoder cache và adapter hiện có; segmentation dùng ResNet18+U-Net độc lập với decoder ngẫu nhiên chuẩn; bank tạm có 64 vector. Không download, train hoặc ghi checkpoint. Synthetic supervised evidence không được cho phép dùng để chấp nhận chất lượng.

| Stage | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Preprocessing | 0.529 | 0.682 |
| Segmentation | 23.896 | 32.943 |
| Anomaly | 10.669 | 15.636 |
| Fusion | 0.580 | 0.667 |
| Postprocessing | 0.125 | 0.137 |
| Total core | 36.159 | 48.352 |

Percentile mỗi stage được tính độc lập, không cộng các p50 để suy ra p50 total. Setup model/index: 341.644 ms. `cold_profile` là lần full pipeline đầu SAU setup, không phải process cold start; anomaly encoder đã chạy để tạo bank. Mẫu đó total 24.795 ms, không giả định cold luôn chậm hơn warm.

Hai CNN riêng: segmentation p50 23.774 ms, anomaly p50 11.128 ms. Cặp tuần tự đo được p50 35.744 ms; cặp CPU đồng thời thực sự đo được p50 27.080 ms, không bằng cách giả định lấy max. Probe ngắn, thứ tự cố định và có dispatch/barrier overhead; chưa đủ kết luận speedup ổn định hoặc thay default serialization.

Scheduler 2 workers có khóa mặc định: 8/8 tile hoàn tất, 0 failed, wall gồm startup/drain 318.953 ms, khoảng 25.082 tile/s CHỈ cho cấu hình toy này; queue/resource wait 0.079–276.423 ms. Tình huống queue đầy: nhận 2, từ chối 1 rõ ràng; 1 lỗi segmentation được báo đúng, worker vẫn drain. Không đổi thành ngưỡng chấp nhận, FPS camera hoặc tốc độ dây chuyền.

## 12. Known limitations

- GPU contention/VRAM chưa đo: PyTorch CPU-only. Đồng bộ CUDA mới được test bằng mock; benchmark CLI hiện chỉ CPU, không chứng nhận đường GPU.
- Không có checkpoint segmentation sản xuất hoặc reference/ảnh đại diện nhà máy trong benchmark. Không đo accuracy, domain shift hay giảm false positive.
- Benchmark in-process không bao gồm chi phí cold-start/checkpoint validation/transport của shadow subprocess Phase 9. Không đo acquisition, PLC, API end-to-end hoặc độ ổn định tải kéo dài.
- Scheduler chỉ in-memory/process-local, chưa tích hợp camera/stream/API; không bền vững khi process chết, không liên tiến trình, không auto retry, không deadline/preempt native call. `shutdown(wait=false)` là yêu cầu drain, không phải kill. Caller phải đóng scheduler và giữ input/model/context còn sống đến khi job hoàn tất.
- Không chia sẻ SQLAlchemy Session cho worker. Không chờ job cùng resource gate từ trong một job đang giữ gate vì có thể deadlock. Chế độ concurrency chủ động đòi hỏi mô hình/caller thread-safe và quản lý thứ tự kết quả.

## 13. Risks introduced

Timing/audit có overhead; CUDA device-wide fences có thể làm đổi lịch thực thi và đo cả contention, không phải kernel time. Khóa chung có thể gây head-of-line blocking giữa scheduler. Payload lớn làm tăng RAM dù số job hữu hạn; job treo có thể chặn graceful shutdown. Explicit concurrent opt-in có thể tranh CPU/GPU hoặc làm lộ model mutable state, nên mặc định vẫn serialized. Logger best-effort không phải telemetry sink bền vững.

## 14. Manual verification steps for me

Từ thư mục dự án, chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_profiling.py tests/test_runtime_scheduler.py tests/test_runtime_profiling_integration.py -q
.\.venv\Scripts\python.exe scripts/check_phase15_runtime.py --benchmark --iterations 12 --warmups 3 --tiles 8 --size 64
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Mong đợi: 64 test Phase 15; hồi quy 1617 test, frontend PASS và 77 dòng PASS smoke. Benchmark JSON ghi batch=1, CPU, 8 accepted/completed, queue pressure bị từ chối, failure rõ ràng, output parity PASS, GPU NOT_MEASURED. Timing sẽ dao động, không cần khớp số đo đã lưu. CLI chỉ in JSON, dùng file tạm hệ điều hành, không ghi báo cáo đè vào storage/references.

Kiểm tra profile trong fixture mà không bật model/reference thật:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_runtime_profiling_integration.py -k "audit or golden or failure" -q
```

Khi chủ động bật flag trong một process thử nghiệm, audit logger là `backend.runtime_profile.audit`, profile nội bộ ở `InspectionResult.runtime_profile`; HTTP không có field mới. Không share DB Session, không đưa `create_fabric_inspection` vào scheduler; chỉ submit callable inference độc lập, xử lý SchedulerBusy và kiểm tra ScheduledResult.succeeded trước khi dùng value. Future bên ngoài chứa profile hoàn chỉnh.

Rollback vận hành: giữ/đặt `FABRIC_RUNTIME_PROFILING_ENABLED=false`, restart process thử nghiệm và bỏ caller opt-in scheduler; API vốn chưa chuyển async. Không cần rollback data/model/DB. Nếu muốn hoàn tác mã, backup working tree rồi chỉ hoàn tác các hunk Phase 15 trong 7 file đã sửa và gỡ 8 file mới nêu trên; không dùng reset/checkout toàn file vì có thay đổi Phase 1–14 và công việc người dùng.

## 15. Git diff summary

Phase 15: **8 file mới, 7 file hiện hữu được sửa**, 342 file hiện hữu ngoài phạm vi giữ nguyên SHA. `git diff --stat` tích lũy: **9 tracked files changed, 510 insertions(+), 132 deletions(-)** và binary người dùng vốn đã sửa; số này gồm phase trước và không gồm các file untracked. Không stage/commit; không sửa binary. Inventory và fingerprint cụ thể có trong báo cáo JSON.

## 16. Recommendation

SAFE TO APPROVE — cho profiler mặc định OFF, scheduler offline có giới hạn và benchmark CPU có phạm vi rõ ràng. Không phải phê duyệt production, GPU concurrency hoặc model quality. Trước khi quyết định GPU parallelism cần GPU thật, checkpoint/reference đại diện và benchmark tuần tự/đồng thời dài hơn; không suy ra `max(T1,T2)`. Phase 16 chưa bắt đầu.

============================================================
PHASE 15 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
