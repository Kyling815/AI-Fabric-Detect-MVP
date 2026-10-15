# Phase 3 — Pattern Period Estimator in Observe-Only Mode

Ngày: 2026-10-13, Asia/Bangkok. Phạm vi: chỉ Phase 3; chờ người dùng review.

## 1. Goal

Thêm `PatternPeriodEstimator` ước lượng chu kỳ texture theo hàng/cột bằng autocorrelation (ACF). Kết quả chỉ là quan sát nội bộ: không thay đổi CNN, patch size, embedding, FAISS, threshold, localization hoặc quyết định QC. Không triển khai Phase 4.

## 2. What I inspected before coding

- Yêu cầu Phase 3 trong pasted request; báo cáo nghiên cứu và kiến trúc đã được đối chiếu ở các phase trước. Đây là triển khai ý tưởng intensity-profile/ACF, không tuyên bố tái hiện đầy đủ kết quả của bài báo.
- Worktree Phase 2: Quality Gate default OFF, hợp đồng nội bộ, orchestration, Settings, registry, golden snapshots, fixtures và runner baseline.
- Hàm FFT/orientation hiện có không cung cấp period estimator theo row/column ACF. Không sửa chúng hay preprocessing cũ.
- Registry thật: A/B INACTIVE, không có active dense reference; CNN ResNet18 layer2+3 với adapter v4, 128 chiều, stride 8. Registry/index/checkpoint thật chỉ được đọc.
- Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`. Các thay đổi Phase 0–2 và user files đã có sẵn; không stage/commit.
- Snapshot hash 199 tệp hiện có trước khi sửa Phase 3. Không có AGENTS.md áp dụng trong workspace/ancestors đã kiểm tra.

## 3. Implementation performed

Estimator độc lập có `estimate(ndarray, config)` và `estimate_path(path, config)`:

1. Kiểm tra integrity khi đọc file, áp dụng EXIF orientation, chuyển grayscale. API ndarray nhận grayscale, RGB/RGBA hoặc float [0,1]; không mutate input.
2. Tạo bản phân tích riêng, giới hạn cạnh dài 1024 px mặc định; không đưa bản này vào CNN. Map lag về pixel của ảnh gốc đã định hướng.
3. Lấy profile trung bình toàn ảnh và tối đa bảy dải hẹp 3 px trên mỗi trục. Checker cân bằng có thể triệt tiêu profile toàn ảnh; dải hẹp giữ được thông tin chu kỳ.
4. Loại mean và slope ánh sáng tuyến tính của profile; bỏ profile có std dưới 2 intensity units.
5. Tính ACF bằng FFT zero-padding, chuẩn hóa theo số mẫu overlap và lag 0; tìm local positive peaks. Kiểm tra peak thứ hai gần 2×lag, correlation tối thiểu 0.45.
6. Giới hạn period gốc 4–256 px, ít nhất bốn chu kỳ, profile tối thiểu 64 mẫu phân tích. Cluster period theo tolerance 10%; confidence mỗi trục là median repeat/spacing score × tỷ lệ đồng thuận trên profile có đủ năng lượng.
7. Chấp nhận period mỗi trục nếu confidence >= 0.65; nếu không, period của trục đó là `None`. Confidence tổng là max hai trục; `is_periodic=True` nếu ít nhất một trục có period được chấp nhận.

`PatternPeriodResult` là dataclass frozen, chứa `period_x_px`, `period_y_px`, `confidence`, `is_periodic`, thêm confidence từng trục và reason nội bộ. `confidence` là heuristic, không phải xác suất đúng được hiệu chuẩn.

Fallback reasons: `low_periodicity_confidence`, `one_axis_only`, `unsupported_input`, `image_read_error`, `configuration_error`, `observer_error`. Khi không đủ tin cậy, không ép thành motif-aware; runtime tiếp tục dùng representation cố định hiện tại.

Tích hợp service:

- Global flag `FABRIC_PATTERN_ESTIMATOR_ENABLED=false`; đường dẫn config tùy chọn. Có override `pattern_estimator` theo fabric, không sửa production registry.
- Chỉ chấp nhận `mode="observe_only"`. Quality Gate giữ nguyên precedence và cơ chế lỗi Phase 2.
- Quan sát chạy sau evidence, localization và quyết định QC. Period chỉ gắn vào `InspectionResult.pattern_period_result` và log; không có consumer trong model/patching/FAISS/rule engine.
- Lỗi config/observer không chặn inspection: log cảnh báo và fallback nội bộ. OFF không đọc config file và không gọi estimator.
- Không bổ sung HTTP/DB fields. CLI read-only in JSON ra stdout để review mà không chạy inspection hay activate reference.

## 4. Files created

- `backend/app/core/pattern.py` — config frozen, validation, observe-only mode.
- `backend/app/ai/pattern_period.py` — estimator, ACF/peaks/consensus, fallback result.
- `config/pattern_observation_example.json` — ví dụ tham số, không tự bật flag.
- `scripts/observe_pattern_period.py` — CLI read-only.
- `tests/pattern_period_fixtures.py` — synthetic deterministic, không đụng locked data.
- `tests/test_pattern_period.py` — 55 unit/CLI cases.
- `tests/test_pattern_observation_integration.py` — 29 integration cases.
- `docs/implementation/phase_3_parity_report.json` — sáu ON/OFF parity cases với model/index thật.
- `docs/implementation/phase_03_pattern_observation.md` — báo cáo này.

## 5. Files modified

- `.env.example` — hai biến pattern observation mặc định OFF và comment.
- `backend/app/core/config.py` — hai Settings defaults, không sửa private `.env`.
- `backend/app/core/inspection_contracts.py` — config/error nội bộ trong context và observation result tùy chọn.
- `backend/app/services/inspection_service.py` — injection, config resolution và quan sát sau QC, fail-soft.

Không chỉnh tests/fixtures cũ, pretrained model, checkpoint, memory bank/index, reference/threshold, DB schema, dataset, frontend, Revision.md hay research report.

## 6. Architecture before

Upload và context của fabric đi qua Quality Gate tùy chọn, rồi CurrentAnomalyDetector chạy CNN/adapter và FAISS. Service map localization về ảnh gốc, áp dụng QC hiện tại, lưu inspection/ảnh annotated và trả response cũ. Legacy placeholder/patch pipeline tồn tại độc lập.

## 7. Architecture after

Luồng quyết định nói trên giữ nguyên. Sau khi đã có QC decision, service có thể gọi PatternPeriodEstimator trên ảnh capture gốc. Nhánh này chỉ trả debug metadata/log vào kết quả nội bộ; nhánh lưu DB/API vẫn dùng evidence/decision cũ. OFF bỏ qua toàn bộ quan sát.

Legacy service từ `create_placeholder_inspection` trở đi giữ nguyên byte so với worktree trước Phase 3.

## 8. Backward compatibility

Default OFF, hợp đồng HTTP/OpenAPI/SQLite/frontend giữ nguyên. ON không truyền period vào feature extraction, adapter, patching, thresholds hay decision. Các trường hợp registry invalid, capture invalid và encoder failure vẫn giữ status/cleanup và không gọi observer.

Các golden snapshots POST/GET của năm case QC giữ nguyên cả ON. Tám ảnh synthetic kiểm tra API/ảnh output ON/OFF với encoder fixture và FAISS thật. Sáu case bổ sung dùng CNN/adapter CPU cache và FAISS v5 thật trong DB/storage/registry tạm: embedding, patch scores, full normalized response, persisted response và annotated SHA-256 đều bằng nhau. Chỉ normalize ID/time/generated URLs.

Hash kiểm tra sau cùng:

| Phạm vi | Kết quả |
| --- | --- |
| 55 artifact/config/cache hashes của Phase 0 | 55/55 khớp |
| storage: 311 files | fingerprint khớp Phase 0, gồm DB/model/index/registry |
| Data: 643 files | fingerprint khớp Phase 0 |
| User files: 10 files | fingerprint khớp Phase 0; dùng đúng thứ tự mixed relative/absolute paths ban đầu |
| Cached weights: 3 files trong torch/hub/checkpoints | fingerprint khớp Phase 0 |
| 199 pre-existing source/document/tracked file hashes | chỉ bốn tệp ở mục 5 thay đổi; 195 tệp khác giữ nguyên |

Fingerprint ba cached weights không bao gồm toàn bộ repository code/cache DINO trong torch/hub. So sánh scope khác hoặc thay đổi thứ tự path sẽ tạo digest khác; không được coi đó là model mutation.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| Pattern estimator + CLI | Synthetic periods, input formats, EXIF, resize, corrupt/missing capture, config validation, repeatability, input immutability | PASS — 55 cases |
| Observation integration | Default OFF, ON parity, per-fabric/global policies, fail-soft errors, Quality Gate precedence | PASS — 29 cases |
| Full regression | Giữ 197 case cũ và thêm 84 case | PASS — 281 cases |
| Golden POST/GET/OpenAPI | Không đổi public contract và năm trạng thái QC fixture | PASS |
| Real CNN/adapter + v5 FAISS ON/OFF | Hai fabrics × checker/scratch/reference Good; exact response, embedding, score và annotated parity | PASS — 6 pairs |
| Phase 0 artifact/API smoke | Cached model, 10 index/bank pairs, API/storage/error paths | PASS — 77 assertions |
| Frontend build | TypeScript và Vite compatibility | PASS — 22 modules |
| Compile/config/Git checks | New Python syntax, JSON validation, whitespace và protected hashes | PASS |

## 10. Full regression test result

Lệnh từ workspace root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Kết quả cuối: `281 passed, 1 warning in 10.02s`; frontend `tsc -b && vite build` PASS; smoke PASS 77 assertions. Warning là FAISS import NumPy private namespace đã có từ baseline. Không cài/nâng cấp dependency.

Targeted cuối: `84 passed, 1 warning in 6.00s`. Test runner tạo DB/storage tạm; build chỉ ghi output bị Git ignore. Không chạy inference ghi DB production.

## 11. Hypothetical / synthetic cases tested

Fixture chính 320×288 px; các giá trị dưới đây là kết quả quan sát, không phải benchmark accuracy:

| Input | period_x_px | period_y_px | Confidence | is_periodic |
| --- | --- | --- | --- | --- |
| Checker/box cân bằng | 16 | 24 | 0.9996 | true |
| Sọc dọc | 16 | None | 0.9996 | true, one_axis_only |
| Chấm lặp | 20 | 24 | 0.9994 | true |
| Đồng màu | None | None | 0 | false |
| Nhiễu ngẫu nhiên | None | None | 0 | false |
| Stretch x1.5, y1.25 nearest-neighbor | 24 | 30 | 0.9996 | true |
| Mild linear illumination | 16 | 24 | 0.9998 | true |
| Checker xoay 12°, reflected border | None | None | 0.5282 | false, graceful fallback |

Thêm sọc ngang, mười noise seeds, pure lighting gradient, profile quá ngắn/không đủ cycles, large image diagnostic resize, RGB/RGBA/normalized float, readonly/non-contiguous arrays, NaN/Inf/complex/invalid dimensions, EXIF và CRC hỏng.

Rotation test chỉ chứng minh một fixture fallback an toàn. Stretch test chỉ chứng minh một phép scale synthetic trục thẳng; không đảm bảo arbitrary rotation/stretch, perspective, shear hoặc vải biến dạng thực tế.

## 12. Known limitations

- Chu kỳ đo theo trục ảnh sau EXIF, không theo trục dệt/motif đã rectification. Không auto-rotate, dewarp, crop ROI hay dùng encoder/mm calibration.
- Profile intensity có thể triệt tiêu motif, alias hoặc chọn harmonic; narrow bands giảm một số trường hợp nhưng không giải quyết mọi texture. Tỷ lệ đồng thuận chỉ tính trên profile đủ contrast, không phải tỷ lệ diện tích ảnh có motif.
- Giới hạn period, contrast, cycles và analysis size là heuristic mặc định chưa hiệu chuẩn theo camera/fabric. Downsampling có thể làm mất period nhỏ; không đủ mẫu thì fallback.
- `is_periodic=True` có thể chỉ phản ánh một trục. Không lấy confidence tổng làm đảm bảo cả hai trục; cần xem confidence_x/y và None từng trục.
- Hai ảnh Good thuộc nguồn reference thật cũng fallback trong parity run; chưa có unseen/locked evaluation hay period ground truth của vải thật.
- Observation chưa lưu bền trong DB/API; xem metadata nội bộ hoặc log INFO (phụ thuộc cấu hình logging). Không phải defect classifier, không sinh motif-aware patches, không cải thiện accuracy CNN trong phase này.

## 13. Risks introduced

ON thêm decode/profiles/FFT và memory/time trước khi trả HTTP; chưa benchmark throughput/latency cho dây chuyền. Giới hạn analysis size giảm tải FFT nhưng decode vẫn xử lý capture gốc. Default OFF giữ đường chạy cũ.

False periodicity/harmonic và missed repeat vẫn có thể xảy ra. Vì không có consumer trong QC, chúng không thay đổi label; không được dùng period/confidence này cho quyết định sản xuất khi chưa có phase được duyệt và validation riêng.

Fail-soft observation có thể che lỗi vận hành nếu bỏ qua warning; cần theo dõi `configuration_error`/`observer_error`. Log chỉ ghi fabric id và kết quả chu kỳ, không acquisition identifiers/private image paths.

Rollback: tắt global flag và mọi override per-fabric; nếu revert source, chỉ bỏ Phase 3 hunks ở bốn tệp và chín additions ở trên. Giữ nguyên Phase 0–2 và user work; không cần rollback DB/index/model.

## 14. Manual verification steps for me

1. Chạy full runner ở mục 10. A/B thật vẫn INACTIVE và inspect trả 409; không activate candidate để review.
2. Xem JSON parity: `docs/implementation/phase_3_parity_report.json`.
3. Chạy standalone synthetic demonstration (không DB/camera/model):

```powershell
@'
from dataclasses import asdict
from backend.app.ai.pattern_period import PatternPeriodEstimator
from tests.pattern_period_fixtures import pattern
for kind in ('checker', 'stripes', 'dots', 'uniform', 'noise', 'stretch', 'lighting', 'rotate'):
    print(kind, asdict(PatternPeriodEstimator().estimate(pattern(kind))))
'@ | .\.venv\Scripts\python.exe -
```

4. Đọc ảnh có sẵn bằng CLI read-only:

```powershell
.\.venv\Scripts\python.exe scripts/observe_pattern_period.py test_image/images.jpeg --config config/pattern_observation_example.json
```

Output là JSON observation; không suy diễn rằng ảnh bất kỳ phải periodic. Missing/corrupt input trả `image_read_error`, period None; config CLI không hợp lệ báo lỗi/nonzero. Runtime service lại fail-soft config để không làm đổi QC.

5. Cấu hình ví dụ cho một test process được chủ động bật sau này (không thực hiện trên checkout/private .env):

```powershell
$env:FABRIC_PATTERN_ESTIMATOR_ENABLED = 'true'
$env:FABRIC_PATTERN_ESTIMATOR_CONFIG_PATH = 'config/pattern_observation_example.json'
```

Restart process để Settings cache đọc lại env; bảo đảm logging INFO nếu muốn xem period logs. Sau thử nghiệm, bỏ hai biến env đã set hoặc khôi phục giá trị trước đó. Không đổi production registry. Ví dụ override trong fixture: `{"pattern_estimator": {"enabled": true, "mode": "observe_only"}}`.

`period_x_px` là khoảng lặp theo chiều ngang (column profile); `period_y_px` là chiều dọc (row profile), đơn vị px gốc sau EXIF. Có thể gọi estimator độc lập hoặc xem `InspectionResult.pattern_period_result` từ `_inspect_fabric`; public API vẫn giữ nguyên.

## 15. Git diff summary

Phase 3 riêng: bốn tệp có sẵn được sửa và chín tệp mới. Không đổi prior tests/docs/fixtures; 195/199 pre-existing hashes ngoài bốn tệp đó giữ nguyên.

`git diff --stat` cumulative so với HEAD, gồm Phase 1/2 và executable user đã sửa trước đó:

```text
.env.example                               | 6 additions
backend/app/core/config.py                 | 4 additions
backend/app/services/inspection_service.py | 215 changed lines
tempCodeRunnerFile.exe                     | Bin 44395 -> 44626 bytes
4 files changed, 172 insertions(+), 53 deletions(-)
```

Untracked contracts/additions chưa có trong tracked diff; toàn bộ Phase 0–3 chưa stage/commit. `git diff --check` không có whitespace errors, chỉ cảnh báo LF/CRLF của Git. Không sửa executable trong phase này.

## 16. Recommendation

SAFE TO APPROVE — cho Phase 3 observe-only scaffolding và compatibility đã kiểm tra. Chưa phải production period-estimation acceptance, throughput/accuracy benchmark hay model-release approval.

Dừng tại đây; chờ người dùng review và duyệt rõ trước Phase 4.

```text
============================================================
PHASE 3 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
```
