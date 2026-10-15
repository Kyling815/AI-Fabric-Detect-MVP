# Phase 2 — Machine Vision Input Metadata and Quality Gate Scaffold

Date: 2026-10-13, Asia/Bangkok. User approved proceeding to Phase 2. Phase 3 has not been started.

## 1. Goal

Chuẩn bị contract cho acquisition metadata và thêm Quality Gate có feature flag. Giữ nguyên upload API/UI khi flag OFF; capture lỗi không được chuyển thành BAD fabric. Không tích hợp camera SDK, encoder hardware hoặc line-scan stream trong phase này.

## 2. What I inspected before coding

- Nhánh `main`, commit `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; Phase 1 chưa commit và được giữ nguyên.
- Thay đổi có sẵn của user: executable, Revision/research documents và test images; Phase 0/1 files vẫn chưa stage.
- Kiểm tra Phase 1 contracts/service/component, API/schema snapshots, Settings, FabricRegistry, image decoder, storage/cleanup và fixtures.
- Đọc quality metrics/validation đã có trong preprocessing v2 để dùng lại, tránh tạo thuật toán blur/exposure thứ hai.
- Chụp hash 191 existing files để kiểm tra scope thay đổi. Không có AGENTS.md mới được tìm thấy.

Quyết định thiết kế: metadata mới chỉ mở ở contract/service nội bộ. Không thêm form fields, response fields hoặc migration DB vào `/api/inspect` trong Phase 2. API vẫn chỉ yêu cầu image và fabric_id.

## 3. Implementation performed

### Acquisition metadata

Thêm frozen `AcquisitionMetadata`:

| Field | Meaning |
| --- | --- |
| `lot_id`, `roll_id` | Optional production source identity |
| `camera_id`, `frame_id` | Optional acquisition identity |
| `timestamp` | Optional capture datetime; timezone-aware when consistency checking is enabled |
| `encoder_position` | Optional raw numeric reading; no assumed mm/meter conversion |
| `roi` | Optional `(x, y, width, height)` in EXIF-oriented source pixels; validation only |

`fabric_id` vẫn nằm trên InspectionInput hiện có. Input có default metadata rỗng; cách khởi tạo cũ vẫn hợp lệ. Service nhận optional keyword `metadata=` và chuyển nguyên object vào anomaly component. Metadata chỉ tồn tại trong request nội bộ; chưa được lưu DB hoặc trả trong public response.

### Quality interface and implementation

Thêm `QualityGate` Protocol, `ImageQualityGate`, `QualityGateResult`, `CaptureStatus` và `InvalidCaptureError`.

Gate kiểm tra:

- file decode/integrity bằng decoder Pillow hiện có;
- min width/height;
- Laplacian variance cho blur;
- dark/bright pixel ratios;
- ROI tuple, dimensions và bounds;
- optional metadata presence/type, camera identity, timezone và finite/nonnegative encoder reading.

Dùng lại `calculate_quality_metrics`, `validate_quality`, QualityMetrics và quality fields của PreprocessConfig. Không chạy ROI crop, CLAHE, illumination correction, denoising hoặc các transforms v2. Chỉ đo capture trên full-resolution RGB/EXIF-oriented image; dữ liệu ảnh gốc và CNN input không bị thay đổi.

Quality result là `VALID_CAPTURE` hoặc `INVALID_CAPTURE`; đây là capture evidence, không phải QC result. Gate chạy trước CNN/FAISS. Capture invalid gây HTTP 400 với string detail `INVALID_CAPTURE: ...`, không tạo inspection, defect hoặc annotated image; upload tạm được cleanup.

### Feature flags and configuration

Settings mới:

```text
FABRIC_QUALITY_GATE_ENABLED=false
FABRIC_QUALITY_GATE_CONFIG_PATH=
```

Flag mặc định OFF. File profile chỉ được đọc khi gate bật. Một fabric record có thể chứa optional `quality_gate` object; `enabled` boolean trên fabric được ưu tiên so với global flag. Fabric thresholds được ưu tiên so với global profile, sau đó tới built-in defaults.

Built-in starting values: min width/height 64 px, blur threshold 70, dark/bright ratio 0.20, dark/bright pixel threshold 20/235. Đây là diagnostic defaults, không phải threshold production tối ưu; cần hiệu chỉnh bằng capture thực và loại vải.

`check_metadata=false` mặc định. Nếu bật, metadata vẫn optional trừ các field được chỉ định trong `required_metadata`. Chỉ ID được cung cấp mới kiểm tra kiểu/nonempty; expected camera chỉ kiểm tra khi được cấu hình. Không suy đoán lot, camera, timestamp hoặc đơn vị encoder.

Policy malformed/missing khi enabled gây HTTP 409 trước upload/model; không silently bypass. Gate execution failure giữ fail-safe HTTP 500 và cleanup, không rơi về GOOD.

Giả định: internal caller cung cấp AcquisitionMetadata typed; ROI dùng tọa độ sau EXIF như dense decoder. Metadata consistency là kiểm tra từng capture, chưa kiểm tra thứ tự stream hay đồng bộ encoder/camera.

## 4. Files created

| File | Purpose |
| --- | --- |
| `backend/app/core/acquisition.py` | Metadata and validated quality configuration |
| `backend/app/ai/quality_gate.py` | Quality interface, results/errors and implementation |
| `config/quality_gate_example.json` | Starting diagnostic profile; does not enable the gate |
| `tests/quality_gate_fixtures.py` | Synthetic images and bad-checksum fixture |
| `tests/test_acquisition_contracts.py` | Optional metadata, frozen fields, default flags |
| `tests/test_quality_gate.py` | Standalone image/ROI/metadata/config validation |
| `tests/test_quality_gate_integration.py` | Service/API gating, flags, profiles, cleanup and compatibility |
| `docs/implementation/phase_02_quality_gate.md` | This report |

## 5. Files modified

- `backend/app/core/inspection_contracts.py`: optional metadata, gate config and internal quality result; old constructors retain defaults.
- `backend/app/core/config.py`: two default-OFF settings.
- `.env.example`: documents the two settings; private `.env` was not edited.
- `backend/app/services/inspection_service.py`: resolves opt-in quality profile, checks capture once, carries quality evidence and handles invalid acquisition.

No pre-existing tests, API routes, public schemas, frontend, model/anomaly/QC algorithms, artifact registries, checkpoint/index/threshold files, DB schema or historical phase documents were modified. Legacy service methods from `create_placeholder_inspection` onward remain byte-identical to the Phase 1 version.

## 6. Architecture before

```text
API -> selected Fabric Context
 -> upload -> decode
 -> InspectionInput(path, fabric_id)
 -> current CNN/FAISS evidence
 -> existing QC -> persistence/response
```

## 7. Architecture after

```text
API (unchanged) -> selected Fabric Context + optional gate profile
 -> upload -> InspectionInput(path, fabric_id, optional metadata)
 -> optional Quality Gate
      OFF: skip
      INVALID: capture error + cleanup; stop before AI/QC/DB
      VALID: carry internal quality result
 -> existing decode + CNN/FAISS + QC + persistence/response
```

```diff
+ Optional acquisition metadata and capture-quality evidence
+ Default-OFF global/per-fabric quality policy
+ INVALID_CAPTURE before model inference
  /api/inspect request/response + frontend + normal QC labels: unchanged
  model weights / FAISS / thresholds / existing preprocessing: unchanged
```

## 8. Backward compatibility

With no new flags or fabric policy, gate is skipped completely; it does not decode, calculate metrics or validate metadata. Missing profile files are not read when gate is disabled.

All Phase 1 POST/GET response snapshots and exact OpenAPI snapshots pass unchanged. Existing GOOD/NEEDS_REVIEW/BAD, confidence, scoring, boxes, persistence, errors and static image serving remain controlled by the old anomaly path when OFF.

With an enabled gate, valid image under a matching diagnostic profile also produces the baseline response. Invalid acquisition intentionally returns a capture error rather than a QC label. No `INVALID_CAPTURE` database label or fake zero-defect Good inspection is created.

Before/after checks preserve 311 storage files, 643 Data files, ten original user files and three cached weights. All 55 Phase 0 artifact/config/cache hashes remain unchanged. Four existing files above are the only source/document changes relative to the Phase 1 worktree; the remaining 187 existing-file hashes are unchanged.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| 3 acquisition contract cases | Optional defaults, metadata identity/frozen fields, gate defaults OFF | PASS |
| 41 standalone gate cases | Image metrics, corrupt/missing/CRC input, ROI, EXIF, extension/grayscale, metadata, configuration | PASS |
| 24 integration cases | Enabled/disabled flags, profile precedence/isolation, API/service metadata, once-per-capture, failures and cleanup | PASS |
| Phase 1 golden snapshots/OpenAPI | Exact default-path backward compatibility | PASS |
| Original 129 tests | Phase 0/1 regression | PASS |
| Full suite | All phases through Phase 2 | PASS — 197 |
| Frontend build | Existing UI/TypeScript/Vite | PASS |
| Phase 0 artifact/API smoke | Real cached model and ten reference/index pairs | PASS — 77 assertions |
| Manual synthetic recipe | Standalone VALID/INVALID quality outputs | PASS |
| Artifact/source checks | No unintended source or data changes | PASS |

Initial 62 new cases passed. Additional integrity/edge tests found one new failure: Pillow PNG verification raises SyntaxError for a bad IDAT checksum, beyond the existing decoder's exception wrapper. Classification: implementation error in new gate exception handling. Fixed only ImageQualityGate to classify it as invalid acquisition; retained the failing assertion and added API integration coverage. Disabled-gate regression explicitly preserves the legacy HTTP 500 on this checksum error. No pre-existing test was modified or weakened.

## 10. Full regression test result

Final focused execution: `python -m pytest -q tests/test_acquisition_contracts.py tests/test_quality_gate.py tests/test_quality_gate_integration.py` in a subprocess with temp DB/storage:

```text
68 passed, 1 warning in 4.72s
```

Final full command after the integrity fix:

```powershell
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py --regression --frontend
```

```text
197 passed, 1 warning in 7.15s
vite v8.0.16: 22 modules; build PASS
77 smoke assertions PASS
Exit code: 0
```

197 = 129 previous tests + 68 Phase 2 cases. No existing test broke, changed, skipped or was removed. The warning remains the baseline FAISS/NumPy deprecation. No dependencies were installed/upgraded and no GPU benchmark was claimed.

## 11. Hypothetical / synthetic cases tested

Images: deterministic 128×96 checker texture with luminance 80/180; fully black; saturated white; Gaussian-blurred checker; 16×16 small texture; corrupt bytes; PNG with altered IDAT checksum; grayscale/nonstandard extension; EXIF orientation; plain uniform image with an explicit blur override.

Metadata/ROI: valid boundary ROI, negative/zero/out-of-bounds/malformed ROI, missing requested fields, wrong camera, blank/type-invalid IDs, naive datetime, valid timezone-aware datetime, negative/NaN/infinite/overflowing encoder values.

Flags/configuration: OFF bypass; global ON; per-fabric ON/OFF override; profile precedence; other fabric isolation; missing/malformed profiles; invalid threshold/flag; gate failure; successful output parity and cleanup before AI/DB.

Synthetic results verify deterministic software behavior, not camera qualification or fabric-detection accuracy. No real-image quality calibration, locked evaluation, candidate activation or hardware acquisition was performed.

## 12. Known limitations

- Internal metadata and successful quality metrics are transient, not DB/public response fields; upload API does not accept the new metadata yet.
- Thresholds need fabric/camera/lighting calibration. A flat or low-contrast Good texture can have low Laplacian variance; use a calibrated override rather than assuming blur=70 universally.
- Metrics measure the full capture. ROI is checked for validity but does not crop image or restrict the measured pixels.
- Gate ON adds decode/integrity/metric work on full-resolution images; latency/memory overhead is not benchmarked here.
- Checks do not detect line dropout, camera-lane seams, encoder synchronization or stream ordering.
- Timestamp timezone checks and nonnegative encoder readings do not establish physical calibration.
- No vendor camera SDK or encoder hardware was integrated. A/B remain INACTIVE; model-quality limitations remain unchanged.

## 13. Risks introduced

An enabled but uncalibrated profile may reject valid low-texture or unusually lit captures. Gate OFF preserves current behavior; configuration errors and execution failures fail closed. Per-fabric overrides take precedence over the global flag, including explicit ON.

The new quality/metadata handoff adds internal types and decode work. Unit/integration/golden regression cover the tested paths; hardware capture behavior remains unverified.

Rollback notes: disable test/global gate settings and any explicit per-fabric enabled override to bypass quality checks; production registry currently has no new override. To revert source, remove only Phase 2 hunks from the four modified files and remove the eight Phase 2 additions. Keep Phase 1 orchestration/contracts, Phase 0 files and unrelated user work. No DB/model/index rollback is required.

## 14. Manual verification steps for me

From repository root, keep the default gate OFF for the full baseline suite:

```powershell
$env:FABRIC_QUALITY_GATE_ENABLED = 'false'
.\.venv\Scripts\python.exe scripts\check_phase0_baseline.py --regression --frontend
```

Expect 197 tests, frontend build and 77 smoke assertions to pass. All ON/OFF scenarios are exercised in temporary integration fixtures.

Standalone quality demonstration without cameras, production DB or reference activation:

```powershell
@'
from pathlib import Path
from tempfile import TemporaryDirectory
from backend.app.ai.quality_gate import ImageQualityGate
from backend.app.core.acquisition import QualityGateConfig
from backend.app.core.inspection_contracts import InspectionInput
from tests.quality_gate_fixtures import write_capture
with TemporaryDirectory(prefix='fabric-phase2-manual-') as directory:
    for kind in ('valid', 'black', 'white', 'blurred'):
        path = write_capture(Path(directory), kind)
        result = ImageQualityGate().evaluate(InspectionInput(path, 'demo_fabric'), QualityGateConfig())
        print(kind, result.status.value, result.reasons)
'@ | .\.venv\Scripts\python.exe -
```

Expected: `valid VALID_CAPTURE`; other three `INVALID_CAPTURE`, with dark/bright/blur reasons.

Optional global enablement for a future test process (not performed on this checkout):

```powershell
$env:FABRIC_QUALITY_GATE_ENABLED = 'true'
$env:FABRIC_QUALITY_GATE_CONFIG_PATH = 'config/quality_gate_example.json'
```

Start/restart that process so cached Settings reflects the environment. Existing A/B still reject inspection with inactive-reference 409; do not activate candidates for this review. Use integration fixtures to verify enabled API behavior. Restore `$env:FABRIC_QUALITY_GATE_ENABLED = 'false'` after experimentation.

Example fabric-record override, shown for test configuration only:

```json
{"quality_gate": {"enabled": true, "min_width": 64, "min_height": 64, "blur_threshold": 70, "check_metadata": false}}
```

Internal metadata call: `InspectionService(db).create_fabric_inspection(upload, fabric_id, metadata=AcquisitionMetadata(...))`. All metadata fields default to None; no additional HTTP form fields are required or supported in this phase.

## 15. Git diff summary

Actual `git diff --stat` is cumulative against HEAD and includes Phase 1 plus the pre-existing executable change:

```text
.env.example                               | 3 additions
backend/app/core/config.py                 | 2 additions
backend/app/services/inspection_service.py | 160 changed lines
tempCodeRunnerFile.exe                     | Bin 44395 -> 44626 bytes
4 files changed, 112 insertions(+), 53 deletions(-)
```

Phase 2 itself changes four existing files and adds eight files listed above. Contract changes and untracked additions are not included in plain tracked diff until staged. Source hashes confirm prior test files, fixtures, algorithms and documents are unchanged. No stage or commit was performed.

## 16. Recommendation

SAFE TO APPROVE — for Phase 2 metadata/Quality Gate scaffolding with default-OFF compatibility. This is not hardware acceptance, production threshold calibration or model release approval.

Await explicit user approval before Phase 3.

```text
============================================================
PHASE 2 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
```
