# Phase 6 — Pattern-Aware Feature / Receptive Field Ablation

Ngày: 2026-10-14, Asia/Bangkok. Chỉ Phase 6; chờ review, chưa bắt đầu Phase 7.

## 1. Goal

Dùng period như một prior cho pooling trên dense feature maps, so sánh fixed baseline với fixed multi-scale/adaptive multi-scale. Không tạo hàng triệu stride-1 image patches. Experimental feature flag mặc định OFF; không đổi production inference hoặc reuse bank cũ với geometry mới.

## 2. What I inspected before coding

- Đọc lại Phase 6 và test/data/Git/report policy trong pasted request; tiếp tục từ Phase 5 đã được người dùng duyệt.
- Nhánh `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`; changes Phase 0–5 và user executable/documents/test_image còn chưa stage/commit, được giữ nguyên.
- Không có AGENTS.md áp dụng trong workspace/ancestors đã kiểm tra. Snapshot 235 pre-existing source/document/tracked files trước Phase 6.
- Baseline SharedDenseResNet18: layer2 native stride 8/channels 128, layer3 stride 16/channels 256; avg_pool 3×3 từng layer, interpolate layer3 về grid layer2, normalize/concat 384 channels, projection adapter v4 → 128.
- Period estimator Phase 3 trả period trong original EXIF-oriented pixels và confidence từng trục; confidence tổng là max hai trục, không đủ để tin cả hai trục.
- Phase 4 memory loader pin identity/config/hashes; Phase 5 hierarchy so sánh encoder/adapter/metric geometry theo provenance. Dimension bằng nhau không chứng minh geometry embedding tương thích.
- Production registry A/B vẫn INACTIVE; cached CPU model/adapter và v5 indexes có sẵn. Full suite trước Phase 6: 486 cases.

## 3. Implementation performed

### Gated dense pooling

`PatternPoolingConfig` frozen, enabled=false, mode period_aware, confidence threshold 0.65, motif_scales [1,2,4], fixed fallback kernels [3,5,7], max odd kernel 31, fusion=mean. Config/CLI phải opt-in rõ; không thêm production Settings/.env flag hay route pooling vào `/api/inspect`.

`PatternAwareDenseExtractor` composition quanh existing SharedDenseResNet18:

- OFF gọi trực tiếp baseline extract_image/extract_path, không gọi period estimator và giữ exact embeddings/boxes/grid/scale. Baseline source không bị sửa.
- ON giữ preprocessing RGB/EXIF/resize, dùng cùng immutable eval-mode model/adapter trong inference_mode để lấy native layer2/3 maps. Không train hay đổi checkpoint/cache.
- Period được quan sát từ ảnh gốc đã định hướng, rồi map về processed pixels bằng actual processed/source ratio riêng x/y, tránh chỉ dùng nominal scale khi rounding resize.
- Kernel mỗi trục/layer ≈ period_original_px × actual_resize × motif_scale / native_layer_stride. Ceil lên integer, tiếp tục lên odd; clamp theo max_kernel và largest odd native map dimension. Kernel tuple là (height/y,width/x).
- Ví dụ period 24 px, resize 1, layer2 stride 8 → 3 cells cho 1× motif. Layer3 dùng stride 16 riêng, không áp kernel layer2 mù quáng lên layer3.
- Chỉ nhận period nếu is_periodic, cả hai periods finite/positive/in-range và confidence tổng + hai trục >= threshold. One-axis stripe, noise/uniform/rotated low confidence, invalid evidence hoặc estimator error dùng fixed multi-scale, không ép adaptive pooling.
- Pool mỗi layer trên native grid, align layer3 về layer2, normalize từng layer/concat. Mean-fuse các scale và normalize để giữ 384 channels; adapter vẫn nhận 384 và xuất 128. Không concatenate scales thành 1152 channels.

PoolingPlan ghi representation, layer2/layer3 kernels và fallback reason. Observed period/result chỉ là research metadata. Fixed mode bỏ qua estimator; period-aware mode có thể chọn kernels khác nhau giữa ảnh, theo cùng policy/geometry.

### Geometry-bound research memory

`FeatureGeometry` schema 1 gồm algorithm version, effective backbone/adapter ids, SHA-256 của actual model/adapter tensor state, pretrained flag, dimension/strides, image_max_size, preprocessing, pooling policy và period-estimator config khi được dùng. Canonical JSON tạo geometry signature. OFF geometry bỏ các config experimental không dùng; ON/fixed/fallback không tự được coi là baseline-compatible.

`encoder_version` của research identity pin backbone + geometry signature. `PatternAwareAnomalyMemory`:

- Build chỉ từ reference feature results cùng geometry và identity đã pin đúng geometry/adapter.
- Search bắt buộc query geometry signature khớp, không chỉ check dimension.
- Save dùng Phase 4 exclusive new-directory/production path guard, thêm `pattern_geometry.json` bên cạnh bốn Phase 4 files.
- Load kiểm tra expected geometry/version/signature rồi dùng strict Phase 4 provenance/hash/index loader. Legacy equal-dimension bank thiếu geometry manifest bị reject; config/weights/preprocessing khác phải dùng bank mới phù hợp.
- Không overwrite, register hay activate artifact. Mọi new banks trong lượt verification này ở TemporaryDirectory.

Typed PatternAware memory API là compatibility boundary cho feature mới. Không gọi raw legacy FAISS scoring với experimental embeddings; low-level numpy arrays tự nó không chứa geometry metadata để old APIs suy ra.

### Diagnostics

`inspect_pattern_pooling.py`: CLI read-only, default feature OFF kể cả config JSON enabled=true; cần `--enable-pattern-aware`. Chỉ dùng cached registered CPU weights, không download. In geometry/period/kernel/embedding shape, không trả QC hoặc ghi artifacts.

`check_phase6_pattern_ablation.py`: tám synthetic inputs × baseline OFF/fixed/adaptive với real cached CNN/adapter; repeatability/dimension checks; baseline-only queries trên v5 A/B; ba geometry-specific synthetic banks save/reload trong temp. Không dùng old bank cho feature đổi geometry, không accuracy/throughput claim.

## 4. Files created

- `backend/app/core/pattern_pooling.py` — config và FeatureGeometry/fingerprint.
- `backend/app/ai/pattern_pooling.py` — plan, native layer pooling, gated extractor wrapper.
- `backend/app/ai/pattern_memory.py` — geometry-bound research FAISS memory.
- `config/pattern_pooling_example.json` — default OFF example.
- `scripts/inspect_pattern_pooling.py` — read-only feature CLI.
- `scripts/check_phase6_pattern_ablation.py` — real-model/synthetic ablation verification.
- `tests/pattern_pooling_fixtures.py` — typed synthetic feature/provenance fixtures.
- `tests/test_pattern_pooling.py` — 51 config/planner/maps/extractor/adapter cases.
- `tests/test_pattern_memory.py` — 18 geometry/memory cases.
- `tests/test_pattern_pooling_integration.py` — 6 existing pipeline/API cases.
- `tests/test_pattern_pooling_cli.py` — 3 flag/help/read-only cases.
- `docs/implementation/phase_6_pooling_report.json` — 24 feature cases, bank parity và new-memory checks.
- `docs/implementation/phase_06_pattern_pooling_ablation.md` — báo cáo này.

## 5. Files modified

Không sửa tệp nào có sẵn trước Phase 6. 235 pre-existing hashes giữ nguyên; không sửa dense extractor/scorer, anomaly service, Phase 3 observer, Phase 4/5 memory modules, tests/fixtures/docs cũ, API/DB/frontend/dependencies hoặc user files.

Chỉ các Phase 6 additions được tạo/cập nhật trong lượt này. Không đổi private .env, weights/indexes/thresholds/registries hoặc locked datasets.

## 6. Architecture before

Production dùng ResNet18 layer2/layer3 với fixed native 3×3 pooling, shared adapter và fabric-specific FAISS/QC. Period observer Phase 3 chỉ metadata/log. Research memory/hierarchy Phase 4/5 không có prior pooling theo period.

## 7. Architecture after

Production giữ nguyên. Research wrapper chọn exact baseline khi OFF; khi ON, period/fixed policy điều khiển multi-scale pooling trên native maps, rồi align/fuse/project thành stable dimension.

Changed feature geometry được pin vào research identity và matching new FAISS memory. Không có đường tự động cập nhật production reference/threshold/QC. Chưa triển khai supervised segmentation/data pipeline Phase 7.

## 8. Backward compatibility

OFF embeddings/boxes exact baseline ở unit, adapter và real cached-model checks. Năm POST/GET QC/OpenAPI golden cases giữ nguyên; integration cố tình khiến new wrapper throw nếu production gọi nó.

ON có thể thay values nhưng không grid/box shape hoặc feature dimension. Dimension-preserving không đồng nghĩa old-memory compatibility; new typed memory rejects baseline/fixed/adaptive geometry mismatch và missing geometry manifest.

V5 A/B scores/boxes khớp baseline-only wrapper comparison. Ba new synthetic research banks (baseline/fixed/adaptive, 128D) reproduce scores sau reload và reject same-dimension wrong geometry. Không activate production reference.

Protected verification:

| Scope | Result |
| --- | --- |
| 55 Phase 0 artifact/config/cache hashes | 55/55 khớp |
| storage: 311 files, gồm production DB/models/indexes/registries | Fingerprint khớp Phase 0 |
| Data: 643 files | Fingerprint khớp Phase 0 |
| User files: 10 files | Fingerprint khớp Phase 0 |
| Cached weights: 3 checkpoint files | Fingerprint khớp Phase 0 |
| 235 pre-existing source/document/tracked files | Tất cả giữ nguyên hash |

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| Planner/maps/extractor/adapter | Period/stride/resize kernels, fallback, shapes, deterministic dimension, eval-mode/no mutation, OFF parity | PASS — 51 cases |
| Geometry-bound memory | FAISS save/reload, same-D mismatch, weights/config/resize/version protection, old-manifest rejection | PASS — 18 cases |
| Pipeline integration | OFF embedding/FAISS parity, five golden QC/API/OpenAPI cases, production isolation | PASS — 6 cases |
| CLI | Explicit flag overrides JSON, no writes, help without model execution | PASS — 3 cases |
| Full regression | 486 prior cases + 78 new cases | PASS — 564 cases |
| Cached real CNN/adapter ablation | 8 inputs × 3 representations, stable 128D and repeatable outputs | PASS — 24 cases |
| V5 bank parity | Baseline-only features, A/B scores/boxes | PASS — 2 fabrics |
| New research banks | Typed baseline/fixed/adaptive memories; exact reload and mismatch rejection | PASS — 3 banks |
| Real CLI ON/OFF | Existing images.jpeg, read-only, shape 192×128 | PASS |
| Baseline smoke/frontend | Existing artifact/API behavior + TypeScript/Vite | PASS — 77 assertions, 22 modules |
| Syntax/JSON/Git/hashes | New source/report validity, protected scopes | PASS |

## 10. Full regression test result

Lệnh thực tế từ workspace root:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Kết quả cuối: `564 passed, 1 warning in 29.00s`; frontend `tsc -b && vite build` PASS; smoke PASS 77 assertions. Warning FAISS/NumPy private namespace đã có từ baseline. Không cài/nâng cấp package hay sửa pre-existing tests.

Targeted ban đầu: `65 passed, 1 warning in 6.55s`; integration riêng: `6 passed, 1 warning in 4.07s`. Bảy defensive/adapter/CLI cases bổ sung được chạy trong final full suite. Không có test failure cần weakening/chỉnh legacy expectations.

Real ablation command:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase6_pattern_ablation.py
```

Output snapshot ở `docs/implementation/phase_6_pooling_report.json`. Script yêu cầu cached model, không import/init production DB và chỉ tạo memory mới trong temp.

## 11. Hypothetical / synthetic cases tested

Tám images: checker, repeated dots, one-axis stripes, uniform, noise, stretched checker, mild illumination, rotated checker. Mỗi ảnh chạy baseline OFF, fixed multi-scale và period-aware; không training, không pixel-defect dataset fabrication.

Checker 320×288 có source period x=16/y=24 px. Với image_max_size=256, actual resize x=0.8/y=230/288, real plan:

| Layer | 1× motif (h,w) | 2× | 4× |
| --- | --- | --- | --- |
| layer2, stride 8 | (3,3) | (5,5) | (11,7) |
| layer3, stride 16 | (3,1) | (3,3) | (5,5) |

Shape checker của mọi branch là 928×128. Uniform/noise/rotated và one-axis stripe fallback fixed multi-scale. Stretched/illumination fixtures kiểm tra bounded kernels và representation behavior; không chứng minh arbitrary distortion handling.

Thêm tiny/odd layer grids, mismatched batch/dtype/rank, nonfinite maps/period, boolean/invalid config, estimator failure, adapter 384→128, weight/resize/version/geometry mismatches và corrupt signature.

Real CPU weights/adapter v4 được dùng, nhưng reference/query trong ablation là synthetic/self-retrieval. Existing images.jpeg chỉ CLI smoke. Chưa có independent real evaluation so fixed vs adaptive, không có accuracy gain, production QC acceptance hoặc throughput result.

## 12. Known limitations

- Kernel là feature-grid pooling prior, không phải full CNN receptive field bằng đúng motif size. Convolutional features đã có receptive field trước pooling; ceil-to-odd/clipping chỉ xấp xỉ motif footprint.
- Period confidence chưa calibrated; alias/harmonics/cancellation/rotation/stretch có thể sai. Cần cả hai axes đủ confidence; stripe một trục fallback có chủ đích thay vì đoán axis còn lại.
- Max size và native-grid clipping có thể làm các scales trùng kernel, đặc biệt ảnh nhỏ. Mean fusion là một lựa chọn research, chưa được chứng minh tối ưu.
- Adapter cũ nhận đúng dimension nhưng được train trên baseline geometry; pooling ON có thể làm embedding ít phù hợp. Không retrain adapter trong phase này.
- Giữ dimension giúp shape compatibility, không bảo đảm embedding comparability với old FAISS/thresholds. Cần matching fresh research bank và calibration/evaluation riêng trước production.
- Geometry pin config/weights/preprocessing policy, không pin từng dynamic kernel. Adaptive policy cho phép image-specific period/kernel/fallback khác nhau giữa reference và query; ảnh nguồn/hashes/config cần giữ để tái lập quan sát.
- Model phải immutable eval mode; không hỗ trợ concurrent training/mutation trong research extraction. SHA fingerprint tensor state không chứng minh approved training data hay source thật sự Good; caller vẫn phải cung cấp provenance đúng.
- Không nối wrapper vào `/api/inspect`, không xuất geometry/period HTTP fields. Phase 3 production observer vẫn observe-only, Phase 5 hierarchy không tự chuyển sang pooled features.

## 13. Risks introduced

Production risk thấp vì zero existing-file edits và không dùng new wrapper trong runtime. Research ON có thể làm mất small defects hoặc đổi anomaly-score distribution; không reuse threshold gốc dù dim=128.

Pooling nhiều scale, stack/fusion, estimator và tensor-state fingerprints tăng CPU/RAM. Fingerprinting cũng có overhead trên research OFF wrapper; production không gọi wrapper này. Chưa benchmark line-speed/peak memory hay optimize caching.

Geometry protection áp dụng qua new typed memory API. Nếu caller tự bóc numpy embeddings rồi dùng raw legacy FAISS APIs, old API không thể suy ra geometry; đó là misuse ngoài contract, không được coi là compatible deployment.

Save giữ Phase 4 fresh-directory guard; IO failure có thể để lại incomplete research directory, thiếu geometry manifest thì typed loader reject. Không tự delete artifact để tránh mất dữ liệu phục hồi; không publish/activate nó.

Rollback: bỏ riêng 13 Phase 6 additions, giữ nguyên Phase 0–5/user work. Không cần rollback DB/model/index/threshold vì không đổi; không dùng reset/clean hoặc destructive broad delete.

## 14. Manual verification steps for me

1. Chạy full runner và real ablation command ở mục 10. A/B thật vẫn INACTIVE; không activate candidates để review.
2. Đọc `docs/implementation/phase_6_pooling_report.json` để xem 24 cases, exact OFF parity, selected kernels và geometry-bound bank checks.
3. So CLI OFF/ON read-only trên ảnh có sẵn:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_pattern_pooling.py test_image/images.jpeg --image-max-size 128
.\.venv\Scripts\python.exe scripts/inspect_pattern_pooling.py test_image/images.jpeg --image-max-size 128 --enable-pattern-aware
```

Đã kiểm tra: hai outputs đều shape [192,128]; OFF baseline, ON fixed_multiscale fallback cho ảnh này; qc_performed=false, writes_performed=false. Geometry signatures khác nhau là đúng, không phải bug.

4. Config ví dụ ở `config/pattern_pooling_example.json` default enabled=false. Muốn fixed multi-scale, chuẩn bị research config mode=fixed_multiscale và truyền --config cùng explicit flag. CLI không tự bật theo JSON enabled=true.

5. API research: tạo `PatternAwareDenseExtractor(baseline, config=..., adapter_version=...)`; lấy PatternFeatureResult cho approved reference/query. Build `PatternAwareAnomalyMemory` với MemoryIdentity.encoder_version=result.geometry.encoder_version và matching adapter_version, enabled MemoryExperimentConfig; save_new vào directory mới. Search nhận typed query result, không raw changed embeddings + old bank.

Không sửa private .env, shared extractor cache, production registry hoặc old memory để thử nghiệm. Low-confidence ON fallback là fixed multi-scale experimental, không baseline; vẫn cần matching geometry bank.

## 15. Git diff summary

Phase 6 riêng: 13 new files, 0 pre-existing modifications; 235 pre-existing hashes giữ nguyên. Không stage/commit.

Actual tracked `git diff --stat` cumulative against HEAD vẫn gồm các phase trước và executable user:

```text
.env.example                               |   6 +
backend/app/core/config.py                 |   4 +
backend/app/services/fabric_registry.py    |   2 +
backend/app/services/inspection_service.py | 216 changed lines
tempCodeRunnerFile.exe                     | Bin 44395 -> 44626 bytes
5 files changed, 175 insertions(+), 53 deletions(-)
```

Phase 6 additions còn untracked nên chưa có trong plain tracked diff. `git diff --check` không có whitespace errors, chỉ LF/CRLF warnings từ existing dirty files. Staged diff trống; không overwrite/delete/activate production artifacts hoặc mutate datasets.

## 16. Recommendation

SAFE TO APPROVE — cho Phase 6 gated pooling ablation và geometry-safe experimental FAISS infrastructure. Chưa phê duyệt pooled features/adapter/thresholds cho production và chưa xác nhận accuracy improvement.

Dừng để review; chưa bắt đầu Phase 7.

```text
============================================================
PHASE 6 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
```
