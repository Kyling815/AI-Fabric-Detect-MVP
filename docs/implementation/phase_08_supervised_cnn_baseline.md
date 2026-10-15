# Phase 8 — Supervised CNN segmentation baseline

Ngày kiểm tra: 2026-10-14, Asia/Bangkok. Chỉ Phase 8; chưa bắt đầu Phase 9 shadow inference.

## 1. Goal

Triển khai CNN segmentation baseline và infrastructure huấn luyện/validation cho known defects: U-Net-like decoder + ResNet18 pretrained encoder độc lập, CE+Dice, Focal tùy chọn, AdamW, early stopping, AMP khi có CUDA và checkpoint save/load.

Do chưa có dataset pixel labels thật/provenance phù hợp, chỉ train tiny synthetic fixtures trong thư mục tạm. Không pretend đã train meaningful production model, không tích hợp `/api/inspect`, không activate registry.

## 2. What I inspected before coding

- Yêu cầu Phase 8 trong attachment người dùng; báo cáo và các contracts/dataset/split/metrics Phase 7; runtime inspection/anomaly, legacy adapter trainer, cached ResNet18 và fixtures API Phase 1.
- Branch `main`, HEAD `530a30b8ee6a2fb05d4df0b14e76c68e10cf3572`. Worktree dirty từ các phase trước và file người dùng; không có AGENTS.md áp dụng trong repo/các thư mục cha đã kiểm tra. Không stage/commit hay ghi đè các thay đổi đó.
- Backend FastAPI/PyTorch; frontend React/TypeScript/Vite. Runtime hiện tại vẫn CNN dense features + FAISS anomaly + rule/QC, không phải non-Deep-Learning.
- Trước Phase 8: 651 tests. Môi trường Torch 2.5.1+cpu, torchvision 0.20.1, không có CUDA. Đọc installed source để kiểm tra torch.amp/weights_only tương thích; không upgrade dependency.
- Company audit chạy lại: A 121 rows (61 real/60 synthetic), B 129 rows (69 real/60 synthetic). Toàn bộ 120 masked rows được khai báo synthetic; runtime manifests thiếu roll_id/lot_id/camera_id xác thực. Bốn adapter_train pairs spot-check hợp lệ nhưng không đủ chứng minh dataset training readiness. Không đọc/evaluate locked pixels.
- Registered anomaly vẫn ResNet18 layer2+3 + adapter v4, 128 dim, stride 8. Hai fabric production vẫn INACTIVE, active=None; không thay encoder hoặc FAISS.

## 3. Implementation performed

1. `SegmentationBaselineConfig` sử dụng lại pipeline contracts Phase 7. Example default OFF; synthetic OFF; supervised/anomaly weight sharing bị từ chối. Các rates/weights là starting values, không được gọi optimal.
2. `ResNet18UNet`: khởi tạo instance ResNet18 mới, không gọi cached shared anomaly factory. Encoder full layer1–4, skip connections từ stem/layer1/layer2/layer3; năm decoder blocks với bilinear upsample, convolution và GroupNorm. Head trả logits BCHW đúng input height/width, kể cả odd geometry.
3. Pretrained dùng IMAGENET1K_V1 trong trusted local cache; kiểm tra SHA prefix filename và lưu full SHA. Không download hoặc silently fallback random. Random init chỉ khi config explicit false và report đánh dấu/warn rõ.
4. Dataset Phase 7 trả RGB float [0,1]. Normalization ImageNet mean/std nằm trong model, chỉ normalize, không dùng classification center-crop làm mất alignment mask. Normalization buffers/contract được kiểm khi reload.
5. Encoder BatchNorm running stats mặc định frozen cho tiny batches; affine parameters và convolution vẫn train. Decoder GroupNorm cho phép batch=1. BN freeze không đồng nghĩa freeze toàn bộ encoder.
6. Forward/predict trả class probabilities, class-ID mask và confidence map per pixel (winning-class probability). Confidence chưa calibrated, không phải image-level/QC confidence. Predict/evaluation giữ deterministic eval và phục hồi model mode.
7. Loss FP32 CE+soft multiclass Dice, active classes gồm background; ignore pixels được loại khỏi cả hai thành phần. Focal tùy chọn chỉ modulate CE; gamma=0 tương đương CE. Invalid/nonfinite/no-valid targets bị chặn.
8. AdamW parameter groups encoder/decoder-head disjoint với LR riêng; gradient clipping, finite loss/gradient guards. AMP chỉ enabled nếu device CUDA và config cho phép; CPU FP32 fallback. Explicit CUDA request khi không có CUDA gây lỗi, không âm thầm đổi ý người dùng.
9. Trainer dùng grouped splits Phase 7; require_roll_id không được hạ xuống source-only. Synthetic cần cả config allow_synthetic=true và caller/CLI opt-in. Real train/validation annotations cần khai báo human_verified; metadata declaration không thay thế external label/provenance audit.
10. Preflight load/check train/validation pairs, class support và validation foreground trước khi tạo artifact directory. Không load test/historical_locked/excluded pixels. Validation chọn best loss/early stopping, không phải fresh acceptance test. Split ratios gần đúng vì source groups không chia nhỏ.
11. New version directory bắt buộc chưa tồn tại. Bảo vệ Data/test_image/source/cache và production storage; trong storage chỉ cho phép new segmentation_candidates path. Ghi config/manifest snapshot/split/report và mỗi improvement thành `best_epoch_XXXX.pt` bằng exclusive create, không replace best/current checkpoint.
12. Checkpoint có model version, architecture/preprocessing/taxonomy, initialization/hash, config, manifest fingerprint, epoch/loss, synthetic flag và production_approved=false; lưu model/optimizer/scaler states. Reload strict shapes/keys/taxonomy/order/normalization/provenance, weights_only và trusted_local explicit. Không có resume-training CLI.
13. Training context restore RNG/thread/backend flags; single-worker seeded DataLoader, train-only augmentation. Không implement API/shadow inference/fusion của Phase 9–10.

## 4. Files created

13 files mới:

- `backend/app/core/segmentation_training.py`
- `backend/app/ai/segmentation_model.py`
- `backend/app/ai/segmentation_loss.py`
- `backend/app/ai/segmentation_trainer.py`
- `config/segmentation_baseline_example.json`
- `scripts/train_segmentation_baseline.py`
- `tests/segmentation_training_fixtures.py`
- `tests/test_segmentation_model.py`
- `tests/test_segmentation_loss.py`
- `tests/test_segmentation_trainer.py`
- `tests/test_segmentation_training_integration.py`
- `docs/implementation/phase_8_baseline_report.json`
- `docs/implementation/phase_08_supervised_cnn_baseline.md`

## 5. Files modified

Không sửa file có sẵn trong Phase 8, kể cả Phase 7 contracts/config/tests. 267 pre-existing tracked/untracked file fingerprints không đổi. Không sửa Revision.md, research report, file executable/ảnh test của người dùng hoặc production artifacts.

## 6. Architecture before

Production: upload → InspectionService → quality/pattern/anomaly/QC pipeline Phase 1–6 → response/persistence.

Offline Phase 7: manifest → grouped split → verified image/mask loader → paired transforms → Dataset/DataLoader; standalone pixel metrics. Training configuration chỉ là specification, chưa có CNN/trainer.

## 7. Architecture after

Production giữ nguyên. Nhánh offline mới:

`Phase 7 Dataset → RGB normalization → independent pretrained ResNet18 → U-Net-like decoder → logits`

`logits + target mask → CE(+optional Focal) + Dice → backward → AdamW encoder/decoder groups`

`independent validation → loss/IoU/Dice → early stopping → exclusive versioned candidate checkpoint`

`trusted checkpoint reload → probabilities + class mask + pixel confidence`

Không share trainable weights, không invalidation FAISS embedding geometry, không shadow logging/QC fusion.

## 8. Backward compatibility

- Năm golden QC cases POST/GET và OpenAPI response contracts giữ nguyên. Integration tests làm supervised forward/trainer ném lỗi nếu bị production gọi.
- Full regression sử dụng isolated temporary DB/storage; real cached model/reference artifacts chỉ đọc. Registry inactive không bị chuyển trạng thái.
- 55 protected artifact hashes và protected scope hashes giữ nguyên: storage 311 files, Data 643, user_files 10, torch checkpoints 3.
- Tất cả actual training/checkpoint fixtures chạy trong temp locations, không để lại production model/candidate/index. Không thay dependency, API, frontend, thresholds, SQLite, storage layout hoặc cached pretrained weights.

## 9. Test cases executed

| Test | Purpose | Result |
| --- | --- | --- |
| test_segmentation_model.py, 13 cases | Forward shape, probability/mask/confidence, odd dimensions, backward both branches, overfit, frozen BN, encoder independence, bad inputs/cache | PASS |
| test_segmentation_loss.py, 16 cases | CE parity, finite gradients, optional Focal, ignore pixels, correct/wrong predictions, invalid config/targets | PASS |
| test_segmentation_trainer.py, 34 cases | LR groups, strict checkpoint/provenance/normalization, no overwrite, preflight gates, validation metrics, early stop, no test pixel access, RNG restore | PASS |
| test_segmentation_training_integration.py, 11 cases | Five golden API cases, CLI temporary pretrained demo, disabled defaults/two-key synthetic opt-in, reproducible training, Focal, nonfinite failure | PASS |
| Existing 651 tests | Regression Phase 0–7 | PASS |
| Frontend build | TypeScript/Vite compatibility | PASS, 22 modules |
| Artifact/API smoke | Existing real encoder/index/reference plus isolated synthetic API decisions | PASS, 77 assertions |
| compileall / git diff --check | Python syntax / tracked whitespace | PASS |
| Source/artifact/scope hashes | Preserve existing work and production data/artifacts | Không đổi |

Failure đã xử lý trong phase: one-batch overfit ban đầu dùng equal LR 0.003 cho toàn bộ random-initialized model, 100 steps, chưa đạt loss-reduction assertion (0.3884 so với initial 1.4403). Đây là fixture training recipe chưa hội tụ, không phải legacy/runtime regression. Chuyển sang encoder LR 1e-4/decoder LR 0.01, 200 steps chỉ cho tiny overfit diagnostic. Giữ nguyên assertions loss giảm >90% và foreground IoU >0.95; không sửa/giảm tiêu chuẩn tests cũ, không gọi các LR này optimal cho production.

## 10. Full regression test result

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Full regression cuối sau checkpoint model_version và manifest snapshot: **725 passed, 1 warning in 76.30s**, frontend build PASS, artifact/API smoke PASS (77 assertions), exit 0. Warning FAISS/NumPy private namespace là warning hiện hữu, không phải regression mới.

Targeted Phase 8: **74 passed, 1 warning in 52.16s**. Không có pre-existing test nào bị sửa hoặc bị giảm assertion.

## 11. Hypothetical / synthetic cases tested

- Actual pretrained demo train hai epochs trên chín tiny synthetic image/mask pairs, ba independent source/roll/lot/camera groups; train/validation/test count 3/3/3. Test partition không được load/evaluate. Training batch size 3, resolution 32×32, CPU FP32, tiny decoder widths [16,8,8,4,4].
- Demo initial train loss 1.496736, epoch-2 train loss 1.461325; selected validation loss 1.435362, foreground IoU 0.041906 và Dice 0.080441. Đây là synthetic software exercise còn chất lượng rất thấp, không phải production accuracy hoặc mô hình đã được duyệt.
- Prediction probability shape [1,2,32,32], mask/confidence [1,32,32]; repeated evaluation deterministic. Checkpoint save/reload probabilities/mask/confidence exact equality trong CPU fixture tests.
- Separate one-batch overfit diagnostic: hai tiny images, random init explicit, 200 optimization steps. Loss giảm hơn 90%, foreground IoU 1.0 trên chính train batch; không phải held-out generalization.
- Mock validation plateau chỉ kiểm tra early-stop control flow; mocked numbers không phải trained-model metrics. NaN training loss không tạo checkpoint/report approved. Test corrupted metadata, weights, normalization, IDs, checksum và unsafe output paths fail closed.

## 12. Known limitations

- Production readiness vẫn `UNVERIFIED_REAL_PIXEL_LABELS_AND_CAPTURE_GROUP_METADATA`. Không train Company real data, không suy ra real semantic labels từ masks synthetic/tên lỗi legacy, không evaluate historical locked pixels. Không có meaningful production segmentation model được tạo.
- CUDA/AMP training được implement nhưng chưa chạy thực vì môi trường CPU-only. Không benchmark latency/throughput, distributed/multiworker training hoặc camera SDK; không đưa ra GPU claims.
- Grouping dựa vào provenance/hashes khai báo, không phát hiện mọi near-duplicate hoặc lineage/roll/camera sai. Validation class imbalance/thiếu một số classes được cảnh báo, không được coi chất lượng per-class đã xác minh.
- Baseline chưa tuning class weights, sampling, image resolution hoặc augmentation theo weave. Resize có thể làm mất lỗi nhỏ; freeze encoder BN running stats mặc định cần ablation thực về sau.
- Lưu optimizer/scaler để inspection/extension, nhưng chưa có exact training resume (không lưu tất cả RNG/DataLoader/early-stop control states). Interrupted run có thể để lại partial candidate directory; không auto-delete hoặc overwrite khi chạy lại. Dùng version directory mới.
- Phiên bản Torch hiện hữu có security advisory cho untrusted checkpoint deserialization; trusted-local gate và weights_only không phải sandbox/security boundary. Không mở checkpoint không rõ nguồn; chưa tự upgrade môi trường. Xem [PyTorch advisory CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6).

## 13. Risks introduced

- Pixel confidence là uncalibrated softmax, không dùng làm QC threshold/GOOD decision. Segmentation branch chưa nằm trong runtime nên không ảnh hưởng quyết định hiện hữu.
- Taxonomy order/preprocessing/architecture version phải nhất quán khi reload; checkpoint loader từ chối mismatch và production_approved=true. Đây chưa phải release governance hoặc activation mechanism Phase 18.
- Real manifest human_verified và capture IDs vẫn là declaration; cần người dùng xác thực bằng audit thu nhận/annotation độc lập. Template training.enabled=false và hai-key synthetic gate giảm accidental training, không thay thế audit.
- CPU full ResNet18 train có thể chậm và tốn RAM; repeated integrity checks tăng I/O. New checkpoint mỗi improvement tăng disk usage nhưng bảo toàn artifacts cũ.
- PyTorch dependency hardening cần task riêng được review/regression trước khi dùng artifacts bên ngoài. Nguồn technical: [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html), [checkpoint loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html). Installed 2.5.1/0.20.1 source và tests mới là chứng cứ compatibility của checkout này.

## 14. Manual verification steps for me

Chạy tại project root:

```powershell
.\.venv\Scripts\python.exe scripts/train_segmentation_baseline.py --demo
.\.venv\Scripts\python.exe -m pytest -q tests/test_segmentation_model.py tests/test_segmentation_loss.py tests/test_segmentation_trainer.py tests/test_segmentation_training_integration.py
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
git diff --check
git status --short
```

Demo báo trained_model=true **chỉ đối với tiny synthetic training**, production_approved=false, production_accuracy_evaluated=false, test_partition_evaluated=false, artifacts_persisted=false. Không có checkpoint dùng production sau demo.

Chỉ khi đã có manifest thật được xác minh, tạo config riêng từ example và set pipeline.training.enabled=true; giữ pretrained_encoder=true, share_anomaly_encoder=false, allow_synthetic=false, require_roll_id=true. Dùng explicit new version ID, ví dụ:

```powershell
.\.venv\Scripts\python.exe scripts/train_segmentation_baseline.py --manifest "C:\path\to\verified_manifest.json" --config "C:\path\to\approved_training_config.json" --output-dir storage/models/segmentation_candidates/supervised_candidate_v001
```

Đây là hướng dẫn conditional, chưa chạy lệnh này với Company data. Không thêm --allow-synthetic cho dữ liệu production. Không tải/open checkpoints lạ; cached pretrained thiếu sẽ fail, không download tự động.

Rollback: runtime không dùng modules mới, không cần restore registry/model/FAISS. Sau review có thể chuyển đúng 13 file mới ở mục 4 sang thư mục backup riêng. Không reset/clean toàn worktree hoặc xóa training output/data của người dùng. Chưa thực hiện rollback/xóa dữ liệu nào.

## 15. Git diff summary

Phase 8: **13 files mới, 0 pre-existing files modified**, chưa stage/commit. All 267 baseline source/user tracked+untracked files giữ nguyên hashes.

Cumulative `git diff --stat` vẫn chỉ 5 tracked files, 175 insertions/53 deletions có trước phase: `.env.example`, core/config.py, services/fabric_registry.py, services/inspection_service.py, và user binary tempCodeRunnerFile.exe. Không quy các changes đó cho Phase 8. Staged diff rỗng.

## 16. Recommendation

**SAFE TO APPROVE** cho Phase 8 infrastructure trong phạm vi trusted local artifacts và synthetic software verification. **Không approve production model/training readiness**; cần pixel labels thật, verified acquisition groups, appropriate validation và fresh acceptance holdout trước production claims. Security/dependency upgrade cần review riêng, không tự thực hiện trong phase này. Chờ người dùng review trước Phase 9.

============================================================
PHASE 8 IMPLEMENTATION COMPLETE
STATUS: WAITING FOR USER REVIEW
NEXT PHASE HAS NOT BEEN STARTED
============================================================
