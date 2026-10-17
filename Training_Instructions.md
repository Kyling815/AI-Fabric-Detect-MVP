# Hướng dẫn chuẩn bị dataset và train model cho hai fabric

Ngày soạn: 2026-10-17 — Asia/Bangkok.

**Phạm vi: tài liệu hướng dẫn, không phải lệnh bắt đầu training.** Project vẫn ở DATA PREPARATION. Chỉ bắt đầu Training Phase 1 sau khi có reviewed real annotation set và được người dùng phê duyệt rõ ràng. Training model thuộc phase tiếp theo, cần approval riêng; việc tạo tài liệu này không cấp approval đó.

## 1. Chọn đúng bài toán trước khi chuẩn bị dữ liệu

Project đã có ResNet18 encoder + U-Net-like decoder để **segmentation**: xác định pixel/vùng lỗi, không chỉ phân loại ảnh GOOD/BAD.

Hai fabric là hai loại vật liệu, **không phải** hai defect classes. Baseline đơn giản có thể dùng một model chung cho hai fabric, taxonomy pixel `background` và `defect`; manifest vẫn phải giữ `fabric_id` để kiểm tra coverage và đánh giá riêng từng fabric. Đây là phương án chuẩn bị để review, chưa phải quyết định training đã được duyệt.

| Dữ liệu thực có | Có thể làm gì? | Chưa được kết luận gì? |
|---|---|---|
| Chỉ có GOOD của cả hai fabric | Review Good, thu thập provenance; có thể đề xuất thí nghiệm anomaly/reference riêng | Không đủ để train supervised segmentation defect |
| GOOD + BAD, chỉ nhãn mức ảnh | Review ảnh và annotate pixel masks | Folder BAD không tự trở thành segmentation ground truth |
| GOOD + BAD + bbox | Bbox hỗ trợ người annotate xác định vị trí | Không được tô kín bbox rồi gọi là pixel mask thật |
| GOOD + BAD + reviewed pixel masks | Có thể đề nghị duyệt xây dựng dataset train/validation | Chưa tự động đủ coverage, chất lượng hay production readiness |
| Chỉ một fabric có BAD/masks | Có thể chuẩn bị annotation của fabric đó và thu thập thêm cho fabric còn lại | Không được suy ra khả năng phát hiện lỗi trên cả hai fabric |

Nếu thiếu BAD thật, **để trống nhóm BAD và báo thiếu dữ liệu**. Không tạo defect/mask giả, không lấy anomaly output làm ground truth, không dùng ảnh synthetic để giả lập nhãn thật còn thiếu. Một dataset GOOD/BAD cũng có thể dùng cho image classification trong một thí nghiệm khác, nhưng đó không phải segmentation trainer hiện tại.

## 2. Quy ước fabric và ảnh

- Dùng `fabric_A`, `fabric_B` trong tài liệu như **tên minh họa**. Thay bằng ID ổn định đã được chủ dữ liệu xác nhận.
- Chỉ dùng `fabric_001`/`fabric_002` nếu xác minh đúng cùng loại fabric đã có; không tái sử dụng ID cũ cho vật liệu khác. Không sửa registry để tạo mapping trong bước chuẩn bị.
- Mỗi ảnh là một `sample_id`; mỗi capture gốc là một `source_id` duy nhất. Crop/biến thể của cùng capture phải giữ cùng ancestry, không trở thành nguồn độc lập.
- Có thể cấp ID quản lý nội bộ mới, nhưng ID đó không chứng minh roll/lot/camera hoặc thời gian chụp. Không suy provenance từ ID/tên file.
- Giữ ảnh gốc, không ghi đè hoặc xóa EXIF trước khi ghi nhận orientation và hash nguồn.
- Tên minh họa: `fabric_A_capture_000001.png`, mask tương ứng `fabric_A_capture_000001_mask.png`. Tên file không quyết định nhãn; quyết định phải đến từ review.

## 3. Cấu trúc folder dataset đề xuất

Đây là **cấu trúc sẽ chuẩn bị khi được duyệt**, chưa được tạo bởi tài liệu này. Không di chuyển/chỉnh sửa bộ dữ liệu cũ để ép nó theo cấu trúc mới.

```text
Data/
├── training/
│   └── two_fabrics_real_v1/
│       ├── README.md
│       ├── incoming/                         # Ảnh nhận về, chưa xác minh
│       │   └── real/
│       │       ├── fabric_A/
│       │       │   ├── GOOD/                 # Good theo khai báo nguồn
│       │       │   ├── BAD/                  # Để trống nếu chưa có
│       │       │   └── UNREVIEWED/           # Chưa xác định nhãn/fabric
│       │       └── fabric_B/
│       │           ├── GOOD/
│       │           ├── BAD/
│       │           └── UNREVIEWED/
│       ├── reviewed/                         # Chỉ chứa mẫu đã review
│       │   └── real/
│       │       ├── fabric_A/
│       │       │   ├── images/
│       │       │   │   ├── GOOD/
│       │       │   │   └── BAD/
│       │       │   └── masks/
│       │       │       ├── GOOD/             # Empty masks sau xác nhận Good
│       │       │       └── BAD/              # Human pixel annotations
│       │       └── fabric_B/
│       │           ├── images/
│       │           │   ├── GOOD/
│       │           │   └── BAD/
│       │           └── masks/
│       │               ├── GOOD/
│       │               └── BAD/
│       ├── annotations/
│       │   ├── provenance.csv
│       │   ├── image_review.csv
│       │   ├── annotation_review.csv
│       │   └── label_schema.json
│       ├── manifests/                        # Chỉ tạo trong Phase 1 đã duyệt
│       │   ├── all_reviewed.json
│       │   ├── train_val.json
│       │   ├── split_review.csv
│       │   └── excluded.csv
│       └── previews/
│           └── annotation_review/             # Overlay để người kiểm tra
└── synthetic/                                # Tách riêng, nếu có thí nghiệm
    └── ...                                   # Không nhập vào real manifest
```

Không cần copy vật lý ảnh vào `train/` và `val/`: trainer hiện tại đọc đường dẫn và split từ JSON manifest, không dùng ImageFolder để tự đọc thư mục GOOD/BAD. Một nguồn ảnh không được copy sang nhiều split như các mẫu độc lập.

`incoming/GOOD` và `incoming/BAD` là nhãn khai báo, **không phải human_verified**. Chỉ đưa vào `reviewed/` sau khi hoàn tất review. Giữ bản gốc và checksum; không tự chuyển file hoặc làm mất source lineage.

Không đặt future locked test trong dataset phát triển này. Historical/postmortem giữ nguyên vai trò cũ, không tái sử dụng thành fresh locked test. Locked test tương lai cần capture mới, custody riêng và source/lot/roll/camera-disjoint theo governance.

## 4. Review ảnh thật trước khi annotate

### 4.1. Với từng ảnh BAD

Người hiểu vật liệu cần xác nhận và ghi vào `image_review.csv`:

1. Thực sự có lỗi hay chỉ là ánh sáng, bóng, nhăn nền/chụp, mép vải hoặc texture bình thường?
2. Fabric nào? Có đủ căn cứ xác định `fabric_id` không?
3. Defect class thực tế, một hay nhiều loại lỗi; đánh dấu vùng mơ hồ để trao đổi, không tự đoán.
4. Orientation/EXIF, độ nét, exposure, scale và mức nhìn thấy lỗi.
5. Có phù hợp annotate segmentation không? Nếu không rõ ranh giới hoặc không có đủ độ phân giải, đưa vào review/excluded với lý do thay vì giả nhãn.
6. Người review, thời điểm review, kết luận và ghi chú.

Nếu dùng 14 ảnh NG cũ, vẫn phải review đầy đủ và ghi lineage/historical status. Không mặc định chúng thuộc hai fabric mới hoặc được phép dùng cho train; không suy class từ tên IQC_Stain/IQC_Spot…

### 4.2. Với từng ảnh GOOD

- Người review phải xác nhận ảnh không có defect trong vùng được dùng, không chỉ tin folder GOOD.
- Ảnh nghi ngờ chuyển UNREVIEWED; không ép nhãn GOOD để tăng số lượng.
- Sau xác nhận, lưu empty mask và review record tương ứng. Mask toàn 0 thể hiện **kết luận Good đã được review**, không phải cách tự gán Good cho ảnh chưa biết.
- Loader hiện cho phép Good không có mask và tạo zero mask trong bộ nhớ; đây là tiện ích kỹ thuật, **không phải bằng chứng human defect-free**. Với dataset mới, hướng dẫn này yêu cầu mask/review record rõ ràng để kiểm toán được.

## 5. Quy chuẩn pixel masks

### 5.1. Binary segmentation baseline

Chỉ dùng khi thực sự có nhãn vùng lỗi thật:

```json
{"classes": ["background", "defect"], "ignore_index": null}
```

- `background`: pixel không lỗi trong vùng được annotate, class ID `0`.
- `defect`: pixel lỗi đã được người annotate/review, class ID `1`.
- GOOD: mọi pixel hợp lệ là `0` sau review.
- BAD: phải có foreground thật; không chấp nhận empty mask.
- Nhiều lỗi trong một ảnh: annotate tất cả vùng đã xác minh. Với binary baseline, các vùng cùng mang class `defect`; loại lỗi chi tiết lưu trong sidecar review, không tự thêm class mới vào trainer.

### 5.2. Encoding và geometry mà code hiện tại hỗ trợ

| Encoding manifest | Giá trị mask | Cấu hình cần có |
|---|---|---|
| `class_ids` | `0` background, `1` defect cho binary baseline | `binary_class_name: null` |
| `binary_255` | `0` background, `255` defect | `binary_class_name: "defect"` |

Hai cách không được khai báo lẫn nhau. File 0/255 nhưng ghi `class_ids` sẽ không hợp lệ với taxonomy hai class. Không dùng ảnh overlay RGB làm label mask.

- Lưu mask lossless PNG single-channel/integer; không dùng JPEG cho mask.
- Mask phải đúng kích thước và hệ tọa độ của ảnh. Không resize mask sai geometry để che lỗi annotation.
- Nên review/annotate trên ảnh đã hiển thị đúng orientation. Nếu khai báo `mask_coordinates: "oriented_image"`, kích thước phải khớp ảnh sau EXIF orientation.
- Nếu mask thực sự theo raw pixels thì khai báo `raw_image`; loader có xử lý orientation tương ứng. Không xoay riêng ảnh hoặc mask mà không cập nhật quan hệ và hash.
- Mask không mang EXIF transform riêng. Tốt nhất lưu bản ảnh annotation đã chuẩn hóa orientation như một derivative có lineage, không ghi đè nguồn.
- Khi resize phục vụ training, mask dùng nearest-neighbor; ảnh dùng biến đổi tương ứng. Code hiện có paired transforms, nhưng điều đó không sửa được nhãn gốc sai.
- Nếu annotation có vùng không biết, đừng mặc định nó là background. Cần quy tắc ignore được review, taxonomy/config/metrics thống nhất; chưa quyết định thì cách ly mẫu đó.
- Không tạo mask từ filename, bbox-fill, anomaly heatmap, predicted mask hoặc synthetic generator rồi đánh dấu `human_verified`.

## 6. Provenance và review records

Mỗi mẫu cần thông tin sau, chỉ điền khi có bằng chứng:

| Trường | Nội dung |
|---|---|
| `sample_id` | ID nội bộ duy nhất cho sample |
| `source_id`, `source_sha256` | Capture gốc và checksum; crops/derivatives giữ ancestry |
| `image_sha256`, `mask_sha256` | SHA-256 nội dung file chính xác đang dùng |
| `fabric_id` | Loại vật liệu đã xác minh |
| `roll_id`, `lot_id`, `camera_id` | ID acquisition thật, không phải số tự đặt theo split |
| `acquisition_timestamp`, `acquisition_session` | Thời gian có timezone/session chụp nếu biết |
| `annotation_version` | Phiên bản nhãn sau review, gắn với mask hash |
| `annotator`, `reviewer`, `review_status` | Người annotate/review và kết luận |
| `origin`, `historical_status` | Real/synthetic; vai trò lịch sử, quyền sử dụng |
| `defect_types`, `multiple_defects` | Kết luận review mức ảnh; có thể khác taxonomy binary |

Không biết thì lưu `null` trong JSON hoặc ô trống trong CSV, kèm lý do/cần truy nguồn. Không ghi `roll_001`, `camera_001`, ngày giả hoặc `human_verified` cho đủ schema. Hash chỉ kiểm tra nội dung/duplicate; không tự chứng minh provenance hoặc nhãn đúng.

Đọc SHA-256 một file bằng PowerShell:

```powershell
(Get-FileHash -LiteralPath 'DUONG_DAN_ANH_THAT' -Algorithm SHA256).Hash.ToLowerInvariant()
```

**Lưu ý schema hiện tại:** `SegmentationSample` có `extra="forbid"`. Acquisition timestamp/session, annotation version, annotator/reviewer chưa phải field được trainer nhận trực tiếp. Lưu chúng trong sidecar CSV keyed by `sample_id`; không nhét tùy ý vào manifest rồi kỳ vọng trainer đọc được. Phase 1 cần kiểm tra sidecar, liên kết phiên bản và pin/hash bằng chứng review; một chuỗi `annotation_source="human_verified"` tự nó không chứng minh đã có người review.

## 7. Chia train/validation — chỉ sau approval Training Phase 1

1. Freeze reviewed source inventory và annotation version trước.
2. Chia theo connected source/ancestor/hash và roll, không random theo ảnh crop.
3. Group theo lot nếu protocol yêu cầu lot-disjoint; camera nếu protocol yêu cầu camera-disjoint. Một camera duy nhất không thể tạo train/val camera-disjoint bằng cách đổi tên camera.
4. Chỉ crop/augment sau khi nhóm nguồn đã thuộc một split; mọi derivative giữ cùng split với nguồn.
5. Tỉ lệ minh họa `train=0.8`, `validation=0.2`, `test=0.0`. Đây là thiết lập phát triển để review, không phải acceptance threshold hoặc cam kết đủ dữ liệu. Tỉ lệ thực tế gần đúng vì không cắt đôi nhóm.
6. Phải review coverage theo từng fabric và Good/BAD trong mỗi split. Splitter hiện tại **không bảo đảm stratification theo fabric/class**; tỷ lệ đẹp không chứng minh validation có đại diện đầy đủ.
7. Train và validation cần BAD với reviewed foreground; trainer hiện không chấp nhận train thiếu active foreground class hoặc validation hoàn toàn không có foreground.
8. Không tạo test mới bằng random split từ dữ liệu đã xem. `test_ratio=0` trong bước này; fresh locked test là quy trình tương lai độc lập.

Trainer hiện yêu cầu `require_roll_id=true`. Thiếu roll metadata thì **BLOCKED BY DATA**; không đặt false để vượt gate. Khi bật `group_by_lot`/`group_by_camera`, mọi eligible sample phải có ID tương ứng và đủ nhóm độc lập. Đừng tắt grouping chỉ để command chạy thành công.

## 8. JSON manifest mà trainer thực sự nhận

CLI hiện nhận `SegmentationManifest` JSON, **không tự chuyển folder GOOD/BAD hoặc CSV thành dataset**. Tên nhãn mức ảnh là `normal` và `defect`, không phải chuỗi `GOOD`/`BAD`.

Ví dụ template dưới đây chỉ minh họa format. Tên, ID và hash placeholder **không phải dữ liệu/annotation có thật**. Chỉ điền `human_verified` sau review; template chưa thể dùng để train. Hai row cũng chưa tạo thành train/validation đủ điều kiện.

Đặt file tương lai ở `Data/training/two_fabrics_real_v1/manifests/train_val.json`. Đường dẫn ảnh/mask trong JSON tính tương đối từ **thư mục chứa manifest**, không phải project root:

```json
{
  "schema_version": 1,
  "dataset_version": "two_fabrics_real_v1",
  "taxonomy": {"classes": ["background", "defect"], "ignore_index": null},
  "samples": [
    {
      "sample_id": "fabric_A_good_EXAMPLE",
      "fabric_id": "fabric_A",
      "image_path": "../reviewed/real/fabric_A/images/GOOD/good_EXAMPLE.png",
      "image_sha256": "REPLACE_WITH_ACTUAL_LOWERCASE_SHA256",
      "mask_path": "../reviewed/real/fabric_A/masks/GOOD/good_EXAMPLE_mask.png",
      "mask_sha256": "REPLACE_WITH_ACTUAL_LOWERCASE_SHA256",
      "label": "normal",
      "defect_classes": [],
      "mask_encoding": "class_ids",
      "binary_class_name": null,
      "mask_coordinates": "oriented_image",
      "origin": "real",
      "annotation_source": "human_verified",
      "source_id": "REPLACE_WITH_VERIFIED_SOURCE_ID_A",
      "source_sha256": "REPLACE_WITH_ACTUAL_LOWERCASE_SOURCE_SHA256",
      "ancestor_source_ids": [],
      "ancestor_source_sha256s": [],
      "roll_id": null,
      "lot_id": null,
      "camera_id": null,
      "split": "unassigned"
    },
    {
      "sample_id": "fabric_B_bad_EXAMPLE",
      "fabric_id": "fabric_B",
      "image_path": "../reviewed/real/fabric_B/images/BAD/bad_EXAMPLE.png",
      "image_sha256": "REPLACE_WITH_ACTUAL_LOWERCASE_SHA256",
      "mask_path": "../reviewed/real/fabric_B/masks/BAD/bad_EXAMPLE_mask.png",
      "mask_sha256": "REPLACE_WITH_ACTUAL_LOWERCASE_SHA256",
      "label": "defect",
      "defect_classes": ["defect"],
      "mask_encoding": "class_ids",
      "binary_class_name": null,
      "mask_coordinates": "oriented_image",
      "origin": "real",
      "annotation_source": "human_verified",
      "source_id": "REPLACE_WITH_VERIFIED_SOURCE_ID_B",
      "source_sha256": "REPLACE_WITH_ACTUAL_LOWERCASE_SOURCE_SHA256",
      "ancestor_source_ids": [],
      "ancestor_source_sha256s": [],
      "roll_id": null,
      "lot_id": null,
      "camera_id": null,
      "split": "unassigned"
    }
  ]
}
```

Mỗi ảnh reviewed của **cả hai fabric** cần row riêng; hai row trên không đại diện đầy đủ coverage. Hash phải là 64 ký tự hex lowercase. Roll/lot/camera còn null thể hiện chưa biết và **chưa đủ điều kiện qua gate training**. Manifest Phase 1 được duyệt cần pin các row train/validation sau khi split đã review, thay vì để splitter âm thầm quyết định lại ở mỗi experiment.

## 9. Kiểm tra môi trường trước training

Không thay dependency `.venv` hiện tại hoặc NVIDIA driver chỉ để chạy hướng dẫn. Không tự sửa `.venv-tensorrt` dùng cho Phase 17.

Kiểm tra chỉ đọc môi trường CPU hiện có:

```powershell
.\.venv\Scripts\python.exe -c "import sys,torch,torchvision; print(sys.version); print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('CUDA build',torch.version.cuda); print('CUDA available',torch.cuda.is_available())"
```

Kiểm tra metadata ngày soạn tài liệu cho thấy `.venv-tensorrt` **thiếu torchvision**. Vì vậy không được coi môi trường đó đã sẵn sàng cho trainer, dù TensorRT research từng hoạt động. Không chạy ngay trainer trong đó.

Nếu muốn train GPU, cần một bước setup được duyệt riêng cho môi trường training, ví dụ `.venv-train`, với PyTorch/torchvision/CUDA tương thích driver. Tài liệu này không cài môi trường hoặc đưa lệnh nâng/downgrade driver. Sau setup phải xác nhận import, CUDA, cached pretrained weights và smoke test độc lập rồi dừng review.

Trainer dùng pretrained ResNet18 từ cache local có kiểm tra hash, không tự tải weights/fallback sang random khi cache thiếu. ImageNet pretrained **không phải** fabric-defect trained. Không dùng checkpoint Phase 17 research probe làm model đã train.

## 10. Config training tham khảo — mặc định vẫn OFF

Sau khi được duyệt, có thể tạo **config riêng mới** `config/training_two_fabrics_v1.json`; không sửa config mẫu hoặc runtime config hiện hữu. Dưới đây là template tương thích schema hiện tại, **không tạo file config trong lần viết hướng dẫn này**:

```json
{
  "pipeline": {
    "taxonomy": {"classes": ["background", "defect"], "ignore_index": null},
    "split": {
      "train_ratio": 0.8,
      "validation_ratio": 0.2,
      "test_ratio": 0.0,
      "seed": 42,
      "require_roll_id": true,
      "group_by_lot": true,
      "group_by_camera": false
    },
    "augmentation": {"enabled": false},
    "training": {
      "enabled": false,
      "architecture": "unet_resnet18",
      "share_anomaly_encoder": false,
      "image_size": [256, 256],
      "batch_size": 1,
      "epochs": 30,
      "encoder_lr": 0.00003,
      "decoder_lr": 0.0003,
      "optimizer": "adamw",
      "weight_decay": 0.01,
      "loss": "cross_entropy_dice",
      "focal_optional": false,
      "early_stopping_patience": 8,
      "amp_if_cuda": false,
      "allow_synthetic": false
    }
  },
  "pretrained_encoder": true,
  "freeze_encoder_batchnorm": true,
  "decoder_channels": [128, 64, 32, 16, 16],
  "device": "cuda",
  "seed": 42,
  "cpu_threads": 2,
  "ce_weight": 1.0,
  "dice_weight": 1.0,
  "focal_gamma": 2.0,
  "min_delta": 0.0,
  "gradient_clip_norm": 5.0
}
```

Giải thích và giới hạn:

- Khởi điểm nghiên cứu cho RTX 3050 4 GB là batch 1, input 256, FP32; đây không phải cấu hình tối ưu hoặc bảo đảm không OOM. Không giữ cả anomaly và segmentation model trên GPU trong lúc train offline.
- `training.enabled=false` giữ gate đóng. Chỉ đổi thành true trong config experiment riêng **sau approval training**, không đổi ngay theo tài liệu này.
- `device="cuda"` fail rõ nếu không có CUDA, không âm thầm fallback CPU. Nếu chủ động chọn chạy CPU, tạo config riêng với `device="cpu"`; không tuyên bố đó là GPU training.
- `group_by_lot=true` đòi lot ID thật và đủ nhóm. `group_by_camera=false` nghĩa là **không chứng minh camera-disjoint**; setting này cần được review theo protocol, không được dùng để claim camera-generalization/production acceptance.
- 30 epoch là trần minh họa, không phải số epoch đã chạy hoặc số đủ để đạt chất lượng. Early stopping hiện theo validation loss.
- AMP đang OFF để thiết lập baseline FP32. AMP training chỉ xem xét trong experiment được duyệt và kiểm chứng numerical stability riêng. FP16 TensorRT deployment vẫn REJECTED, không fallback.
- Không fine-tune anomaly/FAISS encoder bằng segmentation loss. Model supervised hiện dùng encoder độc lập.
- Resize toàn ảnh về 256 có thể làm mất defect nhỏ hoặc méo aspect ratio. Cần review độ phân giải/patch policy trong Phase 1 trước khi tăng lên 512 hoặc dùng crops; không tăng size/batch theo cảm tính khi VRAM chưa đo.

## 11. Quy trình chạy sau khi đủ dữ liệu và được duyệt

### Gate A — được phép bắt đầu Training Phase 1?

Phải có reviewed annotation set, review records, metadata hợp lệ và explicit approval. Nếu thiếu, dừng ở DATA PREPARATION. Chỉ tổ chức folder không tự qua gate.

### Gate B — Phase 1 hoàn thành, được phép training?

Kiểm tra manifest/schema, image/mask SHA, geometry, orientation, class IDs, Good empty masks, BAD foreground, duplicates/ancestry và group disjointness. Không để historical/excluded lọt train/validation. Review số mẫu và số nguồn/roll/lot độc lập theo từng fabric/nhãn.

Không tự ấn định một số ảnh/class tối thiểu rồi kết luận đủ chất lượng. Cần owner review coverage, điều kiện chụp và mức đại diện thực tế. Phase 1 phải test và dừng chờ approval trước Phase 2.

### Gate C — command training, CHỈ sau approval Phase 2

CLI có sẵn:

```powershell
.\.venv\Scripts\python.exe scripts/train_segmentation_baseline.py --help
```

Lệnh training dưới đây là **ví dụ cho tương lai, chưa chạy**. Các path dataset/config/môi trường là output của các bước đã duyệt trước, không phải file được tài liệu này tạo sẵn:

```powershell
# CHỈ SAU APPROVAL TRAINING VÀ SETUP GPU ĐƯỢC XÁC NHẬN.
$TrainPython = '.\.venv-train\Scripts\python.exe'

& $TrainPython scripts/train_segmentation_baseline.py `
  --manifest 'Data/training/two_fabrics_real_v1/manifests/train_val.json' `
  --config 'config/training_two_fabrics_v1.json' `
  --output-dir 'storage/models/segmentation_candidates/two_fabrics_real_v1_run001'
```

Trước command, config experiment phải được review và bật `training.enabled=true` có chủ đích. Không thêm `--allow-synthetic`, không chạy `--demo` rồi gọi đó là model train từ dữ liệu thật. Script hiện không có `--resume`, `--epochs` hoặc `--batch-size`; các tham số đó nằm trong JSON config. Dùng output version mới cho mỗi run, không ghi đè artifact cũ.

Trainer có thể chọn model chung theo loss gộp; điều đó không bảo đảm chất lượng riêng từng fabric. Cần đánh giá per-fabric trong phase evaluation được duyệt sau này; không dùng validation liên tục để suy ra fresh-test performance.

## 12. Artifact và metrics thực sự do trainer hiện tại tạo

```text
storage/models/segmentation_candidates/two_fabrics_real_v1_run001/
├── config.json
├── manifest.json
├── split.json
├── best_epoch_0001.pt      # Ví dụ tên, chỉ xuất hiện nếu epoch đó cải thiện
├── best_epoch_....pt      # Mỗi lần cải thiện lưu file mới, không overwrite
└── report.json
```

Đây là tên/format **của code hiện có**, không cam kết đã có các file hoặc epochs trên. Không mặc định trainer đã tạo `model.pt`, `training_history.json`, `model_metadata.json` riêng. Provenance/checkpoint metadata nằm trong checkpoint; config/manifest/split/report lưu như trên.

- Checkpoint hiện chọn theo **validation loss thấp nhất**, không theo best mean Dice. Nếu workstream muốn tiêu chí khác, phải bổ sung/review trong phase implementation được duyệt; không mô tả nhầm hành vi hiện tại.
- `report.json` hiện ghi train loss, validation loss/metrics theo epoch, checkpoint tốt nhất, split counts và warnings. Checkpoint chứa model/optimizer/scaler state nhưng CLI chưa có resume workflow.
- Metrics hiện là **pixel-level**: precision, recall, IoU, Dice, confusion matrix và pixel accuracy. Không tự gọi pixel recall là image/defect-instance recall hoặc underkill của nhà máy.
- Trainer chưa tự ghi đầy đủ mọi field của training workstream như LR history, explicit F1 field, per-fabric summaries, full environment provenance hay review-sidecar hashes. Các phần thiếu cần hoàn thiện trong phase tương ứng, không tạo số liệu giả.
- `test_partition_evaluated=false`, `production_approved=false`; đây chỉ là research/development candidate.
- Thiếu class support thì metrics có thể null/undefined; không thay null bằng số hoàn hảo. Không lấy pixel accuracy cao trên nền nhiều để kết luận model bắt lỗi tốt.
- Sau training cần checkpoint validation, real validation previews, CPU/CUDA checks và các phase local integration/export theo approval riêng. Không tự load vào API/registry sau khi command kết thúc.

## 13. Kiểm thử và điểm dừng

Sau mỗi phase được duyệt phải chạy unit, integration và full regression, ghi số pass/fail thực tế, rồi STOP chờ review. Không lấy số test của phase cũ làm kết quả mới.

Command full regression hiện có:

```powershell
.\.venv\Scripts\python.exe scripts/check_phase0_baseline.py --regression --frontend
```

Lưu ý: audit integration tests của Training Phase 0 đang pin số lượng dữ liệu tại snapshot cũ. Khi dataset mới được nhập vào `Data/`, một số assertion inventory có thể cần review/update trong phase được duyệt; không xóa ảnh mới hoặc nới assertion một cách tùy tiện để ép test pass. Audit script Training Phase 0 chủ yếu hiểu các manifest cũ, không thay thế validator/review cho dataset mới.

Các lệnh trong mục này là hướng dẫn để chạy đúng lúc, không phải kết quả đã chạy trong lần viết tài liệu. Không chạy trainer chỉ để kiểm tra Markdown.

## 14. Checklist bàn giao dataset

- [ ] Xác nhận hai fabric và ID, không suy từ folder.
- [ ] Review từng ảnh BAD: lỗi thật, loại lỗi, đa lỗi, orientation, suitability.
- [ ] Có human-reviewed pixel masks cho BAD; có annotator/reviewer/version/hash.
- [ ] Review từng ảnh GOOD và xác nhận empty masks sau review.
- [ ] Không biết metadata thì để null/blank; truy nguồn trước khi gate yêu cầu.
- [ ] Có roll/lot thật và đủ nhóm độc lập cho split protocol; camera provenance được ghi đúng.
- [ ] Source/crop/augmentation lineage và SHA rõ ràng; duplicate không qua split.
- [ ] Synthetic tách khỏi real manifest; historical không thành fresh locked test.
- [ ] Dataset mới được kiểm tra coverage riêng theo fabric, class và acquisition group.
- [ ] Training Phase 1 được duyệt, triển khai/test xong và đã dừng review.
- [ ] Môi trường GPU training được duyệt/xác nhận riêng, không sửa CPU/TensorRT env hiện hữu.
- [ ] Có explicit approval Phase 2 trước khi bật training hoặc chạy command train.

**Nếu thiếu BAD thật hoặc reviewed masks/provenance cần thiết: BLOCKED BY DATA.**

Giữ nguyên runtime anomaly, FAISS, thresholds, API và references hiện tại. `fabric_001`/`fabric_002` vẫn INACTIVE; không đăng ký/activate production candidate. FP16 deployment vẫn REJECTED. Không customer SLA, acceptance threshold hay production metric nào được tạo ra từ hướng dẫn này.

## 15. Mã nguồn đã đối chiếu

- `backend/app/core/segmentation_data.py`: schema taxonomy, sample, manifest và split/config.
- `backend/app/core/segmentation_training.py`: baseline config.
- `backend/app/ai/segmentation_dataset.py`: checksum, geometry, encoding, Good/BAD rules.
- `backend/app/ai/segmentation_split.py`: source/ancestor/hash/roll/lot/camera connected groups.
- `backend/app/ai/segmentation_model.py`: independent encoder/decoder, local pretrained cache.
- `backend/app/ai/segmentation_trainer.py`: gates, validation-loss selection, artifact outputs.
- `backend/app/ai/segmentation_metrics.py`: pixel-level metrics.
- `scripts/train_segmentation_baseline.py`: CLI flags thực có.
- `config/segmentation_baseline_example.json`: config mẫu hiện tại, không bị chỉnh sửa.

Tài liệu hướng dẫn được tạo riêng; không tạo dataset, masks, manifest/config thực, môi trường hay model candidate trong lần này.
