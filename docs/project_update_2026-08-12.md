# AI Fabric Detect MVP — Tổng hợp cập nhật dự án

> Cập nhật: 12/08/2026
> Trạng thái: MVP chạy local, đã train projection head, đã build FAISS reference và đã có giao diện kiểm tra ảnh.

## 1. Mục tiêu và phạm vi hiện tại

Dự án phát hiện bất thường trên bề mặt vải bằng cách so sánh từng vùng ảnh với thư viện vùng vải bình thường. Đây không phải mô hình object detection có nhãn bounding box như YOLO. Pipeline hiện tại kết hợp:

- ResNet18 pretrained làm bộ trích xuất đặc trưng;
- projection head được train bằng các cặp embedding;
- FAISS tìm vùng normal gần nhất;
- anomaly score và các ngưỡng để tìm patch bất thường;
- thuật toán gộp patch để tạo bounding box;
- luật diện tích để phân loại `Good`, `Acceptable` hoặc `Bad`.

MVP phù hợp để thử nghiệm anomaly detection theo từng loại vải. Chưa nên coi kết quả hiện tại là chỉ số production vì dữ liệu train, reference và evaluation chưa được tách độc lập hoàn toàn.

## 2. Kiến trúc hệ thống

```text
React/Vite UI :5173
       |
       | /api qua Vite proxy
       v
FastAPI backend :8001
       |
       +-- SQLite: inspection, defect, feedback, model version
       +-- PyTorch/torchvision: ResNet18 + projection head
       +-- FAISS: tìm nearest normal patch
       +-- storage/: upload, ảnh annotated, model, index, metadata
```

Backend dùng cổng `8001` vì cổng `8000` trên máy đang được Apache `httpd` sử dụng. Frontend proxy `/api` tới `http://127.0.0.1:8001`.

## 3. Cấu trúc và quy ước dataset

```text
Data/
  raw/
    Company/                         ảnh IQC nội bộ hiện tại
      normal/                        nơi thêm ảnh Company không lỗi trong tương lai
    external/
      20250129_Fabric_defects/
        Dataset/
          non defective/
          broken thread/
          damage/
          holes/
          Stain/
      Lusitano/
        nondefects/
      WFDD/
        <fabric>/
          train/good/
          test/good/
          test/<defect>/
          ground_truth/<defect>/
  processed/                         dữ liệu được xử lý tùy chọn

storage/
  models/                            checkpoint projection head
  indexes/                           embedding matrix và FAISS index
  metadata/                          manifest, registry, report, metadata reference
  uploads/                           ảnh người dùng tải lên
  annotated/                         ảnh kết quả có bounding box
```

Tổng manifest `raw_all.csv` hiện có **10.212 ảnh đầu vào**:

| Nguồn | Số ảnh | Vai trò |
|---|---:|---|
| Company | 14 | Mẫu lỗi/IQC đã đánh dấu |
| DME | 97 | Normal và defective |
| WFDD | 4.101 | Train normal, test normal/defective |
| Lusitano | 6.000 | Normal |
| Tổng | 10.212 | Train/evaluation candidates |

WFDD có thêm 241 ảnh `ground_truth`. Chúng chỉ được liên kết qua `mask_path`, không bị dùng như ảnh vải đầu vào.

Các ảnh đang nằm trực tiếp trong `Data/raw/Company` được gán mặc định là `defective`. Ảnh có tên `IQC_*` được suy luận loại lỗi từ tên file. Muốn tạo reference cho vải Company cần bổ sung ảnh thật sự không lỗi vào một trong các thư mục:

```text
Data/raw/Company/normal/
Data/raw/Company/good/
Data/raw/Company/nondefects/
```

## 4. Workflow tổng thể

### 4.1 Chuẩn bị manifest

```powershell
.\.venv\Scripts\python.exe scripts\prepare_extended_datasets.py
```

Script duyệt đệ quy toàn bộ ảnh được hỗ trợ trong `raw/Company` và các dataset thuộc `raw/external`, đọc kích thước ảnh, suy luận label và tạo:

- `company_all.csv`;
- `dme_all.csv`;
- `wfdd_all.csv`;
- `lusitano_all.csv`;
- `raw_all.csv` — manifest chính;
- `extended_all.csv` — alias tương thích;
- reference/evaluation candidate manifests;
- `manifest_report.json`.

### 4.2 Train projection head

```powershell
.\.venv\Scripts\python.exe scripts\train_extended_projection_head.py `
  --manifest-csv storage/metadata/extended_dataset_manifests/raw_all.csv `
  --epochs 20 `
  --batch-size 16
```

Quy trình train:

1. Đọc toàn bộ 10.212 ảnh có label `normal` hoặc `defective`.
2. Resize ảnh, chia patch và lấy tối đa 16 patch trải đều trên mỗi ảnh.
3. Dùng ResNet18 pretrained ở chế độ đóng băng để tạo embedding 512 chiều.
4. Tạo các cặp training:
   - positive: hai patch normal cùng nhóm vải;
   - negative: patch normal và defect cùng nhóm;
   - negative chéo: normal của nhóm khác;
   - Company chỉ có defect nên được ghép với normal pool bên ngoài.
5. Projection head học ánh xạ `512 -> 256 -> 128`.
6. Vector đầu ra được L2-normalize.
7. Tối ưu binary cross entropy trên cosine similarity nhân temperature.

Artifact hiện tại:

```text
storage/models/projection_head_raw_all_v1.pt
storage/metadata/training_reports/projection_head_raw_all_v1_report.json
```

Kết quả train:

| Chỉ số | Giá trị |
|---|---:|
| Ảnh đã dùng | 10.212 |
| Patch embedding | 163.375 |
| Tổng cặp | 60.000 |
| Positive pairs | 24.000 |
| Negative pairs | 36.000 |
| Epoch cuối | 20 |
| Train loss | 0,2342 |
| Validation loss | 0,3244 |
| Pair validation accuracy | 88,32% |

### 4.3 Build normal reference

```powershell
.\.venv\Scripts\python.exe scripts\build_extended_references.py --variant raw_all_v1
```

Mỗi loại vải có một FAISS index riêng. Không gộp tất cả texture vào cùng một index vì nearest neighbor từ loại vải khác có thể làm sai anomaly score.

| Reference | Ảnh normal | Patch |
|---|---:|---:|
| `dme_plain_raw_all_v1` | 17 | 2.040 |
| `wfdd_grey_cloth_raw_all_v1` | 288 | 28.800 |
| `wfdd_grid_cloth_raw_all_v1` | 2.132 | 213.200 |
| `wfdd_pink_flower_raw_all_v1` | 288 | 28.800 |
| `wfdd_yellow_cloth_raw_all_v1` | 949 | 94.900 |
| `lusitano_group_auto_raw_all_v1` | 5.999 | 269.955 |

Một file Lusitano bị bỏ qua vì có dung lượng 0 byte:

```text
Data/raw/external/Lusitano/nondefects/20240920_152632_014.tif
```

Registry có thể giữ lịch sử nhiều version của cùng stable `reference_id`. Runtime, evaluator và API danh sách hiện chọn version mới nhất của mỗi stable ID.

### 4.4 Đánh giá

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_extended_pipeline.py `
  --manifest-csv storage/metadata/extended_dataset_manifests/raw_all.csv `
  --reference-id dme_plain_raw_all_v1 `
  --max-images-per-group 50
```

Kết quả DME hiện tại tại threshold `0.075`:

| Metric | Giá trị |
|---|---:|
| TP | 32 |
| FP | 0 |
| TN | 17 |
| FN | 1 |
| Accuracy | 98,00% |
| Precision | 100,00% |
| Recall | 96,97% |
| F1 | 98,46% |

Đây là kiểm tra tính nhất quán của pipeline, chưa phải kiểm thử độc lập hoàn toàn. Projection head đã thấy dữ liệu trong manifest và normal DME cũng được dùng để build reference. Cần một test set chưa từng tham gia train/reference trước khi công bố độ chính xác thực tế.

### 4.5 Chạy ứng dụng

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app `
  --reload --reload-dir backend --host 127.0.0.1 --port 8001
```

Frontend, trong terminal khác:

```powershell
cd frontend
npm.cmd run dev
```

Địa chỉ:

- UI: `http://localhost:5173`
- API docs: `http://127.0.0.1:8001/docs`
- Health: `http://127.0.0.1:8001/api/health`

## 5. Nguyên lý hoạt động của model

### 5.1 Biểu diễn patch

Ảnh đầu vào được resize sao cho cạnh lớn tối đa 768 px, sau đó cắt thành patch `96 x 96` với stride 48. Mỗi patch được resize về `224 x 224`, normalize theo ImageNet và đưa qua ResNet18 pretrained.

ResNet18 bỏ lớp phân loại cuối để trả về vector đặc trưng 512 chiều:

```text
patch RGB -> ResNet18 pretrained -> embedding 512D -> L2 normalize
```

Projection head ánh xạ embedding sang không gian đã thích nghi với dữ liệu vải:

```text
512D -> Linear(256) -> ReLU -> Dropout(0.1) -> Linear(128) -> L2 normalize
```

### 5.2 Học bằng cặp embedding

Với hai vector đã normalize `z1`, `z2`, độ giống nhau là:

```text
similarity = cosine(z1, z2)
logit = similarity * temperature
loss = BinaryCrossEntropyWithLogits(logit, pair_label)
```

`pair_label = 1` biểu thị cặp nên giống nhau; `pair_label = 0` biểu thị cặp normal/defect hoặc khác texture cần được tách ra. ResNet18 không được fine-tune trong workflow hiện tại; chỉ projection head được cập nhật trọng số.

### 5.3 FAISS normal-reference search

Mọi embedding patch normal được lưu vào `IndexFlatIP`. Vì vector đã L2-normalize, inner product tương đương cosine similarity.

Với query patch `q`, FAISS tìm normal patch gần nhất `r*`:

```text
nearest_similarity = max cosine(q, r)
anomaly_score = max(0, 1 - nearest_similarity)
```

Patch được coi là bất thường khi:

```text
anomaly_score >= anomaly_threshold
```

Ngưỡng hiện tại là `0.075`.

### 5.4 Từ patch bất thường đến bounding box

Các patch bất thường chạm hoặc chồng lấn được gom thành connected component. Mỗi component trở thành một bounding box bao ngoài toàn bộ patch trong cụm.

Sau đó áp dụng bộ lọc:

```text
cluster.patch_count >= min_defect_cluster_patches
```

Giá trị runtime hiện tại là `6`. Đây là lý do các lỗi Company nhỏ có thể không tạo bounding box: model có thể chấm một vài patch là bất thường nhưng cụm dưới 6 patch bị loại.

### 5.5 Phân loại ảnh

Tỷ lệ diện tích lỗi:

```text
defect_area_ratio = tổng diện tích bounding box / diện tích ảnh gốc
```

Luật hiện tại:

```text
Không có box hoặc area < 2%  -> Good
2% <= area < 10%            -> Acceptable
area >= 10%                  -> Bad
```

Các ngưỡng chính:

| Cấu hình | Giá trị |
|---|---:|
| `ANOMALY_THRESHOLD` | 0,075 |
| `GOOD_AREA_THRESHOLD` | 0,02 |
| `BAD_AREA_THRESHOLD` | 0,10 |
| `MIN_DEFECT_CLUSTER_PATCHES` | 6 |

## 6. Cấu hình model đang kích hoạt

```text
FABRIC_DEFAULT_REFERENCE_ID=dme_plain_raw_all_v1
FABRIC_MODEL_NAME=resnet18
FABRIC_USE_PRETRAINED_WEIGHTS=true
FABRIC_USE_PROJECTION_HEAD=true
FABRIC_PROJECTION_HEAD_PATH=storage/models/projection_head_raw_all_v1.pt
FABRIC_PROJECTION_DIM=128
FABRIC_DEVICE=auto
FABRIC_IMAGE_MAX_SIZE=768
FABRIC_MODEL_INPUT_SIZE=224
FABRIC_PATCH_SIZE=96
FABRIC_PATCH_STRIDE=48
FABRIC_BATCH_SIZE=16
FABRIC_ANOMALY_THRESHOLD=0.075
FABRIC_GOOD_AREA_THRESHOLD=0.02
FABRIC_BAD_AREA_THRESHOLD=0.10
FABRIC_MIN_DEFECT_CLUSTER_PATCHES=6
```

## 7. Các cập nhật và bug đã xử lý

- Tích hợp toàn bộ ảnh Company, DME, WFDD và Lusitano vào manifest chung.
- Loại WFDD ground-truth mask khỏi tập ảnh đầu vào.
- Cho phép nhóm Company chỉ có defect vẫn tạo negative pairs khi train.
- Bỏ giới hạn mặc định số ảnh train theo group.
- Chọn patch trải đều thay vì chỉ lấy các patch đầu ảnh.
- Build reference bằng toàn bộ ảnh normal theo mặc định.
- Tự động nhận Company normal folder trong tương lai.
- Sửa đường dẫn projection checkpoint bị thiếu chữ `s` trong `storage`.
- Sửa evaluator chạy lặp nhiều version có cùng stable reference ID.
- Sửa API danh sách reference chỉ hiển thị version mới nhất của mỗi ID.
- Sửa tie-break threshold: nếu metric bằng nhau thì ưu tiên threshold cao hơn.
- Chuyển backend từ cổng 8000 sang 8001 do xung đột Apache.
- Frontend lint và production build đã pass.
- Bổ sung kiểm soát upload, kiểm tra tương thích embedding/reference, atomic artifact writes và transaction database trong các cập nhật trước.

## 8. Giới hạn và vấn đề đang mở

### 8.1 Company chưa có normal reference

Ảnh Company hiện được kiểm tra bằng reference DME. Hai nguồn có thể khác texture, ánh sáng, camera và tỷ lệ chụp. Vì vậy kết quả trên Company chưa ổn định. Cần thu thập normal ảnh cùng loại vải, cùng camera và cùng điều kiện chụp.

### 8.2 Bộ lọc 6 patch quá mạnh với lỗi nhỏ

Một lỗi nhỏ có thể chỉ tác động 1–4 patch và bị loại hoàn toàn. Nên đánh giá thử:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_extended_pipeline.py `
  --manifest-csv storage/metadata/extended_dataset_manifests/raw_all.csv `
  --reference-id dme_plain_raw_all_v1 `
  --max-images-per-group 50 `
  --min-cluster-patches 2
```

Chỉ giảm runtime xuống 2 nếu false-positive rate vẫn chấp nhận được trên test set độc lập.

### 8.3 Confidence hiện chưa phải xác suất đã hiệu chuẩn

Confidence đang được tính bằng heuristic, không phải xác suất calibrated. Với nhánh `Good`, công thức hiện tại còn có thể tăng confidence khi `max_anomaly_score` tăng. Vì vậy số `99%` trên UI không nên hiểu là model chắc chắn 99% ảnh không lỗi. Cần sửa công thức và thực hiện calibration trước khi dùng trong QC thực tế.

### 8.4 Localization còn thô

Patch 96, stride 48 tạo box theo vùng receptive patch nên bounding box có thể rộng hơn lỗi thật. Kết quả DME hiện có `average_defect_area_ratio` khoảng 0,61 trên ảnh lỗi được chọn, cho thấy classification tốt hơn localization.

### 8.5 Nguy cơ leakage trong evaluation

Evaluation hiện dùng dữ liệu đã liên quan tới train/reference. Cần tách theo ảnh hoặc theo lô sản xuất thành:

```text
train projection / build reference / validation threshold / final test
```

Không được để cùng ảnh hoặc các crop gần giống của một ảnh xuất hiện ở nhiều split.

## 9. Ưu tiên tiếp theo

1. Thu thập ảnh Company normal theo từng mã vải, camera và điều kiện ánh sáng.
2. Tạo split độc lập, ưu tiên chia theo batch/roll thay vì random từng ảnh.
3. Build `company_iqc` reference riêng cho từng loại vải.
4. Đánh giá `min_cluster_patches` từ 1–6 và đồng thời đo recall/FPR.
5. Calibrate anomaly threshold theo từng reference thay vì dùng một threshold toàn cục.
6. Sửa confidence để phản ánh bằng chứng phát hiện và hiệu chuẩn trên validation set.
7. Đánh giá localization bằng mask WFDD: IoU, Dice hoặc pixel AUROC.
8. Ẩn hoặc archive các reference legacy khỏi UI để người dùng không chọn nhầm.
9. Thêm progress/checkpoint cho bước extract embedding và build reference dài.
10. Chỉ chốt metric production sau khi chạy final test set chưa từng được sử dụng.

## 10. Tệp và artifact quan trọng

| Thành phần | Đường dẫn |
|---|---|
| Unified manifest | `storage/metadata/extended_dataset_manifests/raw_all.csv` |
| Projection checkpoint | `storage/models/projection_head_raw_all_v1.pt` |
| Training report | `storage/metadata/training_reports/projection_head_raw_all_v1_report.json` |
| Build reference report | `storage/metadata/build_extended_references_raw_all_v1.json` |
| Reference registry | `storage/metadata/reference_registry.json` |
| Evaluation summary | `storage/metadata/evaluation_results/extended_evaluation_summary.json` |
| Dataset preparation | `scripts/prepare_extended_datasets.py` |
| Projection training | `scripts/train_extended_projection_head.py` |
| Reference building | `scripts/build_extended_references.py` |
| Evaluation | `scripts/evaluate_extended_pipeline.py` |
| Runtime inspection | `backend/app/services/inspection_service.py` |

## 11. Cập nhật bbox/localization ngày 2026-08-14

- Chuẩn hóa EXIF orientation xuyên suốt lúc đọc ảnh, tiền xử lý và vẽ ảnh annotated để bbox không bị lệch/trục xoay.
- Khi phần lớn patch cùng bị kích hoạt vì ảnh khác miền so với reference, pipeline không còn gộp chúng thành bbox toàn ảnh.
- Lỗi cục bộ được tìm bằng residual so với nền vải làm mượt và connected components.
- Lỗi dạng sợi/hàng kéo dài (`end-out`, `mixed-yarn`, `warp-float`) được tìm bằng profile gradient/variance theo hàng và cột, robust z-score và hysteresis; các điểm thẳng hàng được gom thành một bbox dài.
- Nhãn trên frontend hiển thị `SCORE` thô thay cho phần trăm vì anomaly distance không phải xác suất.
- Hồi quy với `company_iqc_raw_all_v2`, threshold `0.15`: `TP=14`, `FP=0`, `TN=63`, `FN=0`; báo cáo nằm tại `storage/metadata/evaluation_results_bbox_axis_v2/`.
- API thực tế đã xác nhận ảnh `images.jpeg` chuyển từ `Good / 0 bbox` thành `Acceptable / 1 bbox`, bbox `(158, 0, 14, 248)` trên ảnh `355x248`.

Kết quả trên vẫn dùng bộ Company liên quan tới train/reference nên chưa đại diện cho final test độc lập. Ảnh web khác texture/camera có thể vẫn bị bỏ sót; production cần reference riêng theo loại vải và một tập test chưa từng dùng để train hoặc calibrate.

## 12. Tổ chức lại Company và thử nghiệm synthetic ngày 2026-08-14

Cấu trúc ảnh real đã được chuẩn hóa:

```text
Data/raw/Company/
├── real/
│   ├── fabric_A_warp_knit_mesh/good/              # 36 ảnh
│   ├── fabric_B_brushed_pile_velour_like/good/    # 27 ảnh
│   └── unclassified/defect/                       # 14 ảnh
└── fabric_synthetic_testset/
    ├── fabric_A_warp_knit_mesh/
    └── fabric_B_brushed_pile_velour_like/
```

- 63 ảnh trong các thư mục synthetic `original/` là bản sao byte-identical của 63 ảnh real Good nên được giữ để truy vết nhưng loại khỏi manifest model.
- Contact sheet và 20 `defect_masks` không được coi là ảnh đầu vào.
- Synthetic được chia theo `source_original`, không chia ngẫu nhiên theo filename: 76 ảnh train từ 29 nguồn và 24 ảnh held-out từ 8 nguồn, không có nguồn giao nhau.
- Manifest Company: 153 record train (`126 normal`, `27 defective`), 63 real normal reference, 24 synthetic held-out (`17 normal`, `7 defective`).
- Với synthetic defect có mask, trainer chỉ gán nhãn defect cho patch giao với mask; không còn gán nhãn sai cho toàn bộ nền ảnh.
- Trainer đã được sửa để lưu `best checkpoint` trước early stopping thay vì trọng số epoch cuối.
- Projection head tốt nhất của lượt thử nghiệm là `storage/models/projection_head_company_v7.pt`, best epoch 7, validation pair accuracy `0.9983`.

Các reference candidate v3–v8 đã được build và giữ lại để phân tích. Tuy nhiên held-out synthetic cho thấy trade-off FP/FN chưa đạt mức production; candidate v8 có recall `0.8571` nhưng false-positive rate `0.8235`. Vì vậy runtime và `latest_reference.json` đã được khôi phục về `company_iqc_raw_all_v2` thay vì tự động kích hoạt candidate mới.
