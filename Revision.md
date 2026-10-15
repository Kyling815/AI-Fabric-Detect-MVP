# Tổng quan kiến trúc hiện tại và định hướng tối ưu Fabric Detection bằng CNN

## 1. Mục đích tài liệu

Tài liệu này tổng hợp sơ bộ kiến trúc hiện tại của dự án **AI Fabric Detect MVP**, trạng thái mô hình, các giới hạn đã ghi nhận và định hướng kỹ thuật trước khi tiếp tục tối ưu hệ thống bằng Deep Learning CNN.

Điểm cần làm rõ là dự án hiện tại **đã sử dụng CNN**. ResNet18 đang đóng vai trò bộ trích xuất đặc trưng thị giác. Tuy nhiên, hệ thống chưa phải là một mô hình CNN được huấn luyện end-to-end để trực tiếp phân loại hoặc phân vùng lỗi. Kiến trúc hiện tại phù hợp hơn với mô tả:

> CNN feature extractor + nearest-neighbor anomaly detection + rule-based decision.

## 2. Kiến trúc tổng thể

Luồng xử lý chính của hệ thống:

```text
React/Vite UI
    │
    │ POST /api/inspect: image + fabric_id
    ▼
FastAPI Backend
    │
    ├── Fabric Registry
    │     └── Chọn đúng reference theo loại vải
    │
    ├── ResNet18 layer2 + layer3
    │     └── Trích xuất dense feature embedding
    │
    ├── Shared Projection Adapter
    │     └── Điều chỉnh embedding theo dữ liệu công ty
    │
    ├── FAISS Memory Bank
    │     └── So sánh với đặc trưng của các mẫu vải tốt
    │
    ├── Anomaly Map
    │     └── Threshold + morphology + connected components
    │
    ├── Rule Engine
    │     └── GOOD / NEEDS_REVIEW / BAD
    │
    └── SQLite + File Storage
          └── Lưu kết quả và ảnh có bounding box
```

## 3. Các thành phần phần mềm

### 3.1. Frontend

Frontend được xây dựng bằng:

- React 19;
- TypeScript;
- Vite.

Các chức năng chính gồm:

- tải danh sách loại vải từ backend;
- yêu cầu người vận hành chọn `fabric_id`;
- tải ảnh cần kiểm tra;
- gửi ảnh và loại vải tới API `/api/inspect`;
- hiển thị kết quả `GOOD`, `NEEDS_REVIEW` hoặc `BAD`;
- hiển thị anomaly score, số vùng lỗi, tỷ lệ vùng lỗi và ảnh đã đánh dấu;
- hiển thị thông tin backbone và trạng thái hệ thống.

Các tệp chính:

- `frontend/src/App.tsx`;
- `frontend/src/services/api.ts`;
- `frontend/src/components/InspectionBay.tsx`;
- `frontend/src/components/ReferenceList.tsx`;
- `frontend/src/components/ResultLog.tsx`.

### 3.2. Backend

Backend sử dụng FastAPI và được chia thành các nhóm API:

- health check;
- dataset discovery;
- fabric registry;
- inspection;
- reference library;
- operator feedback;
- model information.

Endpoint runtime chính:

```text
POST /api/inspect
multipart/form-data:
  image=<image file>
  fabric_id=<selected fabric id>
```

Backend cũng phục vụ ảnh upload và ảnh annotated qua static route. Các checkpoint, FAISS index, memory bank và metadata không được công khai qua static route.

### 3.3. Cơ sở dữ liệu và lưu trữ

Hệ thống sử dụng SQLite để lưu:

- thông tin inspection;
- kết quả cuối cùng;
- thông tin ảnh;
- vùng lỗi đã phát hiện;
- phản hồi của người vận hành.

Filesystem được tổ chức để lưu riêng:

```text
Data/
  raw/                  Dữ liệu gốc
  processed/            Dữ liệu đã xử lý

storage/
  uploads/              Ảnh được tải lên
  annotated/            Ảnh có bounding box
  models/               Checkpoint và projection adapter
  indexes/              FAISS index
  references/           Memory bank và artifact theo fabric/version
  metadata/             Registry, manifest và báo cáo đánh giá
```

## 4. Pipeline AI hiện tại

### 4.1. Chọn reference theo loại vải

Người vận hành bắt buộc chọn một `fabric_id`. Backend sử dụng `FabricRegistry` để:

- xác minh loại vải tồn tại;
- kiểm tra fabric đang ở trạng thái `ACTIVE`;
- lấy đúng reference đang được kích hoạt;
- kiểm tra đầy đủ metadata, memory bank và FAISS index;
- xác minh phiên bản reference tương thích với phiên bản backbone;
- không tự động fallback sang reference của loại vải khác.

Cơ chế này giúp tránh so sánh ảnh query với texture reference không cùng loại vải.

### 4.2. Tiền xử lý ảnh runtime

Trong pipeline Fabric A/B hiện tại, ảnh được:

1. đọc và chuyển sang RGB;
2. giữ nguyên tỷ lệ khung hình;
3. resize để cạnh dài không vượt quá 768 pixel;
4. chuyển thành tensor;
5. chuẩn hóa bằng mean và standard deviation của ImageNet.

Dự án còn có preprocessing v2 với các khả năng như kiểm tra chất lượng, hiệu chỉnh độ sáng, CLAHE, bilateral denoising, ROI, orientation, overlapping patch và texture view. Tuy nhiên, runtime Fabric A/B hiện tại sử dụng trực tiếp dense extractor và chưa khai thác toàn bộ chuỗi preprocessing v2.

### 4.3. Dense feature extraction bằng CNN

Backbone hiện tại là ResNet18 pretrained trên ImageNet.

Hệ thống lấy feature map từ:

- `layer2`;
- `layer3`.

Quy trình xử lý feature:

1. làm mượt feature bằng average pooling;
2. nội suy feature của `layer3` về cùng kích thước không gian với `layer2`;
3. L2-normalize từng feature map;
4. nối hai feature map theo chiều channel;
5. L2-normalize kết quả;
6. xem mỗi vị trí trên feature map là một local patch embedding.

Dense feature ban đầu có 384 chiều và feature stride là 8 pixel trên ảnh đã resize.

### 4.4. Shared projection adapter

Sau ResNet18, hệ thống có thể sử dụng một projection adapter đã huấn luyện để biến đổi dense embedding sang không gian đặc trưng phù hợp hơn với dữ liệu vải của công ty.

Backbone được đăng ký hiện tại:

```text
shared_resnet18_imagenet1k_dense_adapter_company_v4
```

Kiến trúc được mô tả trong registry:

```text
resnet18_layer2_layer3_dense+shared_projection_adapter
```

Adapter được dùng chung cho các loại vải, trong khi memory bank và threshold được tách riêng theo từng fabric.

### 4.5. Memory bank và FAISS

Mỗi loại vải có một memory bank riêng, chỉ chứa embedding từ ảnh được xác định là Real Good của chính loại vải đó.

Trong inference:

1. mỗi query embedding được đưa vào FAISS;
2. hệ thống tìm reference vector gần nhất với `k=1`;
3. khoảng cách tới vector gần nhất được dùng làm anomaly score của vị trí tương ứng;
4. toàn bộ patch score được reshape thành anomaly map hai chiều.

Thiết kế này gần với phương pháp PatchCore và phù hợp với bài toán anomaly detection khi số lượng ảnh lỗi được gán nhãn còn hạn chế.

### 4.6. Tổng hợp image score

Hệ thống hỗ trợ nhiều phương pháp tổng hợp anomaly score, chẳng hạn:

- maximum score;
- percentile 99.5%;
- trung bình của nhóm top-k;
- median contrast của nhóm top-k.

Phương pháp và threshold được hiệu chỉnh riêng cho từng fabric.

### 4.7. Định vị vùng lỗi

Anomaly map được chuyển thành vùng lỗi bằng quy trình:

1. áp dụng pixel threshold;
2. tạo binary anomaly mask trên feature grid;
3. dùng morphology closing để nối các cell lân cận;
4. resize mask về kích thước ảnh đã xử lý;
5. tìm connected components;
6. loại bỏ component nhỏ hơn `minimum_defect_area`;
7. tạo bounding box và ánh xạ tọa độ về ảnh gốc.

### 4.8. Luật quyết định

Kết quả cuối cùng không được dự đoán trực tiếp bởi một classification head. Hệ thống dùng rule engine dựa trên:

- image anomaly score;
- image review threshold;
- image bad threshold;
- bounding box đã định vị;
- localized anomaly score;
- localized review threshold.

Quy tắc chính:

- dưới review threshold và không có localized trigger: `GOOD`;
- dưới review threshold nhưng có localized anomaly mạnh: `NEEDS_REVIEW`;
- vượt review threshold nhưng không có bounding box: `NEEDS_REVIEW`;
- vượt bad threshold và có bounding box: `BAD`;
- nằm giữa review và bad threshold: `NEEDS_REVIEW`.

Việc đưa các trường hợp không chắc chắn về `NEEDS_REVIEW` là cơ chế fail-safe nhằm hạn chế false negative.

## 5. Quản lý model và reference

Kiến trúc quản lý artifact hiện tại gồm:

- một shared backbone;
- projection adapter có version;
- memory bank riêng theo fabric;
- FAISS index riêng theo fabric;
- threshold profile riêng theo fabric;
- reference version độc lập;
- registry kiểm soát reference được phép chạy runtime.

Hiện có candidate từ v1 đến v5 cho:

- `fabric_001` — Fabric A;
- `fabric_002` — Fabric B.

Candidate v5 sử dụng:

- Fabric A: 33 ảnh Real Good và 12.000 reference vectors;
- Fabric B: 40 ảnh Real Good và 12.000 reference vectors;
- shared dense adapter v4 không thay đổi trọng số.

## 6. Trạng thái đánh giá hiện tại

### 6.1. Kết quả v4 trên locked test

Tiêu chí chấp nhận được đặt ra là:

- không có false negative;
- defect recall đạt 100%;
- false-positive rate không vượt quá 10% cho từng loại vải.

Kết quả:

| Fabric | TP | FP | TN | FN | Recall | FPR | Mean mask IoU | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fabric A | 20 | 18 | 2 | 0 | 100% | 90% | 2,80% | Không |
| Fabric B | 19 | 0 | 20 | 1 | 95% | 0% | 17,23% | Không |

Fabric A phát hiện đủ ảnh lỗi nhưng nhầm phần lớn ảnh tốt. Fabric B phân tách ảnh tốt tốt hơn nhưng bỏ sót một lỗi `broken_or_loose_yarn` ở tầng quyết định image-level.

### 6.2. Kết quả v5 ablation

Core v5 bổ sung localized evidence vào rule engine. Kết quả trên hard-development set:

| Fabric | TP | FP | TN | FN | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fabric A | 20 | 18 | 2 | 0 | 100% | 90% |
| Fabric B | 20 | 1 | 19 | 0 | 100% | 5% |

Thay đổi v5 giúp đưa lỗi bị bỏ sót của Fabric B sang `NEEDS_REVIEW`. Tuy nhiên:

- Fabric A không được cải thiện;
- locked set v4 đã được dùng cho postmortem nên không còn là locked test hợp lệ cho release mới;
- chưa có fresh source-disjoint locked set cho v5.

Do đó, cả Fabric A và Fabric B hiện vẫn ở trạng thái `INACTIVE`, không có active reference dành cho production.

## 7. Các giới hạn chính

### 7.1. Domain shift

Fabric A có sự chồng lấn lớn giữa embedding ảnh tốt và ảnh lỗi khi thay đổi:

- lot sản xuất;
- camera;
- ánh sáng;
- khoảng cách chụp;
- góc chụp;
- texture variation hợp lệ.

Một global scalar threshold không thể đồng thời duy trì recall cao và false-positive rate thấp trong điều kiện này.

### 7.2. Dữ liệu reference còn ít

Memory bank hiện chỉ được xây dựng từ 33 ảnh tốt của Fabric A và 40 ảnh tốt của Fabric B. Số lượng này chưa chắc bao phủ đủ biến thiên bình thường của quá trình sản xuất thực tế.

### 7.3. Khả năng định vị còn yếu

Mean mask IoU còn thấp, đặc biệt với Fabric A. Dense feature stride 8 và việc sử dụng feature trung gian của classification backbone có thể chưa đủ cho các lỗi nhỏ, mảnh hoặc có biên không rõ.

### 7.4. Quyết định còn phụ thuộc threshold

Classification cuối cùng hiện dựa trên rule và threshold được hiệu chỉnh thủ công. Khi phân phối dữ liệu thay đổi, threshold có thể mất hiệu lực.

### 7.5. Chưa huấn luyện end-to-end theo mục tiêu cuối

Backbone, projection adapter, FAISS retrieval, localization và decision engine là các tầng tách rời. Sai số không được lan truyền xuyên suốt pipeline theo loss của bài toán cuối cùng.

### 7.6. Hai nhánh pipeline cùng tồn tại

Codebase vẫn giữ cả:

- legacy patch pipeline với preprocessing v1/v2 và CNN patch embedding;
- dense Fabric A/B pipeline dùng ResNet18 + adapter + FAISS.

Điều này hỗ trợ backward compatibility nhưng làm tăng độ phức tạp khi cấu hình, đánh giá và bảo trì.

## 8. Định hướng tối ưu bằng Deep Learning CNN

### 8.1. Hướng 1 — Fine-tune CNN encoder, tiếp tục dùng FAISS

Đây là bước chuyển ít rủi ro nhất:

- giữ kiến trúc anomaly detection hiện tại;
- fine-tune một phần hoặc toàn bộ ResNet18;
- huấn luyện embedding bằng contrastive loss, triplet loss hoặc supervised metric learning;
- split train/validation theo source image, lot và camera;
- tiếp tục sử dụng memory bank riêng theo fabric.

Ưu điểm:

- phù hợp khi dữ liệu lỗi còn ít;
- giữ được khả năng phát hiện lỗi chưa từng thấy;
- tận dụng phần lớn pipeline hiện tại.

### 8.2. Hướng 2 — CNN classification end-to-end

Mô hình dự đoán trực tiếp:

- Good/Defect;
- hoặc nhiều lớp lỗi như stain, hole, broken yarn, scratch và loose yarn.

Hướng này yêu cầu:

- dữ liệu đủ lớn và cân bằng;
- nhãn ổn định;
- dữ liệu đa lot, camera và điều kiện ánh sáng;
- source-grouped split;
- xử lý class imbalance;
- calibration confidence.

Classification end-to-end có thể đơn giản hóa quyết định image-level nhưng không tự giải quyết bài toán định vị vùng lỗi.

### 8.3. Hướng 3 — CNN segmentation hoặc detection

Có thể sử dụng:

- U-Net hoặc U-Net++;
- DeepLabV3+;
- SegFormer khi dữ liệu và GPU đủ lớn;
- YOLO hoặc Faster R-CNN nếu nhãn chính là bounding box.

Ưu điểm:

- định vị lỗi trực tiếp;
- cải thiện chất lượng mask và bounding box;
- phù hợp hơn với lỗi nhỏ, dài hoặc có hình dạng không đều.

Hạn chế chính là cần bộ nhãn mask/bounding box chính xác và đủ lớn.

### 8.4. Hướng hybrid được khuyến nghị

Kiến trúc mục tiêu có thể kết hợp:

```text
Input image
    │
    ▼
Fine-tuned CNN encoder
    │
    ├── Image classification head
    │     └── Good / Defect / Defect type
    │
    ├── Segmentation or detection head
    │     └── Mask / bounding box
    │
    └── Embedding head
          └── FAISS anomaly detection cho unknown defect
```

CNN supervised xử lý các lớp lỗi đã biết, trong khi FAISS/anomaly branch duy trì khả năng phát hiện lỗi mới. Rule engine khi đó chỉ còn vai trò fusion và fail-safe thay vì là bộ phân loại chính.

## 9. Thứ tự triển khai đề xuất

1. Chuẩn hóa taxonomy cho Good, Defect và các loại lỗi.
2. Bổ sung ảnh Real Good đa lot, camera và điều kiện chụp.
3. Tạo split theo source/lot/camera, tuyệt đối không random theo patch.
4. Tạo fresh locked test chưa từng dùng để hiệu chỉnh.
5. Đánh giá lại baseline ResNet18 + FAISS trên split mới.
6. Fine-tune encoder bằng source-grouped metric learning.
7. So sánh ResNet18 với EfficientNet, ConvNeXt hoặc backbone phù hợp tài nguyên triển khai.
8. Thêm segmentation/detection head nếu chất lượng localization là yêu cầu bắt buộc.
9. Giữ anomaly branch để phát hiện lỗi ngoài taxonomy.
10. Chỉ kích hoạt reference/model khi mọi fabric vượt acceptance gate trên fresh locked test.

## 10. Kết luận

Dự án đã có nền tảng CNN tương đối hoàn chỉnh, bao gồm ResNet18 pretrained, dense multi-layer embedding, projection adapter, FAISS memory bank, anomaly localization, registry theo version và rule-based fail-safe.

Vấn đề chính hiện tại không phải là thiếu CNN, mà là:

- representation chưa đủ ổn định trước domain shift;
- dữ liệu Real Good chưa bao phủ đủ biến thiên;
- localization còn yếu;
- quyết định cuối phụ thuộc nhiều vào threshold;
- chưa có quá trình huấn luyện end-to-end theo mục tiêu inspection cuối cùng.

Bước tiếp theo phù hợp là fine-tune CNN encoder trên dữ liệu được chia theo source/lot, mở rộng reference coverage, sau đó phát triển kiến trúc hybrid gồm classification, localization và anomaly detection. Việc chuyển ngay sang classifier CNN end-to-end khi dữ liệu còn hạn chế có nguy cơ làm mô hình học thuộc lot, nền ảnh hoặc điều kiện chụp thay vì học đúng đặc trưng lỗi vải.

Tại thời điểm tổng hợp tài liệu, bộ kiểm thử tự động của dự án có **95 test passed**.
