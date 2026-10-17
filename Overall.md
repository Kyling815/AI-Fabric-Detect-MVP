# Overall — Đánh giá tổng thể AI Fabric Detect MVP

**Ngày đánh giá: 2026-10-17, Asia/Bangkok.**

Phạm vi: code, cấu hình, dữ liệu/artifact hiện có, báo cáo Phase 1–18, Training Phase 0 và tài liệu hướng dẫn đã tạo. Đây là đánh giá hiện trạng và đề xuất hướng đi, **không phải phê duyệt training, production release hoặc kiến trúc mới**.

Baseline đã push: commit `054da35bd56bd74d5bfd9a48cbfa43c303c46b34`, nhánh `main`. Các tài liệu/audit của workstream training mới vẫn có phần local chưa commit; không đánh đồng toàn bộ working tree với nội dung đã push.

## 1. Kết luận điều hành

**Dự án đã có nền tảng phần mềm/R&D khá đầy đủ, nhưng chưa có bằng chứng rằng một candidate hiện tại kiểm vải thật đủ tốt để đưa vào sử dụng nghiệp vụ.**

Nói ngắn gọn: đã xây được phần lớn “bộ máy để phát triển, thử nghiệm và kiểm soát việc phát hành”, nhưng chưa hoàn thành “dataset thật → model/reference đủ chất lượng → kiểm chứng độc lập → vận hành thực tế”.

| Mục tiêu đánh giá | Điểm hiện tại /10 | Diễn giải |
|---|---:|---|
| Nền tảng kỹ thuật phục vụ phát triển/demo có kiểm soát | **8/10** | Backend/frontend, anomaly pipeline, segmentation infrastructure, export, research benchmark và governance đã có; không đồng nghĩa đang mở cho inspection nghiệp vụ |
| MVP có giá trị trên ảnh vải thật của phạm vi đã chọn | **5/10** | Có ứng dụng và công cụ để tiến tới MVP; thiếu dataset reviewed, candidate hợp lệ và bằng chứng hiệu quả trên hai fabric |
| Production sử dụng để hỗ trợ/quyết định QC trong vận hành thật | **2/10** | Chưa vượt các gate về chất lượng, dữ liệu độc lập, phần cứng/full-hybrid, bảo mật, triển khai và vận hành |

Hai điểm chính trả lời câu hỏi tiến độ là **MVP: 5/10; production: 2/10**. Điểm 8/10 chỉ nói về nền tảng kỹ thuật, không dùng thay cho hai điểm sẵn sàng sản phẩm.

**Trạng thái release vẫn: BLOCKED BY DATA / TRAINED CANDIDATE.** `fabric_001` và `fabric_002` vẫn INACTIVE. Không có production candidate nào được chấp nhận/activate trong phạm vi 18 phase đã duyệt.

### Cách hiểu điểm số

Đây là **ước lượng chuyên môn theo bằng chứng**, không phải metric đo từ model, không phải phần trăm công việc đã hoàn thành và không phải thời gian/chi phí còn lại. Không lấy trung bình số phase hoặc số test để suy ra điểm readiness.

- 0: chưa có nền tảng đủ để triển khai công việc.
- 2: có thành phần kỹ thuật/R&D nhưng các gate sử dụng thực tế còn thiếu căn bản.
- 5: có khung ứng dụng đáng kể, song giá trị nghiệp vụ trên dữ liệu thật chưa được xác minh.
- 8: gần sẵn sàng trong **phạm vi mục tiêu đã định nghĩa**, còn các bước xác nhận/hardening cuối.
- 10: hoàn tất các gate đã thống nhất cho mục tiêu đó, có bằng chứng và approval phù hợp; không có nghĩa phát hiện mọi lỗi trên mọi fabric.

Điểm không được dùng để bỏ qua gate. MVP 5/10 không có nghĩa được phép ACTIVE hoặc cho hệ thống quyết định QC thật ngay.

## 2. “MVP” và “production” ở đây là gì?

### MVP được chấm 5/10

Bản local/development giới hạn cho hai fabric đã xác minh: người dùng đưa ảnh thật vào, nhận kết quả/hiển thị vùng nghi lỗi, có lịch sử và review bởi người phụ trách. Phải có protocol validation và candidate đủ bằng chứng cho phạm vi thử nghiệm; chưa tự động loại hàng hoặc thay thế người QC.

Ứng dụng kỹ thuật đã tồn tại, nhưng hiện không có reference ACTIVE để người dùng chọn và scan hai fabric theo workflow bình thường. Đây là **gate an toàn có chủ đích**, không phải lỗi cần “sửa” bằng cách bật lại reference cũ.

### Production được chấm 2/10

Một hệ thống chạy theo quy trình nhà máy thật, có chất lượng/SLA được owner định nghĩa và kiểm chứng, provenance đầy đủ, fresh locked test, full runtime evidence, quản lý quyền, triển khai/recovery và theo dõi vận hành. Nếu là hệ thống inline còn cần camera/ánh sáng/encoder/tốc độ vải và tích hợp điều khiển đã xác nhận.

Production có thể chỉ là workstation theo yêu cầu nghiệp vụ; không mặc định phải có PLC/cloud/HA. Tuy nhiên, một bản inline công nghiệp cần các kiểm chứng phần cứng tương ứng mà project hiện chưa có.

## 3. Những gì đã hoàn thành

### 3.1. Ứng dụng và anomaly pipeline hiện hữu

- Backend FastAPI có API inspection, fabric/reference, dataset, model-info, feedback và lịch sử inspection.
- Frontend React/TypeScript/Vite có chọn fabric, upload/preview ảnh, bbox overlay, kết quả và diagnostics/log UI.
- SQLite và file storage cho inspection, detected defects, ảnh input và ảnh annotated.
- Nhánh anomaly có ResNet18 dense features layer2/layer3, projection/adapter và FAISS memory theo fabric, anomaly map và decision rules.
- Preprocessing hiện hữu cùng đường preprocessing v2 có ROI, illumination/contrast/denoise, patches và diagnostics/config versioning.
- Kiểm tra fabric/reference compatibility, không fallback sang reference fabric khác, từ chối fabric INACTIVE.

**Giới hạn:** các điểm trên xác nhận đã có code và các lượt kiểm thử phần mềm; không xác nhận recall/FPR thật đạt yêu cầu. Các projection/adapter artifacts cũ không phải supervised segmentation candidate đã được chấp nhận.

### 3.2. Tổng hợp 18 phase kiến trúc

| Phase | Nội dung đã triển khai | Phần chưa được chứng minh/hoàn tất cho sử dụng thật |
|---|---|---|
| 1 | Refactor runtime/contracts và kiểm tra giữ hành vi cũ | Không tự cải thiện chất lượng model |
| 2 | Quality gate opt-in | Ngưỡng/chất lượng gate trên ảnh và camera thật cần validation |
| 3 | Pattern-period observation/diagnostics | Chưa chứng minh ích lợi trên toàn bộ fabric/điều kiện chụp thật |
| 4 | Anomaly memory/coreset experiments | Chưa có cấu hình được chấp nhận bằng evaluation thật độc lập |
| 5 | Fabric/lot memory hierarchy | Thiếu lot/roll provenance thật để xác nhận generalization |
| 6 | Pattern pooling và ablation tools | Lợi ích thật, đặc biệt giảm overkill, chưa được xác nhận |
| 7 | Dataset contracts, masks, grouped split, augmentation và pixel metrics | Chưa có reviewed real segmentation dataset đủ điều kiện |
| 8 | ResNet18 + U-Net-like decoder, trainer, checkpoint và early stopping | Chưa có supervised candidate được train/validate hợp lệ trên nhãn thật |
| 9 | Supervised shadow inference infrastructure, tách khỏi anomaly/QC | Chưa có trained candidate và local shadow evidence trên dữ liệu thật |
| 10 | Fusion evidence/contracts/rules | Chưa chứng minh hybrid hơn anomaly-only hoặc supervised-only |
| 11 | Tách QC business policy, observe/enforce opt-in | Chưa owner-approved criteria và acceptance evidence thực |
| 12 | Physical-coordinate contracts và calibration-aware mapping | Chưa xác nhận calibration/camera/encoder thực tế |
| 13 | Four-lane line-scan simulation, bounded queues/backpressure | Không phải real camera acquisition hoặc hardware synchronization |
| 14 | Seam/duplicate grouping và merge infrastructure | Chưa kiểm chứng biên nối/đếm lỗi trên scan vải thật |
| 15 | Stage profiling, bounded offline scheduler, CPU probes | Scheduler chưa tự nối toàn bộ API/camera/DB; chưa full factory throughput/soak |
| 16 | ONNX export/check/reload/parity trên CPU | Candidate train thật phải export và chạy parity lại |
| 17 | TensorRT FP32 full-architecture research probe, parity/timing/VRAM measurements | Probe synthetic/untrained; không phải trained fabric-defect model hoặc full-hybrid latency |
| 18 | Acceptance/release governance, hash pinning, freshness/disjointness, manual approval/activation adapter | Chưa có real candidate/evidence; không production activation |

**“Hoàn tất 18 phase implementation” không có nghĩa “hoàn tất sản phẩm công nghiệp”.** Nhiều phase hoàn thành dưới phạm vi software/research/fixture đã được duyệt rõ ràng.

### 3.3. Quản trị phát hành đã có nền tảng tốt

Luồng đã triển khai: candidate artifact → validation → fresh disjoint locked test → acceptance report → independent manual approval → manual activation request.

Có checks cho recall, underkill/FN, FPR/overkill, precision, F1, IoU/Dice khi áp dụng, unknown recall, false alarms/missed defects per 100 m, P50/P95, throughput và provenance/version. Config criteria chưa được owner duyệt vẫn null; thiếu bằng chứng thì block thay vì tự tạo kết quả PASS.

Framework này **nhập evidence từ trusted evaluator**, không tự là model runner/event matcher hoặc hệ thống xác minh nhãn/custody. Hash pinning phát hiện thay đổi nội dung, không chứng minh người cung cấp khai báo đúng. Manual reviewer hiện chưa được bảo đảm bằng PKI/SSO/RBAC.

### 3.4. Workstream training mới

- Training Phase 0 đã hoàn thành audit và được chấp nhận với kết luận BLOCKED BY DATA.
- Đã có [hướng dẫn dataset/training](Training_Instructions.md) và tài liệu cấu trúc project `docs/project_structure.md` (file local, chưa được đưa vào commit hai tài liệu này).
- Chưa bắt đầu Training Phase 1, chưa tạo reviewed dataset mới hoặc train supervised model mới.
- Chưa có review records/masks thật để kết luận 14 NG hiện hữu đã được xác minh.

## 4. Bằng chứng dữ liệu tại thời điểm hiện tại

Audit chỉ đọc được chạy lại ngày 2026-10-17; không chạy inference trên historical locked data.

| Hạng mục | Hiện trạng | Ý nghĩa |
|---|---:|---|
| File dưới Data | 645 | Có hai ZIP; audit cũ là 643 file, không phải đã thêm nhãn mới |
| Raster files | 634 | Bao gồm ảnh, mask, bản sao và montage, không phải 634 capture độc lập |
| Good source files được repo khai báo real | 63 | 36 Fabric A, 27 Fabric B; chưa có human defect-free attestations |
| NG source files chưa phân loại | 14 | Chưa có class/fabric/mask/provenance được review đầy đủ |
| Good crop/biến thể từ nguồn thật | 130 | Không phải 130 capture mới độc lập |
| Synthetic samples | 220 | 80 Good + 140 defect, giữ tách khỏi real evidence |
| Synthetic masks | 140 | Hợp lệ binary geometry không có nghĩa là nhãn thật/semantic accuracy |
| Verified real defect masks tìm thấy trong phạm vi audit | **0** | Blocker chính cho supervised segmentation |
| Bbox-only annotation được nhận diện dưới Data | 0 | Crop boxes/generator notes/model predictions không phải human bbox ground truth |
| Missing roll_id/lot_id/camera_id trong canonical rows | 350/350 cho mỗi trường | Không thể tuyên bố roll/lot/camera-disjoint |
| Duplicate-byte groups | 122 | Mỗi nhóm hai file trong snapshot; cần quản lý source lineage |
| Cross-catalog source conflicts | 35 nguồn | Rủi ro nếu ghép catalog cũ rồi chia train/test mới |
| External image paths không hiện diện | 10.198 | Không được tính thành số mẫu local đã sẵn sàng |

Các class hiện có số lượng đáng kể chủ yếu là **synthetic labels**. Chưa có defect class nào được chứng minh đủ nhãn pixel thật để chốt taxonomy supervised training.

Audit hiện nhận diện manifest/schema đang có, không phải bộ chứng nhận annotation tổng quát. Số “verified = 0” là không tìm thấy bằng chứng đủ điều kiện trong phạm vi đã kiểm tra, không phải kết luận bằng mắt rằng mọi ảnh đều không thể dùng. Chủ dữ liệu vẫn phải review thủ công.

## 5. Vì sao chấm MVP 5/10?

### Những phần đã giúp tiến gần MVP

1. Không phải bắt đầu lại backend/UI/storage/anomaly pipeline.
2. Có versioned fabric-specific reference design và fail-safe khi không có reference hợp lệ.
3. Có kiến trúc segmentation độc lập, dataset/trainer/checkpoint infrastructure để tái sử dụng.
4. Có tests và công cụ local diagnostics, export/parity, audit để giảm lỗi triển khai.
5. Có governance chặn việc lấy fixture hoặc candidate rejected làm demo sản phẩm giả.

### Những phần ngăn một MVP thực sự hữu dụng

1. Người dùng hiện không thể chọn hai fabric INACTIVE để scan bình thường; app chạy được không đồng nghĩa feature inspection đã mở cho nghiệp vụ.
2. Chưa có real reviewed labels/provenance để train và validate supervised candidate.
3. Chưa có bằng chứng model/reference hiện hữu đạt mục tiêu phát hiện lỗi/overkill trên ảnh thật của scope đã chọn.
4. Chưa có kết quả so sánh real-data anomaly-only / supervised-only / hybrid trên cùng development set.
5. UI hiện chủ yếu upload, bbox và anomaly results; chưa có trained supervised mask visualization/local hybrid workflow đã kiểm chứng.

**Điểm 5/10 phản ánh ứng dụng khung đã khá sâu nhưng vòng giá trị cốt lõi trên dữ liệu thật chưa khép kín.** Nếu mục tiêu chỉ là trình diễn phần mềm/fixtures cho đội kỹ thuật, nền tảng có thể xem là 8/10; không được giới thiệu demo đó là model kiểm lỗi thật đã tốt.

Để tiến gần mức 8/10 cho MVP: cần reviewed real dataset, một candidate thực sự reload/infer được, validation có coverage theo fabric, UI thể hiện đúng scope/shadow và một đợt thử nghiệm có người giám sát. Không gán trước recall/FPR hoặc thời hạn hoàn thành khi chưa có dữ liệu và criteria.

## 6. Vì sao production mới 2/10?

Phần còn thiếu không chỉ là “thêm một ít code”:

- **Model/data:** chưa có trained/versioned candidate hợp lệ, verified real pixel labels và chứng minh generalization.
- **Acceptance:** chưa có fresh disjoint locked test, owner-approved criteria, full-hybrid evidence hoặc independent approval thật.
- **Performance:** Phase 17 không đo full pipeline camera → preprocess → segmentation + anomaly/FAISS → fusion/QC → DB/API/UI. Chưa có long-running soak, contention, thermal/memory stability hoặc load evidence đại diện vận hành.
- **Machine vision:** line-scan hiện là simulation, không có bằng chứng capture/trigger/encoder/calibration/lighting thật. PLC/MES/rejector chưa được xác nhận nếu scope cần tích hợp đó.
- **Security:** chưa có authentication/RBAC rõ ràng ở các route đã inspect; app còn local/debug mode. Upload/annotated images được serve qua static routes, cần access-control/retention khi dữ liệu khách hàng được chia sẻ ra mạng.
- **Deployment/operations:** chưa thấy bộ CI/CD/deployment manifests/runbooks production trong tracked scope đã kiểm tra; chưa có bằng chứng restore drills, incident handling, model/service rollback diễn tập hoặc monitoring/drift loop thực tế.
- **Governance trust:** approval/custody hiện dựa vào khai báo và kiểm soát bên ngoài; framework chưa thay thế identity/access-control và independent test custody thực.

Có foundation để phát triển tiếp, nhưng chưa đủ để chịu trách nhiệm cho quyết định QC trong môi trường vận hành thật. **Không nên deploy public hoặc để hệ thống tự loại/chấp nhận hàng chỉ dựa trên snapshot hiện tại.** Đây là đánh giá theo repo/evidence, không phải một security audit hoặc factory acceptance hoàn chỉnh.

## 7. Ưu điểm của dự án tính đến hiện tại

| Ưu điểm | Giá trị thực tế | Giới hạn phải giữ rõ |
|---|---|---|
| Phân lớp API/core/AI/services/DB tương đối rõ | Có thể phát triển/test từng thành phần, không cần rewrite toàn bộ | Độ phức tạp tăng khi ghép nhiều research branches |
| Giữ anomaly encoder và segmentation encoder độc lập | Tránh segmentation fine-tuning làm mất compatibility FAISS memory | Chưa đo đầy đủ tài nguyên hai nhánh cùng workload |
| Fabric-specific memory/reference, no cross-fabric fallback | Giảm nguy cơ áp nhầm reference vật liệu | Vẫn cần reference đạt chất lượng thật |
| Fail-safe INACTIVE/NEEDS_REVIEW, flags mặc định OFF | Không silently dùng candidate cũ/rejected hoặc tự đổi QC | User workflow bị chặn đúng lúc chưa đủ evidence |
| Có tests rộng và hồi quy qua nhiều phase | Có cơ sở bảo toàn contracts và xử lý edge cases | Số test không phải coverage %, defect accuracy hoặc production reliability |
| Có hash/version/checkpoint/manifest discipline | Tăng khả năng truy vết, phát hiện stale/tampered artifacts | Hash không xác minh nhãn đúng hoặc provenance khai thật |
| Có ONNX/TensorRT FP32 thực nghiệm thật | Đã giải quyết một phần tính khả thi export/runtime GPU | Probe untrained; phải chạy lại trên candidate thật |
| Từ chối FP16 khi parity không đạt | Ưu tiên đúng semantics, không đổi chất lượng lấy số latency đẹp | FP16 vẫn không eligible, không fallback |
| Governance tách report khỏi activation | Ngăn tự động ACTIVE chỉ vì một báo cáo có accepted=true | Chưa có production approval thật và identity system hoàn chỉnh |
| Audit không che giấu synthetic/missing data | Tránh xây confidence sai từ số lượng crop/fixture | Vẫn cần công sức thu thập/review dữ liệu thật |

## 8. Nhược điểm, rủi ro và hướng giải quyết từng điểm

Ưu tiên dưới đây là thứ tự khuyến nghị để review, không phải authorization triển khai. P0 = chặn dữ liệu/giá trị/release; P1 = cần cho MVP pilot đáng tin; P2 = production hardening theo scope.

| Vấn đề | Tác động hiện tại | Hướng giải quyết | Bằng chứng cần có để đóng điểm |
|---|---|---|---|
| **P0 — Thiếu nhãn pixel thật** | Không thể train supervised segmentation hợp lệ | Review 14 NG và Good; annotate/review mask thật; thu thêm BAD nếu chưa đủ coverage | Review log, annotator/reviewer, mask version/hash, real sample inventory |
| **P0 — Thiếu provenance** | Không xác nhận disjointness, dễ leakage | Truy nguồn roll/lot/camera/session/time; không biết để null và cách ly khỏi protocol đòi đủ metadata | Metadata có nguồn xác thực, ancestry/hash inventory; không dummy IDs |
| **P0 — Candidate chất lượng thật chưa có** | Chưa cung cấp giá trị detection được chứng minh | Sau approval, train candidate mới độc lập, reload/checkpoint validation; không dùng probe Phase 17 | Checkpoint/version, manifest/config hashes, training history và genuine validation results |
| **P0 — Validation coverage/metric protocol chưa chốt** | Dễ overfit hoặc chấm sai do imbalance | Review scope hai fabric, classes, nhóm capture và metric definitions; tách pixel/image/event metrics | Per-fabric/class/group coverage, lỗi âm/dương được review; criteria do owner duyệt |
| **P0 — Chưa fresh locked test** | Không có production acceptance evidence | Custodian thu capture mới độc lập với toàn training/reference/calibration/postmortem history | Real labels, complete provenance, custody, source/roll/lot/camera disjoint checks |
| **P0 — Criteria/SLA chưa được owner duyệt** | Không thể kết luận đạt/không đạt nghiệp vụ | Owner xác định chi phí underkill/overkill, mục tiêu và protocol; giữ null khi chưa duyệt | Policy version/hash/approval; không dùng con số benchmark làm SLA |
| **P1 — Hai fabric INACTIVE** | UI không mở inspection nghiệp vụ | Giữ nguyên gate; chuẩn bị candidate/evidence và xin đúng approval, không bật reference cũ để demo | Qualified scope/candidate; mọi activation tương lai phải qua governance và manual approval |
| **P1 — Research modules chưa thành vòng local hoàn chỉnh trên model thật** | Có component nhưng chưa có validated end-to-end user workflow | Sau các phase được duyệt, checkpoint → shadow → local API → UI → development comparison | Real-image input/output/overlay/versions, failure cases; anomaly behavior không bị phá |
| **P1 — Metric/artifact trainer còn khoảng trống so với workstream mới** | Không nên nói trainer đã đáp ứng mọi output mong muốn | Bổ sung trong phase đã duyệt: per-fabric reports, LR/F1/history/provenance packaging nếu yêu cầu; chốt selection criterion | Artifact schema và test thực; hiện trainer chọn validation loss, không phải mean Dice |
| **P1 — GPU training env chưa sẵn sàng** | TensorRT env hoạt động không đồng nghĩa train được | Setup môi trường training riêng sau approval; kiểm tra torch/torchvision/CUDA/cache/smoke | Môi trường training tương thích, không sửa CPU env hoặc driver; hiện GPU env thiếu torchvision |
| **P1 — Chưa full-hybrid runtime/VRAM evidence** | Không suy được capacity hay P95 thực tế | Đo candidate thật, batch 1 FP32 trước, toàn đường chạy cần dùng và competing resources | Protocol/raw samples P50/P95/throughput/VRAM có scope rõ; không segmentation-only thay full latency |
| **P1 — Tài liệu onboarding bị lệch trạng thái** | Người mới có thể chạy nhầm workflow legacy | Cập nhật README: bỏ mô tả Current Phase 1, sửa contract fabric_id và lệnh legacy activation đã bị disable | Một entry point hiện hành, lệnh được kiểm chứng; giữ historical docs có nhãn lịch sử |
| **P1 — Inventory tests pin snapshot cũ** | Thêm ZIP/dataset mới có thể làm test fail do số file, không phải model regression | Review và tách invariants khỏi inventory snapshot; version expected inventory khi data thay đổi hợp lệ | New snapshot reviewed + invariants không bị nới để che mất dữ liệu/leakage |
| **P2 — Auth/access control chưa đủ** | Không phù hợp exposed service hoặc dữ liệu khách hàng | Thiết kế authentication/RBAC, quyền build/activate/download, retention và audit theo scope | Threat model, permission tests và kiểm tra truy cập image artifacts |
| **P2 — Camera/lighting/calibration thật chưa xác nhận** | Sai hình học/sync/ánh sáng có thể phá model dù offline tốt | Nếu inline: pilot acquisition/encoder/camera/lighting thật, calibration và seam/event matching | Đo sai số hình học, missing/duplicate frames, scan coverage và sync trên phần cứng |
| **P2 — Recovery/soak/concurrency chưa có evidence** | Khó bảo đảm chạy lâu, lỗi tài nguyên và khôi phục | Load/soak/failure-injection, timeout/retry/backpressure phù hợp, backup/restore/rollback drills | Log đo được, recovery evidence, quy trình operator; không suy từ unit fixtures |
| **P2 — MLOps/monitoring vòng đời còn thiếu** | Sau release dễ bỏ sót drift và regression | Versioned build/deploy, CI theo scope, artifact trust, drift/quality monitoring và response owner | Reproducible installation/build, staged rollout, audit và alert/review loop |

### Điểm yếu mang tính chiến lược

Project đã đầu tư nhiều vào architecture/optimization trước khi có dữ liệu thật đủ điều kiện. Điều đó tạo foundation tốt nhưng khiến tiến độ “đã có nhiều module” vượt xa tiến độ “đã chứng minh giải quyết bài toán”. Rủi ro tiếp theo là tiếp tục thêm model/framework/hardware optimization trong khi bottleneck chính vẫn là real labels, provenance và validation.

Hướng khuyến nghị: **tạm giữ kiến trúc hiện tại, ưu tiên data → candidate → evidence → controlled MVP**. Không có cơ sở yêu cầu rewrite, đổi sang kiến trúc phức tạp hơn hoặc mua GPU mới ở thời điểm này; RTX 3050 4 GB cần đo workload thật, không mặc định đủ/không đủ cho mọi cấu hình.

## 9. Những nội dung chưa được phép tuyên bố đã hoàn thiện

- Chưa có model segmentation được train và kiểm chứng hợp lệ cho hai fabric thật.
- Chưa chứng minh giảm false positives của Fabric A hoặc giữ/tăng recall trên Fabric B.
- Chưa có real-world segmentation IoU/Dice, unknown-defect recall hoặc per-100 m quality evidence đủ điều kiện acceptance.
- Chưa chứng minh hybrid tốt hơn từng nhánh đơn lẻ.
- Chưa có full hybrid factory P50/P95/throughput/SLA hoặc concurrent dual-model VRAM evidence.
- Chưa có fresh disjoint locked-test acceptance và independent approval thật.
- Chưa có factory camera/encoder/lighting/PLC loop được xác nhận nếu triển khai inline.
- Chưa có cơ sở gọi hệ thống là autonomous production QC hoặc bảo đảm toàn bộ defect classes.

Không phải mọi artifact trong storage đều “chưa từng train”: project có các projection/adapter artifacts cũ. Điểm thiếu cụ thể là **candidate hiện tại đủ điều kiện cho mục tiêu và evidence đang yêu cầu**, không phải phủ nhận toàn bộ công việc ML trước đây.

## 10. Kiểm thử và giới hạn của bằng chứng

| Bằng chứng | Kết quả đã ghi nhận | Có thể kết luận | Không được kết luận |
|---|---|---|---|
| Phase 18 full regression | 1.920 passed, 1 warning; frontend build PASS; 77 smoke assertions | Software/contracts/fixtures ở snapshot đó đã pass | Model accuracy, coverage %, factory reliability |
| Lượt kiểm tra trước push 18 phase | 1.920 passed, 1 warning; frontend/smoke PASS | Architecture suite trước push đã pass, tách Training Phase 0 khỏi scope lượt đó | Toàn working tree hiện tại chắc chắn pass mọi test hôm nay |
| Training Phase 0 report | 18 unit + 6 integration; full 1.944 passed, 1 warning tại snapshot Data 643 file | Audit implementation đã có lượt test thành công, gồm 24 test mới | Snapshot Data hiện tại 645 file chắc chắn khớp mọi assertion cũ |
| Phase 17 GPU measurement | Full-architecture untrained probe FP32 parity PASS; FP16 REJECTED | Khả thi kỹ thuật của workload research đã đo | Trained defect quality, production benchmark, full-hybrid latency |
| Audit dữ liệu chạy lại 2026-10-17 | BLOCKED BY DATA; 645 files, 140 synthetic masks, 0 verified real masks | Current inventory được audit nhận diện vẫn chưa đủ nhãn thật | Đã review ảnh bằng mắt hoặc chứng nhận mọi annotation có thể có |
| Governance preflight chạy lại 2026-10-17 | BLOCKED BY DATA / TRAINED CANDIDATE, accepted=false; metrics null; hai fabric INACTIVE | Current project chưa đủ acceptance inputs | Đã chạy real candidate evaluation hoặc activation |
| Settings/environment inspection 2026-10-17 | Local/debug; feature flags kiểm tra đang OFF; GPU env thiếu torchvision | Nhiều implementation chưa bật trong runtime hiện tại, train env chưa được xác nhận | Full backend đã migrate GPU hoặc mọi registry override tương lai đã được audit |

**Lượt viết Overall.md không chạy lại full regression, frontend build, inference hoặc benchmark.** Chỉ đọc code/report/settings, chạy audit/preflight read-only và kiểm tra metadata môi trường. Các số test lịch sử được ghi đúng phạm vi, không trình bày như test mới hôm nay.

### Scope đúng của Phase 17

FP32 được chấp nhận là technical/research baseline. Benchmark candidate có pretrained encoder và decoder/head untrained, **không phải model fabric-defect trained**. Các measurements chỉ segmentation architecture; không kế thừa thành full-hybrid hoặc production performance.

FP16 vẫn REJECTED, không enable/promote/fallback. Probe không được đăng ký hoặc activate, kể cả đổi tên artifact. Candidate train thật phải export/parity/benchmark lại bằng protocol được duyệt.

## 11. Roadmap đề xuất để khép khoảng cách

Đây là phương hướng, **không bắt đầu hoặc tự approve phase nào**. Không đưa deadline vì khối lượng annotation và chất lượng dữ liệu mới chưa rõ.

| Thứ tự | Công việc | Chủ trì phù hợp | Điều kiện bàn giao |
|---|---|---|---|
| 1 | Review 14 NG và Good; xác nhận hai fabric mục tiêu | Domain QC/data owner | Review records, nhãn/fabric/orientation/suitability thực |
| 2 | Pixel annotation + review + provenance/capture inventory | Annotator, reviewer, acquisition owner | Real masks, Good attestations, metadata có căn cứ, các điểm chưa biết được giữ rõ |
| 3 | Xin duyệt Training Phase 1; tạo dataset train/val source-grouped | ML/data engineer + data owner | Coverage/integrity/leakage tests và manifest version; dừng review |
| 4 | Sau approval riêng: train/version/checkpoint validation | ML engineer | Genuine train/val history, reload/CPU-CUDA checks, real validation previews; không production |
| 5 | Shadow local inference/API/UI và development comparison | ML/backend/frontend + QC reviewer | Các ảnh/test cases thật, per-fabric failures; anomaly-only/supervised-only/hybrid report |
| 6 | Controlled MVP có người giám sát trong scope được duyệt | Product owner + QC/operator | Người dùng kiểm tra được input/output và limitations; không tự tạo production ACTIVE |
| 7 | Candidate thật export/parity/performance và runtime hardening | Runtime/MLOps/vision engineers | Full intended runtime evidence; phần cứng/soak/security/recovery phù hợp scope |
| 8 | Owner criteria → fresh locked custody/evaluation → independent manual review | Owner, custodian, evaluator, independent approver | Acceptance report thật; chỉ sau đó mới xem xét activation qua Phase 18 |

Có thể thu hẹp MVP nếu dữ liệu chỉ đủ một phần scope, nhưng phải được owner duyệt lại phạm vi rõ ràng; không tự bỏ một fabric, bỏ supervised branch hoặc bỏ release gate. Không cần hoàn thành tất cả khả năng inline công nghiệp để đánh giá một local MVP giới hạn, nhưng cũng không được gọi local MVP đó là factory production.

## 12. Tiến độ so với yêu cầu

| Nhóm yêu cầu | Đã làm được | Còn thiếu | Kết luận |
|---|---|---|---|
| 18 phase architecture/software | Có deliverables và report; đã push main | Không thay thế real acceptance hoặc integration thực tế | Hoàn thành phạm vi implementation đã duyệt |
| Local anomaly app | Backend/UI/storage/FAISS và guards có sẵn | Candidate/reference đủ evidence để mở đúng scope | Foundation tốt, inspection nghiệp vụ đang bị chặn an toàn |
| Supervised fabric-defect CNN | Architecture/dataset/trainer/shadow/export infra | Reviewed labels, train thật, checkpoint validation, local inference evidence | Chưa hoàn thành mục tiêu trained model thật |
| Hybrid inspection | Fusion/QC contracts và opt-in integration | Real branch comparison, calibrated policy, full local/runtime evidence | Infrastructure có, giá trị thực chưa chứng minh |
| Machine vision/inline | Physical contracts, simulator, seam/scheduler tools | Real acquisition/calibration/sync/integration/soak | Research/simulation, chưa factory-ready |
| Acceptance/release | Governance software, null policy, manual adapter | Candidate/fresh test/owner criteria/independent approval | BLOCKED BY DATA / TRAINED CANDIDATE |
| Training workstream | Phase 0 audit được duyệt, hướng dẫn đã viết | Data preparation và review set trước Phase 1 | DATA PREPARATION, chưa train |
| MVP | Nền tảng kỹ thuật khoảng 8/10 | Vòng giá trị ảnh thật/candidate/evidence chưa khép | **5/10 — chưa ready cho sử dụng nghiệp vụ độc lập** |
| Production | Có một số foundations quan trọng | Data/model/acceptance/full runtime/operations gates | **2/10 — không production-approved** |

## 13. Nguồn đối chiếu nội bộ

- [README.md](README.md): đã đối chiếu nhưng có nội dung legacy lệch hiện trạng, không dùng làm nguồn duy nhất.
- Cấu trúc project: `docs/project_structure.md` (file local, chưa được đưa vào commit này).
- [Training instructions](Training_Instructions.md).
- Training Phase 0 data readiness: `docs/training/data_readiness_report.md` (file local, chưa được đưa vào commit này).
- [Phase 16 ONNX](docs/implementation/phase_16_onnx_export.md).
- [Phase 17 TensorRT](docs/implementation/phase_17_tensorrt_optimization.md).
- [Phase 18 governance](docs/implementation/phase_18_release_governance.md), [machine-readable report](docs/implementation/phase_18_release_report.json).
- Các báo cáo Phase 1–15 dưới `docs/implementation/` và code tương ứng trong `backend/`, `scripts/`, `tests/`.
- Code kiểm tra trực tiếp: `backend/main.py`, `backend/app/core/config.py`, inspection/reference routes và services, `frontend/src/App.tsx`, `ReferenceList.tsx`, `InspectionBay.tsx`, trainer/metrics và các CLI audit.
- Audit/preflight trực tiếp: `scripts/audit_training_data.py --summary`, `scripts/check_phase18_release.py --audit-current`.

## 14. Khuyến nghị cuối cùng

**Không cần bắt đầu lại dự án. Cần đổi trọng tâm từ “thêm implementation” sang “đóng khoảng trống evidence trên dữ liệu thật”.**

Giữ nền tảng đã có; ưu tiên reviewed real data và provenance, sau đó xin duyệt từng bước để tạo candidate thật, kiểm chứng local MVP và cuối cùng acceptance production. Không dùng số phase, số test, pretrained encoder hoặc tốc độ probe để thay thế chất lượng phát hiện lỗi.

Trạng thái giữ nguyên: **DATA PREPARATION** cho training; **BLOCKED BY DATA / TRAINED CANDIDATE** cho release. Hai fabric INACTIVE, FP16 REJECTED, acceptance metrics còn null. Lần đánh giá này chỉ tạo `Overall.md`, không sửa code/data/registry hoặc tự triển khai bất kỳ roadmap nào.
