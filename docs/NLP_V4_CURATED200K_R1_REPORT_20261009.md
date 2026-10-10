# Báo cáo thực nghiệm NLP V4 — Curated 200K / Fine-tune R1

**Dự án:** Phenikaa Campus Courier 2026 (COURIER2)
**Ngày:** 09/10/2026 (UTC+7)
**Run:** NLP_V4_Curated200K_Finetune_R1
**Trạng thái:** Train hoàn tất, đã so sánh offline, chưa nộp V4.
**Bản public tốt nhất đã biết:** submission #1107 — 68,78% (NLP V2 ensemble, goal threshold 0.8, via threshold 0.95, solver train+validation).

## 1. Kết luận điều hành

- Dữ liệu curated 200k đem lại **cải thiện goal trên holdout tổng hợp**, nhưng fine-tune V4 một seed **làm giảm via trên validation thật**.
- Cùng 300 câu validation, so V2 seed 0 và V4 fine-tuned seed 0: neural goal 90,00% → 93,67% (+3,67 điểm); neural via 89,33% → 82,33% (−7,00 điểm); neural all 64,33% → 63,00%.
- Cùng 2.000 câu holdout tổng hợp, V4 đạt neural all 55,70%, cao hơn V2 seed 0 (44,05%) và V2 ensemble (47,30%). Kết quả này KHÔNG dự báo chắc chắn public/private.
- Cùng ngưỡng hybrid goal=0.8, via=0.95, flags=2.0 trên validation: V2 ensemble 98,00% all; V2 seed 0 89,67%; V4 85,00%. **Không có cơ sở thay toàn bộ V2 bằng V4 ngay**.
- Vấn đề cụ thể: với 116 câu validation có via kiểu named, V4 nhận nhầm thành không có via **31 câu**, V2 seed 0 chỉ **10 câu**.
- Không đủ căn cứ quy lỗi cho chất lượng corpus nói chung; phương pháp fine-tune và độ lệch phân bố có khả năng góp phần vào sự đánh đổi goal/via.

## 2. Corpus, baseline và giới hạn đo

**Corpus:** results/nlp_v4_data_350k/replacement_20261009_review/nlp_v4_curated_200000_quality_v2_reindexed.jsonl

- 200.000 dòng JSONL đã qua kiểm tra cấu trúc, nhãn và trùng lặp.
- SHA-256 đã kiểm tra trong đợt review: 4c2c54e46adedbd91e73d80b87ea77daa6b03a3b3b31c011fa6dab4f2654670e.
- Goal anchor_near: 29,61% corpus so với 41,67% validation.
- Không có via: 49,94% corpus so với 48,00% validation.
- Kiểm tra hợp lệ kỹ thuật không đồng nghĩa 100% nhãn đúng ngữ nghĩa.
- **Không dùng dữ liệu test để huấn luyện hay phân tích ngôn ngữ phục vụ training.**

**NLP V2:** artifacts/nlp/neural_parser_v2.pt. Metadata kiểm tra: 3 seed, 15 epoch/seed, synthetic 200.000 câu sinh mới mỗi epoch (fixed_synthetic=False), LR 2e-3, batch 128, embedding 96, hidden 192, real-repeat 5. Kiến trúc 7.515.212 tham số/mạng. NLP dùng validation để chọn checkpoint và ngưỡng, không trực tiếp đưa validation vào gradient theo trainer đã đọc. **Solver của #1107 fit trên train+validation**, vì thế validation end-to-end không độc lập hoàn toàn.

## 3. Diễn biến huấn luyện thực tế

### 3.1. Pilot ban đầu

Thư mục: results/NLP_V4_Curated200K_Finetune_R1/pilot_LR1e4

- Fine-tune từ mạng số 0 của NLP V2, seed 20261009, corpus curated cố định qua các epoch, real-repeat 5, real-augment 1, batch 128, LR 1e-4, dim 96, hidden 192.
- Epoch 1 hoàn thành sau 717 giây, checkpoint được lưu lúc 18:27:28.
- Epoch 2 kéo dài bất thường; Windows ghi nhận chuyển đổi nguồn AC nhiều lần khoảng 18:39:50–18:40:19. Không xác định chắc nguyên nhân chậm (tiến trình vẫn tiêu thụ CPU/GPU).
- Tiến trình PID 22732 **bị dừng thủ công khi chưa chứng minh được treo**, khoảng 19:03. Phần cập nhật chưa lưu của epoch 2 mất; checkpoint epoch 1 được giữ nguyên. Đây là sai sót trong điều hành training.

### 3.2. Fine-tune từ checkpoint pilot epoch 1

Thư mục: results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5

- Nạp lại trọng số cuối pilot epoch 1; **không train lại từ đầu**.
- Batch 64, LR 5e-5, 3 epoch bổ sung, seed 20261009; optimizer và OneCycleLR **khởi tạo mới**, do checkpoint chỉ chứa trọng số.
- Chạy CUDA trên RTX 3050 Laptop 4 GiB; ba epoch mất 782, 755 và 781 giây.
- Hoàn thành và lưu model/report lúc khoảng 19:46 ngày 09/10; stderr trống.
- **Checkpoint tốt nhất: epoch bổ sung 2**, do điểm validation all + holdout all đạt cao nhất.

## 4. Kết quả từng epoch

### 4.1. Pilot

| Chỉ số | Pilot epoch 1 |
|---|---:|
| Loss | 0,6919 |
| Validation goal | 94,00% |
| Validation via | 82,33% |
| Validation all | 63,00% |
| Holdout goal | 74,20% |
| Holdout via | 83,00% |
| Holdout all | 53,95% |

### 4.2. Fine-tune bổ sung

| Chỉ số | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Loss | 0,0465 | 0,0271 | 0,0222 |
| Validation goal | 94,67% | 93,67% | 93,67% |
| Validation via | 82,33% | 82,33% | 82,33% |
| Validation all | 63,33% | 63,00% | 63,00% |
| Holdout goal | 74,45% | 75,15% | 75,10% |
| Holdout all | 54,60% | 55,70% | 55,40% |
| Thời gian | 782 s | 755 s | 781 s |

Checkpoint epoch 2: validation goal 93,67%; validation via 82,33%; holdout goal 75,15%; holdout via 83,65%; holdout all 55,70%.

**Diễn giải:** loss giảm mạnh nhưng holdout chỉ nhích lên rồi giảm 0,30 điểm ở epoch cuối. Đây là dấu hiệu lợi ích biên chững lại, có rủi ro overfitting; không đủ bằng chứng kết luận overfit nghiêm trọng.

## 5. So sánh công bằng trên cùng bộ câu

Đánh giá bằng các class NLP có sẵn, chỉ đọc model và dữ liệu, không sửa source/checkpoint. Tham số hybrid cố định goal=0.8, via=0.95, flags=2.0 (khác với ngưỡng tự chọn mặc định).

### 5.1. Validation thật, 300 câu

| Chỉ số | V2 seed 0 | V2 ensemble (3 seed) | V4 best |
|---|---:|---:|---:|
| Neural goal | 90,00% | 91,67% | 93,67% |
| Neural via | 89,33% | 92,33% | 82,33% |
| Neural all | 64,33% | 68,33% | 63,00% |
| Hybrid goal | 94,67% | 99,33% | 97,33% |
| Hybrid via | 94,33% | 98,67% | 86,00% |
| Hybrid all | 89,67% | 98,00% | 85,00% |

Kể cả khi loại bỏ ưu thế ensemble, V4 vẫn giảm via rõ rệt so với V2 seed 0.

### 5.2. Holdout tổng hợp, cùng 2.000 câu (seed 20261010)

| Chỉ số | V2 seed 0 | V2 ensemble | V4 best |
|---|---:|---:|---:|
| Neural goal | 62,35% | 66,95% | 75,15% |
| Neural via | 81,40% | 88,60% | 83,65% |
| Neural all | 44,05% | 47,30% | 55,70% |
| Hybrid all | 43,65% | 41,45% | 53,35% |

**Giới hạn:** Đây là holdout tổng hợp, có thể chia sẻ cách sinh/ngữ pháp với corpus curated, không phải một phép đo public/private độc lập. Không diễn giải tăng holdout thành tăng leaderboard.

### 5.3. Phân rã lỗi via trên validation

Có 144/300 câu không có via và 116/300 câu có via kiểu named.

| Quan sát | V2 seed 0 | V4 best |
|---|---:|---:|
| Nhận đúng không có via | 144/144 | 144/144 |
| Có via named nhưng đoán không có via | 10 | 31 |
| Đúng hoàn toàn via named | 99/116 | 80/116 |

Dấu hiệu nổi bật: V4 **quá thường xuyên bỏ sót điểm ghé thật**, không phải thường xuyên thêm via giả. Cần nghiên cứu nhãn, phrasing và cơ chế chọn none-via trước khi kết luận nguyên nhân.

## 6. Ngưỡng hybrid, báo cáo trainer và quyết định submission

Trainer tự chọn HybridThresholds(goal=2.0, via=2.0, flags=2.0) vì parser luật đạt validation/challenge 100%. Các ngưỡng này gần như **vô hiệu hóa neural override**. Do đó:

- Validation hybrid all=100% trong report.json **không phản ánh lợi ích neural trên câu lạ**.
- Holdout hybrid all theo ngưỡng tự chọn chỉ 19,87%, kém neural thuần.
- Kết quả so sánh thực tế ở mục 5 đã cố định goal=0.8, via=0.95, flags=2.0 để phản ánh hướng đã dùng ở #1107.
- Chưa chạy full CV → NLP V4 → solver, chưa tạo/nộp submission V4, và chưa có public score V4.

**Quyết định đề xuất:** Giữ submission #1107 68,78% làm chuẩn; chưa thay bằng V4 nguyên khối dựa trên kết quả hiện tại. Không khẳng định V4 tốt hơn hoặc tệ hơn trên private khi chưa nộp.

## 7. Bài học và nghiên cứu tiếp (chưa được thực thi)

1. Corpus 200k **có giá trị về khả năng tổng quát hóa goal trên holdout**, nhưng cách fine-tune hiện tại làm giảm via trên validation.
2. Điểm yếu lớn nhất: via named bị dự đoán thành không có via ở 31 câu. Cần kiểm tra lại độ phủ cụm từ ghé qua và câu nhiều mệnh đề trong corpus.
3. Giữ kiến trúc, mô hình V2 và checkpoint V4; chỉ thực hiện các phép so sánh công bằng trên cùng tập đánh giá.
4. Có thể **nghiên cứu** giữ phần via của V2 và sử dụng goal của V4 (chưa sửa pipeline, chưa đánh giá end-to-end), hoặc thí nghiệm đối chứng theo cách V2 sinh câu mới mỗi epoch và ensemble nhiều seed.
5. Trước khi đổi cấu hình train, dừng/restart tiến trình, chỉnh pipeline, tạo file nộp hoặc nộp, cần **xác nhận riêng của người dùng**.
6. Không chỉnh sửa mã nguồn dự án trong đợt lập báo cáo này.

## 8. File nguồn để kiểm chứng

**Dữ liệu:** results/nlp_v4_data_350k/replacement_20261009_review/nlp_v4_curated_200000_quality_v2_reindexed.jsonl

**Pilot:** results/NLP_V4_Curated200K_Finetune_R1/pilot_LR1e4/train.log

**Pilot checkpoint:** results/NLP_V4_Curated200K_Finetune_R1/pilot_LR1e4/checkpoints/seed_20261009_best.pt

**Fine-tune log:** results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/train.log

**Checkpoint tốt nhất (epoch bổ sung 2):** results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/checkpoints/seed_20261009_best.pt

**Model được lưu:** results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/model.pt

**Báo cáo máy:** results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/report.json

**Baseline V2:** artifacts/nlp/neural_parser_v2.pt

**Cấu hình public #1107:** _incoming_gui_codex_v4_readonly/DOC_TRUOC.md

---
Báo cáo được tạo mới dưới docs/. Không sửa nguồn, checkpoint, artifact V2 hoặc submission.
