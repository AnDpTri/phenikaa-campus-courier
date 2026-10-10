# Báo cáo nghiên cứu và cải tiến dữ liệu Semantic Fusion — 10/10/2026

## Mục tiêu và giới hạn
Nghiên cứu nguyên nhân overfit và làm một bộ dữ liệu ứng viên mới với ngữ nghĩa đối chứng. Chỉ tạo file trong `D:\phenikaa\results\NLP_Data_Research_20261010_a`. Không can thiệp training 3×5 epoch, source gốc, checkpoint hay official validation/test. **Chưa huấn luyện mô hình mới trên dữ liệu cải tiến**, vì thế chưa có bằng chứng tăng hiệu năng sau training.

## Phát hiện từ dữ liệu và checkpoint cũ
1. **Template/lexical shortcut:** 200K câu train nhưng khoảng 91,910 tiền tố 8-token khác nhau; giới hạn tiền tố không bảo đảm khác cấu trúc ngữ nghĩa. `via=True` train 56.99% (riêng recipe `pickup_reasoning` 69.95%) so với V2/weighted/hard synthetic probes khoảng 51–52%. Các cue `lay`, `nhan`, `truoc`, `sau` có quan hệ nhãn không ổn định giữa bộ sinh và probe.
2. **Thiếu khái quát `fragile`:** trên 750 hard probe, checkpoint Scratch cũ E1/E2 đoán không có fragile cho cả 333/333 mẫu positive. Chỉ thay một cụm từ tương đương bằng cách diễn đạt đã gặp, fragile accuracy từ 55.6% lên 100%, all exact E2 từ 31.20% lên 52.80%. Đây là stress test counterfactual, không phải bằng chứng real-world.
3. **Khó khăn suy luận trạng thái `via`:** dữ liệu cũ có active, cancelled, replaced nhưng các tổ hợp cùng câu chưa cân xứng. Trong thí nghiệm 18 vòng trên các cặp positive/cancelled, model E2 cải thiện nhiều ở loại cancelled nhưng vẫn bỏ sót actual pickup; không nên lấy train loss gần 0 làm dấu hiệu tổng quát hóa.
4. **Mức độ khó khác nhau theo family:** trên tập ứng viên V5 mới, các frozen model cũ có all exact chỉ 12.9–17.55% trên 2.000 câu của 6 family; trên 2 family khác 28.1–44.65%. Vì **chưa model nào được train bằng ứng viên này**, chênh lệch là minh chứng về sự phụ thuộc cấu trúc/độ khó family, **không phải train-vs-validation gap**. Kiểm nghiệm chi tiết tại `checkpoints\semantic_v5_frozen_model_diagnosis.json`.

## Dữ liệu mới đã tạo
### V5 thử nghiệm
- `candidate_semantic_v5.py`, 8.000 câu tạo từ 1.000 cảnh × 8 tổ hợp `via/urgent/fragile`.
- 6.000 câu ứng viên train (6 family), 2.000 câu kiểm tra giữ riêng (2 family).
- Nhãn mỗi trục đúng 50/50; không trùng nguyên văn với 200K câu scratch epoch 1.
- **Khuyết điểm mới phát hiện:** quá nhiều mẫu đều có ngữ cảnh hủy điểm lấy hàng => mẫu có nguy cơ quá nhân tạo.

### V6 đề xuất sử dụng cho pilot
- `candidate_semantic_v6_fixed.py` dùng `candidate_semantic_v5.py` làm thư viện chỉ đọc, không chỉnh sửa file nguồn.
- **9.600 câu = 1.200 cảnh × 8 tổ hợp**: `via`, `urgent`, `fragile` đúng 50/50 **trong từng cảnh**, không chỉ cân bằng tổng thể.
- **Ba tình huống bằng nhau:** bình thường, hủy điểm lấy hàng cũ, đính chính chỉ dẫn; trong cả hai split.
- **7.200 câu** ở `experiments\semantic_v6_balanced_train.jsonl` (6 họ template); **2.400 câu** ở `experiments\semantic_v6_family_challenge.jsonl` (2 họ template riêng).
- Không trùng nguyên văn với 200K câu gốc. Kiểm tra token, tính toàn vẹn 8 nhãn, cùng goal trong mỗi cảnh, provenance theo family/regime/seed, SHA256 đã hoàn tất.
- Dữ liệu V6 chỉ có kiểm tra cấu trúc tự động, **chưa có kiểm tra ngôn ngữ thủ công / human annotations và chưa train A/B**. Không dùng 2.400 câu challenge làm train.

## Cảnh báo còn tồn tại
- Trong V6, `P(via=True | 'khong')` là 40.74% ở candidate_train và 34.21% ở family_challenge (chưa đạt cân bằng cue). Một số gia đình template vẫn thiên về phủ định/no-via. Không đánh đồng cân bằng nhãn 50/50 với việc xóa hoàn toàn shortcut.
- Không phải mọi câu synthetic đều tự nhiên; cần kiểm tra tay các ví dụ spatial, lớp phủ định lồng nhau và thứ tự đính chính.
- Hai family heldout cùng từ vựng/địa danh và ngữ nghĩa tổng quát nên chưa phải blind human test.
- Không được train trên candidate challenge và không được chọn checkpoint theo challenge.

## Khuyến nghị thử nghiệm tiếp theo (chưa thực hiện)
Đầu tiên, kiểm tra ngôn ngữ và nhãn thủ công ~100 ví dụ ngẫu nhiên phân tầng theo 3 regime, named/spatial và nhãn. Sau đó mới thực hiện controlled A/B: cùng số update, seed, OneCycleLR, cùng tỷ lệ real data, cùng generator V4 nền; một nhánh giữ nguyên baseline, nhánh còn lại **thay khoảng 5–10%** synthetic bằng train V6 (oversampling cặp theo group). Báo cáo all exact, via FPR/FNR, urgent/fragile, calibration, 8-way scene consistency; so sánh trên family challenge V6 một lần và một tập câu người viết độc lập. Không dùng official validation/test để tune.

## Artifact checkpoint
- `checkpoints\corpus_audit_v1.json`
- `checkpoints\fragile_lexical_ablation_v1.json`
- `checkpoints\semantic_v5_dataset_manifest.json`
- `checkpoints\semantic_v5_frozen_model_diagnosis.json`
- `checkpoints\semantic_v6_dataset_manifest.json`
- `experiments\semantic_v6_balanced_train.jsonl`
- `experiments\semantic_v6_family_challenge.jsonl`

**Trạng thái cuối:** mới hoàn thành nghiên cứu nguyên nhân + tạo & QA ứng viên dữ liệu; chưa chứng minh hiệu quả bằng model training mới.
