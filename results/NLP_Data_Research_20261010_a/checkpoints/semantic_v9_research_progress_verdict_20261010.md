# V9 – Tiến độ nghiên cứu sửa urgent/via (10/10/2026)

**Trạng thái: NGHIÊN CỨU, CHƯA PHÊ DUYỆT. KHÔNG dùng checkpoint mới cho submission hoặc artifact chính.**

## Phạm vi
- Chỉ ghi file mới trong `D:\phenikaa\results\NLP_Data_Research_20261010_a`.
- Đọc real train 2.000 câu và generator gốc read-only; **không đọc official validation/test** trong các phép thử mới.
- Không sửa source, config, checkpoints gốc; không có submission, không có promotion.
- Mô hình V5 #1182 public score 70,39% được giữ nguyên. Đánh giá bên dưới chỉ là NLP synthetic holdout.

## Dữ liệu sửa lỗi
- `experiments/semantic_v9_focused_urgent_via_train_v2.jsonl`: 960 câu thành cặp phản thực cho `urgent` và `via` (480 câu mỗi nhóm), median 50 tokens.
- `experiments/semantic_v9_focused_urgent_via_holdout_v2.jsonl`: 192 câu đối chứng mẫu ngôn ngữ riêng, không dùng để train.
- `checkpoints/semantic_v9_focused_urgent_via_manifest_v2.json`: sinh dữ liệu và kiểm tra không trùng nguyên văn với V9/V8 family.
- Tất cả vẫn do template sinh; **chưa có kiểm duyệt người độc lập**.

## Thử nghiệm 1 – B pilot cũ so với C thêm 600 câu
- Cùng checkpoint V5 scratch seed0 epoch05, cùng 2.000 câu real train, cùng 1.000 mẫu V9 cũ tại đúng vị trí, cùng seed/optimizer/budget (3 epochs, 282 updates).
- C thay thêm 600 mẫu weighted synthetic bằng câu sửa urgent/via.
- Chênh lệch **C − B** exact-all: v2 mới **−0,11 điểm %**; weighted mới **−7,89**; hard mới **−16,83**; V8 family **+5,33**; focused holdout192 **+35,42**.
- Đây là **catastrophic forgetting** trên phân bố weighted/hard, chủ yếu ở `fragile`. `via` FPR cũng tăng. **C bị loại.**
- Chi tiết: `checkpoints/v9_focused_via_urgent_controlled_BvsC_report_v2.json`.
- Checkpoint C: `experiments/v9_focused_C_seed0_continuation_v2.pt` – chỉ để nghiên cứu.

## Thử nghiệm 2 – Trộn trọng số B↔C
- Kiểm tra tỷ lệ C là 10%, 25%, 40%, 60% trên synthetic probes, kiểm định gate bảo toàn accuracy/fragile/via FPR.
- **0/4 cấu hình vượt gate**; alpha 10% vẫn giảm hard exact-all 3,42 điểm %, hard fragile 6,17 điểm %.
- Không lưu mô hình trộn, không promote.
- Báo cáo: `checkpoints/v9_focused_weight_interpolation_study_v1.json`.

## Thử nghiệm 3 – Đóng băng backbone, goal, fragile; chỉ học urgent/via
- Khởi tạo cả B0 và D1 từ cùng V9 pilot B frozen. Mỗi arm 6.400 câu, 2 epochs, cùng seed/optimizer. B0 có 600 câu weighted thường; D1 thay đúng 600 câu đó bằng câu sửa urgent/via.
- Các đầu `goal`, `fragile` và encoder **không đổi một bit**, được assert sau training.
- D1 − B0 exact-all: v2 mới **+0,375 điểm %**, weighted mới **+0,188**, hard mới **+1,200**, family challenge **+1,500**, focused192 **−1,042**.
- Urgent accuracy trên family **+4,167 điểm %**, urgent false-positive rate giảm 8,333 điểm %.
- **Nhưng via false-positive rate tăng:** weighted **+1,913 điểm %**, family **+2,167 điểm %**; focused192 exact-all còn giảm.
- Vì vậy D1 là thử nghiệm chưa qua gate về độ đặc hiệu via, **không được coi là bản V9 hoàn thiện hoặc đem nộp.**
- Báo cáo: `checkpoints/v9_restricted_heads_control_B0_vs_D1_report_v1.json`.
- Models B0/D1: `experiments/v9_restricted_B0_control_v1.pt`, `experiments/v9_restricted_D1_focused_v1.pt`.

## Nhận định và bước nghiên cứu tiếp
1. Cần chống nhầm `via` khi xuất hiện từ ngữ điểm ghé trong câu phủ định/đã hủy: tăng negative controls, kiểm tra label; ưu tiên specificity hơn chỉ cải thiện recall.
2. Urgent family bị false-positive nghiêm trọng; mẫu positive/negative phải cân bằng tín hiệu, tránh shortcut.
3. Giữ frozen backbone/fragile hoặc rehearsal phù hợp để tránh catastrophic forgetting.
4. Bắt buộc xác minh nhiều seed và một bộ câu do người viết độc lập, chưa dùng để chọn cấu hình.
5. Không dùng test text hoặc public leaderboard để gắn nhãn/train/chỉnh luật. Chưa được phép đưa bất kỳ checkpoint mới nào vào `artifacts/`.

**Kết luận: V9 đã tiến về phân tích lỗi và xác lập cơ chế bảo toàn mô hình, nhưng chưa đạt chuẩn triển khai.**
