# Cập nhật gói gui_ben_kia — 08/10/2026

## Đính chính bắt buộc về điểm Public Test

Đề bài `Phenikaa_Campus_Courier_2026_v3/DE_BAI.md`, mục 7: không phải mọi cảnh test đều được chấm; bảng public dùng khoảng 30% số cảnh được chấm. Vì vậy 907/12000 thay đổi KHÔNG giới hạn điểm public trong khoảng 48,11–63,23%. Khoảng này chỉ đúng nếu chấm toàn bộ 1.200 cảnh với cùng nhãn và cùng metric. Lập luận trước trong báo cáo bàn giao và `DOC_TRUOC.md` của gói bên kia đã dùng sai mẫu số.

Điểm 25,44% chưa chứng minh nộp nhầm file hoặc lỗi phía BTC. Cần xác minh file đã upload, vòng COURIER2 và khả năng regression tập trung vào các cảnh public. 69 khác biệt giữa hai candidate cũng không tương đương giới hạn 0,575 điểm public.

## Nội dung đã tích hợp và kiểm tra

- Source gói bên kia trùng code local sau patch 0004–0005, bỏ qua LF/CRLF; chỉ thêm `scripts/compare_submissions.py`.
- Công cụ compare được bổ sung và sửa: kiểm tra list số nguyên 0..3, độ dài theo sample, SHA, thay đổi theo robot/cảnh. Với điểm public và số cảnh được chấm chưa biết, công cụ báo không xác định được giới hạn điểm. Tính giới hạn khi truyền số cảnh được chấm đã biết bằng `--scored-scenes`, hoặc khẳng định điểm toàn test bằng `--full-test-score`.
- Thêm `--disable-map-aware` và `--disable-repair` vào create_submission/evaluate_system để tái lập và chạy ablation.
- Map-aware NLP v1 + style routing validation: CV scene exact 271/300 = 90,33%; full macro 73,27%; oracle và NLP-only 73,60%. Bằng kết quả trước map-aware, chưa có cải thiện đo được. Report: `results/nlp_map_aware/system_validation.json`.
- Toàn bộ 75 test đạt, gồm 3 test cho giới hạn thay đổi trên tập con.

## Artifact solver train+validation

Artifact kèm gói đã nạp kiểm tra thành công, scikit-learn 1.7.2 tương thích runtime; có đủ 10 model, digest dataset khớp bộ local, predict thử thành công. Đã đặt tại `artifacts/solver/candidate_strategy_trainval.joblib`; pipeline mặc định vẫn dùng `candidate_strategy.joblib`.

SHA-256: `417E106C43F0FBF65CAB83185537A30C1E2B9D0B9B8066C65F331C9085B302B6`.

Validation lưu trong artifact là 87,33% nhưng model đã học trên validation, KHÔNG phải kết quả held-out và không được dùng để khẳng định tổng quát hóa. Style classifier local đã train/kiểm tra 300/300 validation nên giữ artifact local hiện có.

## Train NLP v2 đang chạy

Khởi động nền lúc 21:49 giờ Việt Nam ngày 08/10, PID 15892, CUDA RTX 3050 Laptop 4GB.

- 3 seed × 15 epoch.
- 200.000 câu tổng hợp mới mỗi epoch + 2.000 train lặp 5 lần.
- dim 96, hidden 192, batch 128, lr 0.002, CPU threads 4.
- 7.515.212 tham số mỗi seed, tổng ensemble 22.545.636.
- Checkpoint tốt nhất theo seed được lưu trong `results/nlp_neural_v2/checkpoints`.
- Log: `results/nlp_neural_v2/full_run/training.log` và `training.stderr.log`.
- Artifact cuối: `artifacts/nlp/neural_parser_v2.pt`.
- Report cuối: `results/nlp_neural_v2/report.json`.

Đã chạy smoke test cơ chế checkpoint thành công; không dùng model smoke để sinh submission. Chưa có kết quả cuối NLP v2. Sau khi kết thúc phải kiểm tra ngưỡng hybrid, validation grounded, holdout và chạy end-to-end với artifact cuối trước khi chọn candidate. Không gửi submission trong bước này.
