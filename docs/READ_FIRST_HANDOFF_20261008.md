# Đọc trước — gói mã nguồn và báo cáo ngày 08/10/2026, 21:52 (UTC+7)

Gói này là snapshot mã nguồn đang dùng tại D:\phenikaa, gồm cập nhật patch 0001–0005, công cụ compare đã sửa, cờ ablation và cơ chế lưu checkpoint NLP v2. Base Git là e29d99a3a241a3d84e129ff9f12f564186bd0dc6; các cập nhật mới đang nằm trong working tree, chưa được push.

## Báo cáo ưu tiên đọc

1. `docs/UPDATE_GUI_BEN_KIA_20261008.md`: hiện trạng mới nhất, đính chính điểm public, artifact solver và cấu hình train v2.
2. `docs/HANDOFF_REPORT_20261008.md`: báo cáo bàn giao đã sửa kết luận sai về giới hạn điểm public.
3. `reports/`: bản sao số liệu validation và NLP v1, log train v2 tại thời điểm đóng gói.

README.md và các tài liệu tiến trình cũ mô tả các mốc lịch sử. Nếu thông tin khác nhau, dùng báo cáo cập nhật ở mục 1 làm hiện trạng. Đặc biệt, không lấy điểm 87,33% của solver train+validation làm validation độc lập và không dùng 907/12000 làm giới hạn thay đổi điểm public.

## Kết quả đã xác nhận

- 75/75 unit test đạt trên máy local.
- Baseline: CV scene exact 85,67%, full validation macro 73,10%.
- Style routing print → detector 768: CV scene exact 90,33%, full validation macro 73,27%.
- Thêm map-aware NLP v1: full validation macro vẫn 73,27%.
- Điểm public người dùng báo: v2 55,67%, candidate style 25,44%; nguyên nhân giảm chưa xác định.
- Public chấm một tập con ẩn của test; test còn có các cảnh không chấm. Không thể suy ra giới hạn thay đổi điểm public chỉ từ tỷ lệ khác biệt trên 12.000 phần tử.

## Train NLP v2

Đang chạy nền tại máy gốc, PID 15892, khởi động 21:49 giờ Việt Nam ngày 08/10. Cấu hình: CUDA, 3 seed × 15 epoch, 200.000 synthetic mới mỗi epoch + 2.000 train lặp 5 lần, dim 96, hidden 192, batch 128, lr 0.002, threads 4. Mỗi seed có 7.515.212 tham số.

Gói này chưa có kết quả cuối hoặc trọng số NLP v2. Các file log trong `reports/` chỉ là ảnh chụp tiến trình khi đóng gói. Sau khi train kết thúc phải đọc report cuối và chạy validation end-to-end trước khi sinh candidate.

## Môi trường và dữ liệu cần có

Python 3.12, torch với CUDA tương thích, scikit-learn 1.7.2, opencv-python < 5. Cài thư viện CV/solver theo pyproject.toml; cần lấy đúng bản torch CUDA cho GPU. Gói source không kèm dataset hoặc artifact model nên không thể tự chạy dự đoán đầy đủ chỉ bằng ZIP này.

Đặt bộ dữ liệu tại `Phenikaa_Campus_Courier_2026_v3/delivery_public/`. Các test phụ thuộc dữ liệu cần có train/validation tại vị trí đó. Các artifact cần bổ sung:

- `artifacts/cv/detector.pt`, `edge_net.pt`, `node_net.pt`, `weather_classifier.joblib`.
- `artifacts/cv/style_classifier.joblib` nếu route print.
- `results/cv_experiments/print_768_aug_20261008/detector.pt` cho print 768.
- `artifacts/solver/candidate_strategy.joblib` cho baseline solver.
- `artifacts/nlp/neural_parser.pt` cho NLP v1; `neural_parser_v2.pt` chỉ có sau khi train xong.

## Lệnh chạy

Chạy ở thư mục root đã giải nén, thay đường dẫn output bằng đường dẫn mới. Các chương trình evaluation/submission từ chối ghi đè report hoặc predictions đã tồn tại.

```powershell
$env:PYTHONPATH = "src;."
py -3.12 -m pip install -e ".[cv,solver]"
py -3.12 -m unittest discover -s tests -v

# NLP v2: lấy cấu hình từ docs/NIGHT_PLAN.md; không chạy trùng lượt đang chạy trên máy gốc.
py -3.12 scripts/nlp/train_neural_parser.py --device cuda --threads 4 --synthetic 200000 --epochs 15 --seeds 3 --dim 96 --hidden 192 --batch 128 --lr 2e-3 --out artifacts/nlp/neural_parser_v2.pt --report results/nlp_neural_v2/report.json --checkpoint-dir results/nlp_neural_v2/checkpoints

# End-to-end với NLP v1 và route print.
py -3.12 scripts/evaluate_system.py --nlp-model artifacts/nlp/neural_parser.pt --print-detector results/cv_experiments/print_768_aug_20261008/detector.pt --detector-device cuda --report results/new_validation/report.json

# So sánh file; reference-score ở đây là public nên không tính full-test score bound.
py -3.12 scripts/compare_submissions.py <candidate.json> --reference <v2.json> --reference-score 55.67
```

`--disable-map-aware` và `--disable-repair` trong evaluate_system/create_submission dùng để đánh giá riêng từng thay đổi. Tắt neural bằng cách bỏ `--nlp-model`, tắt route print bằng cách bỏ `--print-detector`. Solver train+validation là artifact tùy chọn, không phải mặc định.

Không dùng nội dung test để train, gán nhãn hay viết luật. Chưa có đề nghị nộp mới cho các candidate đang thất bại public.

## Nội dung ZIP

`src/`, `scripts/`, `tests/`, `docs/`, cấu hình dự án, đề bài Markdown và số liệu báo cáo chọn lọc. Không chứa dữ liệu train/validation/test, trọng số model, predictions, .git, thư mục giải nén tạm, cache hoặc shortcut. `MANIFEST_SHA256.json` ghi hash của các file trong snapshot để kiểm tra toàn vẹn.
