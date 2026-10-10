# NLP: mô hình nhỏ + hybrid với parser luật

## Vì sao cần

Phân tích ngày 2026-10-08 trên đầu ra của parser (chỉ chạy dự đoán, đếm tổng, không đọc câu test):

| Kiểu goal | Train (nhãn) | Validation (nhãn) | Test (parser đọc ra) |
|---|---:|---:|---:|
| Gọi tên thẳng | 41,9% | 23,7% | 78,5% |
| `anchor_near` ("gần X nhất") | 36,1% | 41,7% | 6,3% |
| `*_most` (cực bắc/nam/đông/tây) | ~16,5% | ~24,7% | 4,7% |
| Không ra goal | 0% | 0% | 8,0% |

Test và validation cùng profile `hard`; đồ thị CV dự đoán trên test có thống kê
gần như trùng validation. Nhiều khả năng test diễn đạt khác train/validation và
parser luật hiểu sai khoảng một nửa số câu mà không báo lỗi. Trên dữ liệu tự sinh
với cách nói mới, parser luật chỉ đúng goal 29–38%.

## Thành phần

| File | Vai trò |
|---|---|
| `src/courier/nlp/synth.py` | Sinh câu nhiệm vụ có nhãn từ ngữ pháp viết tay (không dùng test). Mỗi bank cụm từ chia train/holdout (1/5) để đo khả năng hiểu cách nói chưa thấy. |
| `src/courier/nlp/neural.py` | `NeuralParserNet` (embedding băm từ + trigram ký tự, BiGRU 2 lớp, 8 đầu ra có attention riêng, ~4,8 triệu tham số), `NeuralMissionParser`, `HybridMissionParser`. |
| `scripts/nlp/train_neural_parser.py` | Train, chọn checkpoint, chọn ngưỡng hybrid, lưu artifact + báo cáo. |
| `scripts/create_submission.py`, `scripts/evaluate_system.py` | Thêm `--nlp-model PATH`; không truyền thì giữ nguyên parser luật. |
| `tests/nlp/test_neural.py` | Test bộ sinh, mã hoá, decode, lưu/nạp, hybrid. |

Hybrid: luôn chạy parser luật trước. Mô hình chỉ thay kết quả khi
(a) luật không ra goal, hoặc (b) mô hình khác luật và độ tự tin ≥ ngưỡng.
Ngưỡng chọn tự động trên validation + holdout tự sinh + bộ challenge, với điều
kiện validation không được giảm so với parser luật. Ngưỡng được lưu trong artifact.

## Chạy

```powershell
$env:PYTHONPATH = "src"
# 1. Test
py -3.12 -m unittest tests.nlp.test_neural tests.nlp.test_parser
# 2. Train (GPU tự dùng nếu có; CPU 4 nhân ~9 phút/epoch với cấu hình mặc định)
py -3.12 scripts/nlp/train_neural_parser.py --out artifacts/nlp/neural_parser.pt --report results/nlp_neural/report.json
# 3. Đánh giá toàn hệ thống trên validation (so với bản không có --nlp-model)
py -3.12 scripts/evaluate_system.py --nlp-model artifacts/nlp/neural_parser.pt --report results/nlp_neural/system_validation.json
# 4. Sinh submission
py -3.12 scripts/create_submission.py --nlp-model artifacts/nlp/neural_parser.pt --diagnostics-dir results/submission_nlp_neural/diagnostics
```

`create_submission.py` ghi thêm bộ đếm `nlp_goal_neural_fill`, `nlp_goal_neural_override`, ...
vào diagnostics để biết mô hình đã thay bao nhiêu cảnh.

## Tiêu chí trước khi nộp

- `validation_grounded / hybrid / all` phải bằng parser luật (1,0000).
- `holdout / hybrid / goal` phải cao hơn rõ `holdout / rules / goal`.
- Validation toàn hệ thống không giảm.
- Điểm public là thước đo cuối cùng.

## Ràng buộc

- Không dùng test để train, gán nhãn, chọn ngưỡng hay viết luật; test chỉ để sinh dự đoán.
- Không gọi dịch vụ AI bên ngoài khi dự đoán. Mô hình ~4,8 triệu tham số (giới hạn 200 triệu).
- Không commit artifact `.pt` lớn hoặc submission nếu đội không muốn.
- Bộ challenge 50 câu và bộ sinh do cùng một người viết nên có thiên lệch; holdout chỉ
  đo được cách nói mới trong phạm vi ngữ pháp này, không đại diện cho test.

## Chỉnh nếu cần

- Thêm cách nói: bổ sung vào các bank trong `synth.py` (`ANCHOR_NEAR_FRAMES`,
  `MOST_PHRASES`, `GOAL_FRAMES`, `EXTRA_PLACES`, ...). Nguồn phải là hiểu biết tiếng Việt chung.
- Nhiều dữ liệu hơn: `--synthetic 120000 --epochs 10`.
- Mô hình lớn hơn: `--dim 96 --hidden 192`.
