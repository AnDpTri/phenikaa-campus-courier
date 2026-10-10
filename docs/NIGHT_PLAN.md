# Kế hoạch đêm 08–09/10: mục tiêu, lệnh chạy, tiêu chí quyết định

Hạn nộp: 0h ngày 10/10 (giờ Việt Nam), 5 lượt nộp/ngày. Điểm public hiện tại 55,67%
(v2 candidate). Validation toàn hệ thống 73,10%.

Các gói mã (áp theo thứ tự lên main `e29d99a`, bằng `git am`):

| Patch | Nội dung | Cần train? |
|---|---|---|
| 0001 | NLP: bộ sinh câu, mô hình nhỏ, hybrid với parser luật | Có (GPU) |
| 0002 | Solver: sửa đồ thị khi mission không tới được | Không |
| 0003 | NLP v2: thêm cách diễn đạt, ensemble nhiều seed, dữ liệu mới mỗi epoch; CV: phân loại style + định tuyến ảnh print sang detector 768 | Có (NLP); style ~1 phút CPU |
| 0004 | NLP: khi loại goal không có trên bản đồ (245 cảnh test, trước đây lấy landmark đầu danh sách), dùng goal của mô hình hoặc loại có xác suất cao nhất trong số loại có trên bản đồ. Tự bật khi có `--nlp-model` | Không |

Môi trường: Python 3.12, `scikit-learn==1.7.2` (artifact solver trên main được lưu bằng bản này),
torch, `opencv-python<5`. Luôn đặt `$env:PYTHONPATH = "src"`.

---

## Việc 1 — NLP (ưu tiên cao nhất)

**Mục tiêu:** parser hiểu được cách nói chưa có trong train/validation. Bằng chứng: trên test
parser chỉ đọc ra `anchor_near` ở 6,3% cảnh, trong khi train/validation là 36–42%; 8% cảnh
không ra goal. Đây nhiều khả năng là nguồn mất điểm lớn nhất.

**Lệnh (đêm nay, GPU):**
```powershell
py -3.12 -m unittest tests.nlp.test_neural tests.nlp.test_parser
py -3.12 scripts/nlp/train_neural_parser.py --synthetic 200000 --epochs 15 --seeds 3 `
  --dim 96 --hidden 192 --batch 128 --lr 2e-3 `
  --out artifacts/nlp/neural_parser_v2.pt --report results/nlp_neural_v2/report.json
```
Ước lượng: mã hoá văn bản ~2 phút/epoch trên CPU + phần GPU; 3 seed × 15 epoch ≈ 2–3 giờ.
Nếu thiếu thời gian: `--seeds 1` hoặc `--synthetic 100000 --epochs 10`.

**Tiêu chí chấp nhận** (đọc ở cuối log / report.json):
- `validation_grounded hybrid all` = 1,0000 (không thua parser luật).
- `holdout hybrid goal` cao hơn `holdout rules goal` ít nhất 20 điểm.
- `hybrid thresholds: goal` < 2,0 (nếu = 2,0 thì mô hình không bao giờ được dùng).
- Mô hình v1 đang train vẫn dùng được; chọn bản có `holdout hybrid all` cao hơn.

---

## Việc 2 — Solver sửa đồ thị (không cần train, nộp được ngay)

**Mục tiêu:** 112 cảnh test (1.051 lượt robot) có đồ thị CV bị đứt; trước đây đoán bừa.
Mô phỏng trên validation: 35,1% → 53,2%. Ước lượng public **+1,5 điểm**.
Cảnh bình thường không đổi (validation 73,60% giữ nguyên từng dự đoán).

```powershell
py -3.12 -m unittest tests.solver.test_repair
py -3.12 scripts/create_submission.py --out results/sub_repair/predictions.json `
  --diagnostics-dir results/sub_repair/diagnostics
```
Kiểm tra `diagnostics/summary.json → fallback_modes` có `repaired_*`.

---

## Việc 3 — CV định tuyến theo style (print → detector 768)

**Mục tiêu:** detector 768 tăng print scene exact 66% → 85% nhưng giảm classic/sketch;
chỉ dùng nó cho ảnh print. Bộ phân loại style: validation 300/300 = 100%.

```powershell
py -3.12 scripts/cv/train_style.py --out artifacts/cv/style_classifier.joblib
py -3.12 -m unittest tests.cv.test_style
py -3.12 scripts/evaluate_system.py --print-detector results/cv_experiments/print_768_aug_20261008/detector.pt `
  --report results/cv_style_route/system_validation.json
```
**Tiêu chí:** `full macro` ≥ 73,10% (baseline) và `cv_scene_exact` tăng.

**Kết quả đã đo (validation, CPU, cùng máy):**

| | Gốc 512 | 768 cho mọi ảnh (báo cáo cũ) | **Định tuyến print → 768** |
|---|---:|---:|---:|
| CV scene exact | 86,00% | 90,00% | **90,33%** |
| cv_only macro | 72,87% | — | **73,23%** |
| full macro | 73,10% | 72,87% | **73,27%** |

→ Nên bật `--print-detector` (không giảm ở đâu, CV tăng rõ).

---

## Việc 4 — Solver cuối: train trên train + validation (làm sau cùng)

Đường cong học còn tăng (1.500 → 2.000 cảnh: +0,44 điểm). Ước lượng +0,3–0,5.
Sau bước này validation không còn là số đo độc lập cho solver, nên làm cuối.
Artifact này Claude đã train sẵn (scikit-learn 1.7.2) và gửi kèm: `candidate_strategy_trainval.joblib`
→ chép vào `artifacts/solver/`. Không cần chạy lệnh dưới nếu dùng file gửi kèm.

```powershell
py -3.12 scripts/solver/train_candidate_strategy.py --fit-on train+validation `
  --out artifacts/solver/candidate_strategy_trainval.joblib
```
(Lần đầu sẽ dựng cache đặc trưng nếu chưa có; dùng `--cache` trỏ tới cache 48 profile nếu đã có.)

---

## Thứ tự nộp đề xuất (mỗi lượt đổi một thứ để biết tác dụng)

| Lượt | Cấu hình | So với |
|---|---|---|
| A | v2 + sửa đồ thị (việc 2) | 55,67% |
| B | A + `--nlp-model` (việc 1) | A |
| C | B + `--print-detector` (việc 3) nếu validation không giảm | B |
| D | C + `--strategy-artifact …trainval.joblib` (việc 4) | C |

Lệnh đầy đủ cho lượt D:
```powershell
py -3.12 scripts/create_submission.py `
  --nlp-model artifacts/nlp/neural_parser_v2.pt `
  --print-detector results/cv_experiments/print_768_aug_20261008/detector.pt `
  --strategy-artifact artifacts/solver/candidate_strategy_trainval.joblib `
  --out results/sub_D/predictions.json --diagnostics-dir results/sub_D/diagnostics
```

Nếu lượt nào giảm điểm public so với lượt trước, bỏ thay đổi của lượt đó.

## Ràng buộc (giữ nguyên)

- Không dùng test để train, gán nhãn, chọn ngưỡng, viết luật hay chọn augmentation.
  Test chỉ để sinh dự đoán. Điểm public là thước đo hợp lệ duy nhất trên test.
- Không gọi dịch vụ AI bên ngoài khi dự đoán; tổng tham số mô hình < 200 triệu
  (NLP ensemble 3 × ~8 triệu, CV vài triệu, solver cây quyết định).
- Không commit API key, dữ liệu test trích xuất hay submission.
