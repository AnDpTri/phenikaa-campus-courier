# NLP v3 — hướng dẫn train (09/10)

## Mục tiêu

Bản tốt nhất #1102 (64,78%) dùng NLP v2. Mô hình v2 khi chạy một mình chỉ đúng goal 91,7%
trên câu validation **thật**, trong khi đúng 83,7% trên câu tự sinh cách nói mới. Tức là
nó yếu ở **cấu trúc câu thật** (nhiều mệnh đề, câu gây nhiễu, người nhận đi kèm vị trí).
v3 nhắm đúng chỗ này.

## Thay đổi so với v2

1. **`synth.augment_real`**: mỗi epoch, mỗi câu train thật được viết lại `--real-augment`
   lần: tên địa điểm → tên khác **cùng loại**, cụm "cực bắc/nam/…" → cụm khác **cùng
   hướng**. Giữ nguyên cấu trúc câu thật và nhãn. Tên nằm trong tên đồ vật/người
   ("thẻ thư viện", "cán bộ thư viện") không bị đổi. Chỉ dùng bank phần train (holdout
   vẫn sạch). Parser luật chỉ còn đọc đúng ~81% các câu đã viết lại → đúng loại câu mô
   hình cần học thêm.
2. **Người nhận ở một vị trí mô tả**: "Bếp trưởng ở nơi gần KTX nhất đang chờ…" → goal là
   vị trí, không phải nơi làm việc của người đó (giống train thật).
3. **`--init`**: train tiếp từ trọng số v2 (net i → seed i), không train từ đầu.
4. **`--checkpoint-dir`** (gộp từ bản của Codex) + **`scripts/nlp/merge_checkpoints.py`**: nếu
   phải dừng sớm, gộp các seed đã xong thành một artifact dùng được.

## Lệnh (GPU)

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m unittest tests.nlp.test_neural tests.nlp.test_parser
py -3.12 scripts/nlp/train_neural_parser.py `
  --init artifacts/nlp/neural_parser_v2.pt --dim 96 --hidden 192 `
  --seeds 3 --epochs 6 --lr 1e-3 --batch 128 `
  --synthetic 200000 --real-repeat 5 --real-augment 10 `
  --checkpoint-dir results/nlp_v3_ckpt `
  --out artifacts/nlp/neural_parser_v3.pt --report results/nlp_neural_v3/report.json
```

Ước lượng: ~230.000 dòng/epoch ≈ 8–9 phút/epoch trên RTX 3050 → ~50 phút/seed, ~2,5 giờ
cho 3 seed. Thiếu thời gian: `--seeds 1` (~50 phút).

Dừng sớm sau khi có ít nhất 1 seed xong:
```powershell
py -3.12 scripts/nlp/merge_checkpoints.py results/nlp_v3_ckpt/seed_*_best.pt --out artifacts/nlp/neural_parser_v3.pt
```

## Sau khi train

- Dòng `hybrid thresholds:` trong log **không quan trọng**: ngưỡng nộp bài sẽ đặt lại khi tạo
  submission (`--nlp-goal-threshold`, `--nlp-via-threshold`), chọn từ số đo offline.
- Gửi file `neural_parser_v3.pt` qua repo GitHub như v2. Claude sẽ đo ngưỡng, tạo file
  nộp, kiểm tra SHA và số dự đoán thay đổi trước khi đưa nộp.
- Kiểm tra trong log: `holdout-phrasing … goal` của từng seed phải ≥ ~0,83 (v2), và
  `validation all` không thấp hơn v2 nhiều.

## Ứng viên nộp không cần v3 (đã đo offline với v2)

Cho mô hình ghi đè **via** (hiện `via=2.0`, chưa bao giờ ghi đè):

| Cấu hình (goal 0,8) | Validation goal / via / all | Holdout goal / via / all |
|---|---|---|
| via 2,0 (#1102) | 99,3 / 100 / 99,3 | 77,1 / 75,6 / 50,6 |
| **via 0,95** | 99,3 / 98,7 / 98,0 | 77,1 / **90,8** / 60,2 |

Lệnh: thêm `--nlp-via-threshold 0.95` vào lệnh #1102. Không nên ghi đè urgent/fragile
(validation urgent tụt còn 89,7%).
