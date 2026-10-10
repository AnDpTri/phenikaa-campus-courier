# Gửi nhóm tiếp quản — phản hồi báo cáo 08/10 21:39

## 1. Về điểm 25,44%

Báo cáo của các bạn lập luận đúng: điểm này **không thể** đến từ file SHA `B90F…`.

- `B90F…` chỉ khác v2 (`D5675EC6…`, 55,67%) ở 907/12.000 dự đoán, nên điểm của nó
  chắc chắn nằm trong khoảng **48,1% – 63,2%**.
- 25,44% ≈ mức đoán ngẫu nhiên với 4 hướng (25%). Đây là dấu hiệu file nộp không khớp
  thứ tự với nhãn hoặc nộp nhầm file, không phải dấu hiệu mô hình kém.
- Kiểm tra chéo hai điểm đã biết: v1 (52,39%) và v2 (55,67%) khác nhau 1.886 dự đoán,
  khoảng hợp lệ suy ra 36,7–68,1% → hai điểm này nhất quán. 25,44% thì không.
- Mã nguồn trong gói bàn giao của các bạn trùng với mã của tôi (chỉ khác ký tự xuống
  dòng CRLF/LF); không có lỗi nào phát sinh khi tích hợp.

## 2. Việc cần làm ngay (theo thứ tự)

**Bước 1 — kiểm tra đúng file đã upload** (không phải file nghĩ là đã upload):

```powershell
$env:PYTHONPATH = "src"
py -3.12 scripts/compare_submissions.py <file_đã_upload> `
  --reference results/submission_20261008_nlp57f5dd2_cv512_solverv2/predictions.json `
  --reference-score 55.67
```

Script in: SHA-256, kiểm tra định dạng (danh sách JSON 12.000 số nguyên 0..3), số dự
đoán khác v2 theo từng robot, và **khoảng điểm hợp lệ**. Gửi lại toàn bộ output.

**Bước 2 — xử lý theo kết quả:**

| Kết quả | Kết luận | Làm gì |
|---|---|---|
| SHA ≠ `B90F17670B…` | Nộp nhầm file | Nộp lại đúng `results/submission_nlp_repair_style_20261008/predictions.json` |
| SHA = `B90F17670B…` | Trang thi đọc file khác cách / chấm khác | Nộp lại chính v2 `D5675EC6…`; nếu không còn ra 55,67% thì lỗi ở phía trang thi |

**Bước 3 — chưa nộp bản nào khác** cho tới khi xong bước 1–2.

**Từ giờ, trước mọi lượt nộp:** chạy `compare_submissions.py` với v2 làm mốc, ghi
SHA + số dự đoán thay đổi + khoảng điểm hợp lệ vào nhật ký. Nếu điểm trả về nằm
ngoài khoảng đó thì chắc chắn có lỗi file/upload.

## 3. Sau khi xác minh xong: nộp từng thay đổi riêng (ablation)

Mỗi lượt chỉ thêm một thay đổi so với lượt trước, đều tính từ v2:

| Lượt | Lệnh thêm vào `create_submission.py` | Ghi chú |
|---|---|---|
| A | (không thêm gì) | Chỉ có solver sửa đồ thị — tự bật, không cần cờ |
| B | `--nlp-model artifacts/nlp/<model>.pt` | NLP hybrid + chọn goal theo bản đồ |
| C | `--print-detector results/cv_experiments/print_768_aug_20261008/detector.pt` | Định tuyến ảnh print sang 768 |
| D | `--strategy-artifact artifacts/solver/candidate_strategy_trainval.joblib` | Solver train trên train+validation |

Nếu lượt nào giảm điểm so với lượt trước, bỏ thay đổi đó.

## 4. Những gì mới trong gói này so với gói bàn giao của các bạn

1. **Chọn goal theo bản đồ** (`HybridMissionParser.parse_for_map`, tự bật khi có
   `--nlp-model`): khi loại goal đọc được không có trên bản đồ (245 cảnh test), trước
   đây resolver lấy landmark *đầu tiên trong danh sách* (gần như ngẫu nhiên). Giờ
   dùng goal của mô hình nếu tìm được trên bản đồ, nếu không thì loại có xác suất cao
   nhất trong các loại có trên bản đồ. Chưa đo được trên validation (không có cảnh
   nào rơi vào trường hợp này).
2. **`scripts/compare_submissions.py`**: kiểm tra file trước khi nộp (mục 2).
3. **`candidate_strategy_trainval.joblib`** (gửi kèm, scikit-learn 1.7.2): solver train
   trên train+validation → chép vào `artifacts/solver/`.
4. **`style_classifier.joblib`** (gửi kèm) → chép vào `artifacts/cv/`.

## 5. Về NLP v1: ngưỡng = 2,0

Báo cáo ghi ngưỡng chọn ra là `goal=2.0, via=2.0, flags=2.0`: mô hình **không bao giờ
ghi đè** kết quả của luật, chỉ điền goal khi luật không ra goal (85 cảnh test). Tức là
phần lớn lợi ích của mô hình (holdout goal 28,8% → 69,4%) **chưa được dùng**. Lý do: ở
mọi ngưỡng thấp hơn, mô hình v1 làm sai ít nhất một cảnh validation nên bị loại.

Bản v2 (dữ liệu đa dạng hơn, ensemble 3 seed, xem `docs/NIGHT_PLAN.md` mục Việc 1)
được kỳ vọng tự tin và đúng hơn trên validation, để ngưỡng < 2,0 được chọn. Khi train
xong, kiểm tra dòng `hybrid thresholds:` trong log.

## 6. Ràng buộc (giữ nguyên)

Không dùng test để train, gán nhãn, chọn ngưỡng hay viết luật; test chỉ để sinh dự
đoán. Không gọi dịch vụ AI bên ngoài khi dự đoán. Không commit API key, dữ liệu test
trích xuất hay submission.
