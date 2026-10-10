# Sửa đồ thị khi mission không tới được

## Vấn đề

Audit submission v2: 112 cảnh test (1.051 lượt robot) có goal hoặc via không tới
được trên đồ thị CV dự đoán; validation có 0 cảnh như vậy. Với đồ thị đúng, goal
luôn tới được, nên đây là lỗi CV (724 lượt không có đường kể cả khi bỏ chiều,
327 lượt chỉ bị chặn bởi chiều đường). Bản cũ đoán: đi theo hướng đang quay hoặc
action hợp lệ nhỏ nhất.

## Cách làm (`src/courier/solver/repair.py`)

Khi đặc trưng không tính được vì mission không tới được, nới đồ thị theo bước nhỏ
nhất đủ để tới được, rồi để candidate model xếp hạng action trên đồ thị đã sửa:

1. bỏ ràng buộc một chiều;
2. thêm đường thường giữa mọi cặp nút kề nhau trên lưới chưa có đường;
3. coi đường đóng là mở;
4. thêm mọi nút lưới còn thiếu (và nút robot/landmark).

Không sửa được thì đi tham lam về goal/via gần nhất theo khoảng cách ô.
Cảnh bình thường không bị ảnh hưởng: dự đoán validation sạch giống hệt từng phần
tử (73,60%). Tắt bằng `strategy.repair_unreachable = False`.

## Đánh giá (mô phỏng, không dùng test)

`scripts/solver/evaluate_repair.py` làm hỏng đồ thị thật dọc tuyến của robot (xóa
cạnh, xóa nút, đảo một chiều, đóng đường) đến khi mission không tới được, rồi so
với nhãn thật.

| Tập | Cảnh mô phỏng | Cũ | Sửa đồ thị |
|---|---:|---:|---:|
| Validation (seed 0) | 281 | 35,09% | **53,20%** |
| Train (seed 1, lạc quan vì model học trên train) | 1.882 | 36,92% | **58,04%** |

Validation theo kiểu lỗi: xóa cạnh 34,3→50,7; xóa nút 37,1→51,8; một chiều
36,1→68,3; đóng đường 32,8→41,5.

Ước lượng trên test: nếu 1.051 lượt tăng ~18 điểm phần trăm thì macro tăng khoảng
+1,5 điểm. Đây là ước lượng từ mô phỏng, chưa phải kết quả.

## Chạy

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m unittest tests.solver.test_repair
py -3.12 scripts/solver/evaluate_repair.py
py -3.12 scripts/create_submission.py --diagnostics-dir results/submission_repair/diagnostics
```

Không cần train lại gì. Có thể kết hợp với `--nlp-model` (xem `docs/NLP_NEURAL.md`).
Trong `diagnostics/summary.json`, `fallback_modes` sẽ ghi `repaired_<bước>` hoặc `greedy`.
