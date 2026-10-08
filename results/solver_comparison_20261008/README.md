# So sánh solver ngày 2026-10-08

Mỗi phiên bản được lưu riêng. File nộp gốc `D:/phenikaa/predictions.json`
được giữ nguyên; bản sao nằm trong `v1_submitted/predictions.json`.

| Chỉ số | v1_submitted | v2_candidate |
|---|---:|---:|
| Solver với graph + mission chuẩn, 300 cảnh validation | 69,10% | 73,60% |
| Toàn hệ thống với CV/NLP, 300 cảnh validation | 68,57% | 73,10% |
| Điểm public do người dùng cung cấp | 52,39% | **55,67%** |
| Robot yếu nhất trên public do người dùng cung cấp | 43,33% | **49,44%** |
| File submission test | `v1_submitted/predictions.json` | `v2_candidate/predictions.json` |

Không so sánh điểm validation với public test như cùng một tập đánh giá.

File test v2 đã sinh đủ 12.000 số nguyên 0..3, cùng số dòng observations và
sample submission. Runtime 513,9 giây; 1.051 lượt strategy fallback, cùng mức
v1 vì CV/NLP chưa đổi. Sau khi nộp, người dùng báo điểm public v2 là 55,67%,
tăng 3,28 điểm phần trăm; robot yếu nhất tăng 6,11 điểm phần trăm.

Audit v2 đã hoàn tất trong 495,09 giây, dự đoán trùng v2 từng byte. Có 112 ảnh
fallback (69 ảnh x 9 robot và 43 ảnh x 10 robot). Cả 1.051 lượt đều không có
tuyến hoàn thành mission trên graph dự đoán. Resolver còn thay goal ở 245 ảnh,
trong đó parser không trích xuất được goal ở 96 ảnh. Xem
`v2_fallback_audit/diagnostics/summary.json` và `docs/FALLBACK_AUDIT.md` ở dự án.

Solver mới tăng 4,50 điểm phần trăm với đầu vào chuẩn và tăng 4,53 điểm phần
trăm cho toàn hệ thống. Benchmark mới có 0 fallback trên 300 cảnh validation;
CV scene exact vẫn là 257/300 = 85,67%. Tổng runtime cho bốn kịch bản
oracle/NLP/CV/full là 248,327 giây với giới hạn 2 luồng CPU.

Số liệu mới được lưu trực tiếp từ lần chạy vào
`v2_candidate/system_validation.json`. Số liệu end-to-end cũ được ghi lại từ
kết quả đã chạy trước khi cập nhật tại `v1_submitted/system_validation_recorded.json`.

## Nội dung

- `v1_submitted`: submission đã nộp, model solver cũ, báo cáo validation cũ và
  snapshot mã solver 34 profile từ Git.
- `v2_candidate`: model candidate ranker đã train lại bằng scikit-learn 1.7.2,
  bản gốc từ ZIP (scikit-learn 1.9.1), báo cáo oracle từ ZIP đã kiểm tra lại,
  mã nguồn sau tích hợp và báo cáo toàn hệ thống mới.
- `shared_cv_nlp/cv_artifacts`: các model CV dùng chung cho hai phiên bản.
  CV/NLP chưa thay đổi trong lần cập nhật solver này; mã CV/NLP nằm trong
  `v2_candidate/src/courier`.

Model solver cũ cần chạy với snapshot mã cũ trong
`v1_submitted/source/src`; schema 34 profile không khớp mã solver mới 48 profile.
Không thay model cũ bằng model mới ở cùng một đường dẫn.

## Chạy phiên bản mới

Từ `D:/phenikaa`:

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "2"
py -3.12 scripts/create_submission.py `
  --strategy-artifact results/solver_comparison_20261008/v2_candidate/candidate_strategy.joblib `
  --out results/solver_comparison_20261008/v2_candidate/predictions.json
```

Lệnh trên từ chối ghi nếu file output đã tồn tại. Khi cần chạy tiếp một bản
khác, dùng thư mục phiên bản mới. Nếu không truyền `--out`, script tự tạo
`results/submission_YYYYMMDD_HHMMSS/predictions.json`.

SHA-256 của submission cũ:
`3ACC42B280C0141FEEE713413FF225D965E8AFD335F4AC7CEF61628884333B67`.
