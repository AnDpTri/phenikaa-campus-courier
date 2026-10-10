# Báo cáo bàn giao hiện trạng — 08/10/2026

## 1. Tóm tắt

Mã nguồn hiện tại đã tích hợp ba nhóm thay đổi:

1. NLP neural nhỏ kết hợp parser luật (`rule + neural hybrid`).
2. Solver sửa đồ thị dự đoán khi goal/via không tới được.
3. Bộ phân loại style ảnh và định tuyến ảnh `print` sang detector 768 px.

Toàn bộ 71 unit test hiện có đều đạt. Tuy nhiên, candidate kết hợp NLP + solver repair + style routing được người dùng báo điểm Public Test chỉ **25,44%**, thấp hơn rất nhiều so với submission v2 đã biết (**55,67%**). Vì vậy chưa được coi các thay đổi mới là an toàn để nộp tiếp.

Base Git trước khi tích hợp các patch mới:

`e29d99a3a241a3d84e129ff9f12f564186bd0dc6`

## 2. Kết quả kiểm thử

### Unit test

- Lệnh: `py -3.12 -m unittest discover -s tests -v`
- Kết quả: **71/71 đạt**.

### NLP neural v1

- Số tham số: **4.789.324**.
- Validation grounded, parser luật: **100%**.
- Validation grounded, hybrid: **100%**.
- Holdout goal:
  - rule: **28,80%**;
  - neural: **69,43%**;
  - hybrid: **46,80%**.
- Ngưỡng được chọn: `goal=2.0`, `via=2.0`, `flags=2.0`; neural không override kết quả hợp lệ của rule, nhưng vẫn điền goal khi rule không tìm thấy.
- Trên test, diagnostics ghi nhận neural đã điền goal cho **85 cảnh**.

### Solver repair

Benchmark làm hỏng đồ thị validation có chủ đích:

- Số cảnh unreachable mô phỏng: **281**.
- Fallback cũ: **35,09%**.
- Graph repair: **53,20%**.
- Trường hợp lỗi one-way: **36,06% → 68,31%**.

Đây là số liệu mô phỏng, không phải đánh giá bằng nhãn test thật.

### CV style routing

- Style classifier train: **100%**.
- Style classifier validation: **300/300**.
- Baseline detector 512:
  - CV scene exact: **85,67%**;
  - full validation macro: **73,10%**.
- Route ảnh print sang detector 768:
  - CV scene exact: **90,33%**;
  - full validation macro: **73,27%**.

Validation cho thấy tăng nhẹ, nhưng chưa chứng minh tổng quát hóa sang test public.

## 3. Các submission đã tạo

### Submission v2 đã biết

- Đường dẫn: `results/submission_20261008_nlp57f5dd2_cv512_solverv2/predictions.json`
- SHA-256: `D5675EC638F7F0A8F021BE463F3C279888BBD668C7132962CF6D4A0F30553AED`
- Điểm Public Test đã biết: **55,67%**.

### NLP + solver repair

- Đường dẫn: `results/submission_nlp_repair_20261008/predictions.json`
- SHA-256: `35508B292B51EBC80068F3AD92BAC9CD41E610BD9BF0EE4A01FBFB9BFA0E0925`
- Khác v2: **844/12.000** dự đoán.
- Chưa có điểm public riêng.

### NLP + solver repair + style routing

- Đường dẫn: `results/submission_nlp_repair_style_20261008/predictions.json`
- SHA-256: `B90F17670B3E0047774244914FFC9EE4F2A0B37A546FA476D59318883F966FB9`
- Khác v2: **907/12.000** dự đoán.
- Khác bản không style: **69/12.000** dự đoán.
- Điểm Public Test người dùng báo: **25,44%**.

## 4. Đính chính về điểm public và việc cần kiểm chứng

Đề bài mục 7 ghi rõ test gồm cả cảnh được chấm và không chấm; leaderboard public chỉ dùng khoảng 30% số cảnh được chấm. Vì thế giới hạn `907 / 12000 = 7,56` điểm phần trăm CHỈ đúng nếu chấm toàn bộ 1.200 cảnh, KHÔNG phải giới hạn của điểm public. Kết luận trước đây rằng 25,44% là bất khả thi đã dùng sai mẫu số và được rút lại. Nếu các thay đổi tập trung ở cảnh public thì mức giảm có thể lớn hơn nhiều.

Cần kiểm chứng các khả năng sau, chưa có bằng chứng để quy lỗi cho file upload hay hệ thống chấm:

1. File thực tế upload có đúng SHA `B90F...` hay không.
2. Điểm 55,67% trước đó có đúng thuộc file SHA `D567...` hay không.
3. Hai lượt có được chấm trên cùng bộ dữ liệu, thứ tự observation và công thức macro accuracy hay không.
4. Trang nộp có biến đổi/đọc nhầm cấu trúc file hoặc người dùng đã chọn nhầm file khác hay không.

Không nên vội dùng thêm lượt nộp cho bản `35508...`. Nó chỉ khác bản bị báo 25,44% ở 69 vị trí trên toàn test, nhưng cũng chưa thể dùng tỷ lệ 69/12000 làm giới hạn điểm public. `scripts/compare_submissions.py` đã được chỉnh để chỉ tính giới hạn điểm khi biết số cảnh được chấm hoặc xác nhận chấm toàn bộ test.

## 5. Điểm yếu của validation hiện tại

- Validation sạch không kích hoạt solver repair, nên số **73,27%** không đo trực tiếp chất lượng repair trên lỗi CV thật của test.
- Benchmark repair dùng lỗi mô phỏng và có thể không giống lỗi detector trên test.
- Style classifier đạt 100% trên validation nhưng vẫn có thể gặp domain shift trên test.
- NLP holdout được sinh từ ngữ pháp viết tay cùng nguồn với train, nên kết quả có thiên lệch tác giả.
- Vì không có nhãn test, mọi cải thiện test ngoài leaderboard chỉ là ước lượng.

## 6. Khuyến nghị cho nhóm tiếp quản

1. Xác minh SHA của file đã upload, đúng vòng COURIER2 và lịch sử điểm; đồng thời xem xét khả năng regression tập trung vào các cảnh được chấm.
2. Dùng v2 SHA `D567...` làm mốc bất biến.
3. Tạo từng ablation riêng từ đúng mốc v2: repair-only, NLP-only, style-only; mỗi file phải có số lượng thay đổi và SHA trước khi nộp.
4. Thêm replay test bảo đảm candidate chỉ thay đúng các observation dự kiến.
5. Không dùng test để train, gán nhãn, chọn threshold hoặc viết luật.
6. Không nộp đồng thời nhiều thay đổi chưa được ablation vì không xác định được nguyên nhân tăng/giảm điểm.

## 7. Nội dung gói bàn giao

Gói source chỉ chứa mã nguồn, script, test, tài liệu và cấu hình dự án. Không chứa:

- dữ liệu train/validation/test;
- artifact model `.pt`/`.joblib`;
- submission/predictions;
- API key;
- thư mục tạm giải nén;
- shortcut cá nhân.

