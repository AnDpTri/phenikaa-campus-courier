# Báo cáo hiện trạng solver

Ngày cập nhật: 2026-10-08 (Asia/Saigon)

## 1. Phạm vi

Solver chịu trách nhiệm dự đoán hành động đầu tiên của từng robot khi đã có:

- `SceneGraph` chính xác từ CV;
- `Mission` chính xác từ NLP;
- `robot_id` từ 0 đến 9.

Báo cáo này đánh giá riêng tầng chiến thuật bằng **oracle graph + oracle
mission**. Vì vậy, kết quả dưới đây chưa bao gồm lỗi truyền từ CV và NLP và
không phải độ chính xác end-to-end của toàn hệ thống.

## 2. Kiến trúc hiện tại

Pipeline solver gồm hai tầng:

1. `OracleSolver` dựng đồ thị di chuyển hợp lệ và tính chi phí đường đi:
   - loại đường đóng;
   - tuân thủ đường một chiều;
   - chỉ robot 4 được đi bậc thang;
   - bắt buộc đi qua `via` trước `goal`;
   - hỗ trợ nhiều ứng viên địa điểm và tham chiếu không gian.
2. `OracleStrategyModel` dùng mười mô hình độc lập, mỗi robot một mô hình,
   rồi giới hạn đầu ra trong tập hành động hợp lệ.

Đặc trưng hiện tại có 646 chiều, gồm:

- kích thước và mật độ bản đồ;
- vị trí, hướng quay và các cạnh kề robot;
- thời tiết, `urgent`, `fragile`, `via`, loại địa điểm và tham chiếu;
- chi phí, regret, thứ hạng và cờ tối ưu của từng hành động dưới 34 hồ sơ
  chi phí dựng sẵn.

Mỗi robot được chọn một trong ba họ mô hình bằng bốn fold stratified CV trên
train: Random Forest, Extra Trees hoặc Histogram Gradient Boosting. Artifact
cuối chỉ chứa mười mô hình được chọn, không chứa các mô hình CV trung gian.

## 3. Kết quả hiện tại

Benchmark trên 300 cảnh validation:

| Phương pháp | Macro accuracy |
|---|---:|
| Đường ngắn nhất đồng nhất | 55,70% |
| Strategy model trước | 68,87% |
| Per-robot train-CV hiện tại | **69,10%** |

Mô hình hiện tại tăng 13,40 điểm phần trăm so với baseline đường ngắn nhất,
nhưng chỉ tăng 0,23 điểm so với strategy model trước. Chênh lệch này tương
đương khoảng bảy dự đoán trên tổng số 3.000 dự đoán validation và chưa đủ lớn
để coi là một bước tiến chắc chắn.

### 3.1. Theo từng robot

| Robot | Họ mô hình | CV train | Validation | Chênh lệch |
|---|---|---:|---:|---:|
| R0 | Random Forest | 86,05% | **84,33%** | -1,72 |
| R1 | Random Forest | 69,10% | **60,67%** | -8,43 |
| R2 | HistGradientBoosting | 74,80% | **70,67%** | -4,13 |
| R3 | Extra Trees | 72,70% | **64,00%** | -8,70 |
| R4 | HistGradientBoosting | 69,10% | **62,33%** | -6,77 |
| R5 | Random Forest | 75,50% | **64,67%** | -10,83 |
| R6 | HistGradientBoosting | 78,20% | **76,33%** | -1,87 |
| R7 | HistGradientBoosting | 76,85% | **63,33%** | -13,52 |
| R8 | HistGradientBoosting | 80,15% | **72,67%** | -7,48 |
| R9 | HistGradientBoosting | 72,40% | **72,00%** | -0,40 |
| **Macro** |  | **75,49%** | **69,10%** | **-6,39** |

Độ chính xác khi chấm lại chính tập train là 99,82%. Khoảng cách lớn giữa
train, CV và validation cho thấy mô hình đang overfit.

### 3.2. Theo nhóm cảnh validation

| Nhóm | Số cảnh | Macro accuracy |
|---|---:|---:|
| Tất cả | 300 | **69,10%** |
| Có `via` | 156 | 69,62% |
| Không `via` | 144 | 68,54% |
| Mưa | 128 | 68,44% |
| Khô | 172 | 69,59% |
| Có `goal_ref` | 229 | 68,78% |
| Không `goal_ref` | 71 | 70,14% |

Thời gian dự đoán toàn bộ validation là 18,86 giây, tương đương khoảng 63 ms
mỗi cảnh trên máy phát triển.

## 4. Những phần đang hoạt động tốt

- Toàn bộ 10 test solver hiện đều qua.
- Nhãn train đều là hành động hợp lệ theo graph.
- Đường đóng, một chiều, cầu thang và thứ tự `via -> goal` được xử lý đúng.
- Fallback khi thiếu đích được ghi rõ lý do và vẫn chỉ trả hành động hợp lệ.
- Cache đặc trưng có schema và provenance; artifact có dataset digest, phiên
  bản Python/scikit-learn, family, CV score và validation score.
- R0 luôn được giữ trong tập hành động thuộc đường ngắn nhất trên validation.
- Kích thước artifact khoảng 86,7 MB, vẫn an toàn dưới giới hạn 200 triệu tham
  số của đề.

Các test hiện tại chủ yếu chứng minh tính đúng đắn của graph và ràng buộc di
chuyển. Chúng chưa chứng minh mô hình đã khôi phục đúng chiến thuật bí mật của
từng robot.

## 5. Phân tích nút thắt

### 5.1. Distribution shift giữa train và validation

Validation được thiết kế khó hơn train và khác rõ về phân phối:

| Đặc điểm | Train | Validation |
|---|---:|---:|
| Diện tích lưới trung bình | 44,51 | 56,96 |
| Số node trung bình | 43,10 | 53,81 |
| Số cạnh trung bình | 61,24 | 75,93 |
| Tỷ lệ có `via` | 32,70% | 52,00% |
| Tỷ lệ có `goal_ref` | 58,15% | 76,33% |
| Tỷ lệ có `via_ref` | 3,35% | 13,33% |

Stratified K-fold ngẫu nhiên trên train không mô phỏng được dịch chuyển này,
nên CV hiện đánh giá quá lạc quan.

### 5.2. Mô hình quay về đường ngắn nhất quá thường xuyên

Khi classifier không chắc chắn, dự đoán có xu hướng giống R0 thay vì thực hiện
chiến thuật riêng của robot. Ví dụ:

| Robot | Nhãn thật thuộc đường ngắn nhất | Dự đoán thuộc đường ngắn nhất |
|---|---:|---:|
| R1 | 76,33% | 89,33% |
| R3 | 70,00% | 86,67% |
| R7 | 69,67% | 86,33% |

Đây là một nguyên nhân trực tiếp khiến các robot có chiến thuật mạnh như né
đường đông, phụ thuộc thời tiết hoặc phụ thuộc hàng dễ vỡ bị kéo về baseline.

### 5.3. Thư viện profile chưa bao phủ đủ chiến thuật

Tỷ lệ cảnh mà nhãn đúng là tối ưu dưới ít nhất một trong 34 profile hiện tại:

| Robot | Được profile bao phủ |
|---|---:|
| R0 | 100,00% |
| R1 | 89,67% |
| R2 | 94,33% |
| R3 | 86,00% |
| R4 | 93,67% |
| R5 | 79,00% |
| R6 | 81,33% |
| R7 | 83,00% |
| R8 | 85,67% |
| R9 | 71,67% |

R9 là trường hợp rõ nhất: gợi ý công khai nói robot chỉ nhìn một bước, nhưng
phần lớn đặc trưng profile hiện tại lại tối ưu cả tuyến đường. R5, R6 và R7
cũng cần không gian chiến thuật mới thay vì chỉ thêm cây vào classifier.

### 5.4. Phá hòa đang bị trộn với chọn chiến thuật

Với R0:

- cảnh có duy nhất một hành động ngắn nhất: 100% accuracy;
- cảnh có nhiều hành động ngắn nhất: 59,48% accuracy;
- tỷ lệ cảnh có hòa: 38,67%.

Graph solver đã tìm đúng tập hành động tối ưu; phần sai còn lại chủ yếu nằm ở
quy tắc phá hòa bí mật. Chọn chiến thuật và phá hòa cần được tách thành hai mô
hình hoặc hai tầng luật riêng.

## 6. Hướng cải thiện đề xuất

### Ưu tiên 1: học điểm cho từng hành động hợp lệ

Thay phân loại trực tiếp bốn hướng bằng candidate ranking:

1. tạo một mẫu cho mỗi hành động hợp lệ;
2. trích xuất đặc trưng riêng của hành động và tuyến tốt nhất bắt đầu bằng hành
   động đó;
3. học utility hoặc xác suất thắng của từng ứng viên;
4. chọn ứng viên có điểm cao nhất.

Cách này tăng số mẫu huấn luyện hữu ích, giảm phụ thuộc vào hướng tuyệt đối và
phù hợp hơn với bản chất tối ưu có ràng buộc của bài toán.

### Ưu tiên 2: lưu thành phần đường đi thay vì chỉ lưu kết quả profile

Cho mỗi hành động cần có tối thiểu:

- số bước tới `via` và `goal`;
- số đoạn normal/crowded/covered/stairs;
- số lần đi thẳng, rẽ trái, rẽ phải và quay đầu;
- hướng đầu tiên so với heading hiện tại;
- thuộc tính cạnh đầu tiên;
- đích cụ thể được chọn khi có hai địa điểm cùng loại;
- chênh lệch chi phí với ứng viên tốt thứ hai;
- đặc trưng riêng cho greedy one-step.

Các giá trị thành phần tổng quát hóa sang bản đồ lớn tốt hơn 34 profile rời rạc
và cho phép suy ra trọng số chưa được liệt kê trước.

### Ưu tiên 3: mô hình hóa điều kiện riêng của từng robot

- R0: shortest path chính xác + tie-breaker riêng.
- R1: số đoạn đông và ưu tiên né đông dạng lexicographic.
- R2: lợi ích mái che so với độ dài đường vòng.
- R3: policy riêng cho mưa và khô.
- R4: giá trị của bậc thang và tie-breaker riêng.
- R5: số lần đổi hướng, chuỗi đi thẳng và heading ban đầu.
- R6: policy riêng cho `urgent=true/false`.
- R7: policy riêng cho `fragile=true/false`.
- R8: hướng rẽ tương đối tại giao lộ và trên tuyến.
- R9: heuristic một bước, không dùng full-route profile làm tín hiệu chính.

### Ưu tiên 4: tách tie-breaker

Pipeline mục tiêu:

```text
scene -> utility/profile -> tập hành động tối ưu -> tie-breaker của robot
```

Tie-breaker có thể dùng hướng tuyệt đối, hướng tương đối so với heading, vị trí
robot, điều kiện nhiệm vụ và đặc trưng giao lộ. R0 là robot nên triển khai đầu
tiên vì phần utility đã được xác định chính xác.

### Ưu tiên 5: đánh giá bằng hard split

Bên cạnh random CV, cần có các split mô phỏng private test:

- train bản đồ nhỏ, đánh giá bản đồ lớn;
- train không `via`, đánh giá có `via`;
- giữ riêng cảnh có `goal_ref` và `via_ref`;
- giữ riêng giao lộ có ba hoặc bốn hành động hợp lệ.

Chỉ chấp nhận thay đổi nếu cải thiện macro validation và không làm giảm hard
split. Sau khi khóa kiến trúc, có thể huấn luyện artifact nộp cuối bằng cả
train và validation nếu quy trình cuộc thi cho phép; tuyệt đối không sử dụng
nội dung test để huấn luyện hoặc viết luật.

## 7. Thứ tự triển khai

1. Bổ sung route-component features và candidate-level dataset.
2. Làm R0 shortest-path + tie-breaker làm mẫu kiến trúc.
3. Làm R9 greedy one-step vì profile hiện tại bao phủ kém nhất.
4. Tách policy điều kiện cho R3, R6 và R7.
5. Tối ưu R1, R4 và R5, là nhóm có validation thấp và gap lớn.
6. Chuyển R2 và R8 sang candidate ranker.
7. Thêm hard-split evaluation, ablation và kiểm tra deterministic artifact.
8. Chỉ sau đó mới tinh chỉnh family/hyperparameter hoặc ensemble.

Vì điểm là macro trung bình của mười robot, nâng năm robot yếu nhất thêm trung
bình 10 điểm phần trăm sẽ tăng khoảng 5 điểm macro, hiệu quả hơn tối ưu tiếp
những robot đã ổn định như R6.

## 8. Tái lập kết quả

Từ thư mục gốc dự án, dùng Python 3.12:

```powershell
$env:PYTHONPATH = "src"
py -3.12 scripts/solver/train_oracle_strategy.py
py -3.12 scripts/solver/evaluate_strategy_model.py `
  --split validation `
  --report artifacts/solver/validation_report.json
py -3.12 -m unittest discover -s tests/solver -v
```

Các file sinh ra được chủ động bỏ khỏi Git:

- `artifacts/solver/oracle_strategy.joblib` (~86,7 MB);
- `artifacts/solver/validation_report.json`;
- `data/solver/oracle_features.npz`.

Artifact cần được tái tạo từ mã nguồn và dữ liệu tương ứng. Không dùng
validation score 69,10% như ước lượng không chệch cho private test vì
validation đã được phân tích trong quá trình phát triển.
