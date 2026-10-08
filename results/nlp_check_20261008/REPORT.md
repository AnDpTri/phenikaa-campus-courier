# Kiểm tra NLP — 08/10/2026

## Phạm vi và cách đo

Chạy parser và resolver hiện tại với landmark chuẩn của train/validation, để tách lỗi NLP khỏi CV. Không sửa mã NLP, không thay submission, không thêm quy tắc từ test. Kết quả được lưu riêng tại thư mục này.

Chỉ số “đúng toàn bộ” yêu cầu tập node đích, tập node ghé qua, urgent và fragile cùng đúng. Đây không phải điểm hành động của solver hay điểm leaderboard. Chỉ số goal_ref của script gốc chỉ so loại tham chiếu; goal_nodes kiểm tra tập node đích sau khi resolve.

## Kết quả dữ liệu gốc

| Chỉ số | Train (2.000 cảnh) | Validation (300 cảnh) |
| --- | ---: | ---: |
| Node đích | 2.000/2.000 — 100% | 300/300 — 100% |
| Loại đích | 2.000/2.000 — 100% | 300/300 — 100% |
| Loại tham chiếu đích | 2.000/2.000 — 100% | 300/300 — 100% |
| Node ghé qua | 1.999/2.000 — 99,95% | 300/300 — 100% |
| Urgent | 2.000/2.000 — 100% | 300/300 — 100% |
| Fragile | 2.000/2.000 — 100% | 300/300 — 100% |
| Đúng toàn bộ | 1.999/2.000 — 99,95% | 300/300 — 100% |

18/18 unit test NLP đạt. Nội dung vocabulary hiện tại trùng vocabulary dựng lại từ train (241 unigram, 1.333 bigram); khác hash do cách trình bày JSON, không phải khác nội dung.

Lỗi duy nhất trên train là train-01552: “ghe khu the tao lay bo vot truoc”. Nhãn yêu cầu ghé sports, nhưng parser không nhận “the tao” thành “the thao”, nên bỏ via. Bộ sửa lỗi giữ nguyên một từ vốn có trong vocabulary nếu thiếu bằng chứng ngữ cảnh đủ mạnh.

## Thử độ bền có kiểm soát

Biến đổi văn bản train/validation và giữ nhãn ban đầu. Đây là stress test nhân tạo, không phải ước lượng độ chính xác test. Bỏ dấu câu cũng có thể làm mất tín hiệu phạm vi và khiến câu nhập nhằng hơn.

| Biến thể | Train đúng toàn bộ | Validation đúng toàn bộ |
| --- | ---: | ---: |
| Gốc | 99,95% | 100% |
| Bỏ dấu tiếng Việt / chuẩn hóa fold | 99,95% | 100% |
| Bỏ dấu ngắt câu . ! ? ; ( ) [ ] | 92,45% (1.849/2.000) | 91% (273/300) |
| Thêm lỗi xóa/đảo ký tự với xác suất 5% mỗi từ đủ dài | 99,65% (1.993/2.000) | 100% (300/300) |

Khi bỏ dấu ngắt câu, parser không lấy được goal ở 121 cảnh train và 13 cảnh validation; resolver vẫn tạo goal thay thế. Kết quả typo chỉ áp dụng cho một seed và kiểu nhiễu này, không bảo đảm mọi dạng lỗi chính tả.

## Điểm yếu hiện tại

NLP là parser dựa trên từ điển/quy tắc và bộ sửa lỗi, không phải mô hình ngôn ngữ đã huấn luyện. Việc tách câu, nhận phủ định, xác định goal/via và lựa chọn đề cập cuối cùng có thể nhạy với cách diễn đạt và dấu câu.

Resolver che một số lỗi: khi goal thiếu hoặc loại đích không xuất hiện trong landmark, nó chọn đích thay thế; khi via không có trên bản đồ, nó bỏ via. Thiếu anchor cũng có thể làm mất tham chiếu gần/xa. Vì vậy “pipeline vẫn chạy” không đồng nghĩa hiểu đúng nhiệm vụ.

Audit test đã có trước đó cho thấy 96/1.200 cảnh không có parsed goal; 245/1.200 cảnh dùng goal thay thế, trong đó 148 cảnh yêu cầu một loại địa điểm không có trong landmark CV. Đây là tín hiệu chẩn đoán, không phải tỷ lệ lỗi NLP: test không có nhãn chuẩn và thiếu địa điểm có thể do CV, NLP hoặc dữ liệu. Không đọc thêm văn bản test để xây quy tắc trong lần kiểm tra này.

## Hướng cải thiện

1. Hiển thị riêng trạng thái goal/via/anchor không giải được và nguyên nhân thay thế, tránh coi fallback là parse thành công.
2. Đo và cải thiện phạm vi phủ định, tin cũ/tin mới, goal/via trên train và validation; bổ sung thử nghiệm biến thể dấu câu.
3. Cải thiện sửa lỗi theo cụm/ngữ cảnh cho trường hợp từ sai nhưng vẫn là từ hợp lệ, kiểm tra cả nguy cơ sửa nhầm.
4. Tách phép đo NLP với landmark chuẩn khỏi phép đo end-to-end dùng landmark CV; không dùng kết quả validation 100% để khẳng định test 100%.

Chi tiết máy đọc: metrics.json. Lỗi train: train_errors.json. Lỗi validation: validation_errors.json. Script stress test: check.py.
