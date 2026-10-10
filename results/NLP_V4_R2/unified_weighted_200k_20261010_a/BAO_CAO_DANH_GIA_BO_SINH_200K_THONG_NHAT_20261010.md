# BÁO CÁO ĐÁNH GIÁ BỘ SINH DỮ LIỆU NLP THỐNG NHẤT 200K

**Dự án:** Phenikaa Campus Courier — NLP V4 R2  
**Phiên bản bộ sinh:** Unified Weighted V4  
**Ngày báo cáo:** 10/10/2026  
**Phạm vi:** Chất lượng bộ sinh và corpus sinh ra; không đánh giá kết quả huấn luyện mô hình.

## 1. Đánh giá tổng quan

**Kết luận:** Bộ sinh thống nhất có lợi thế rõ ràng về **độ phủ ngữ nghĩa, tình huống suy luận khó và mức cân bằng các tổ hợp nhãn** so với cấu hình V2 thuần đã đối chiếu. Corpus 200.000 câu đã đạt các phép kiểm tra kỹ thuật và hợp đồng ngữ nghĩa có thể tự động hóa, đồng thời tương thích với pipeline huấn luyện hiện tại.

Tuy nhiên, **chưa đủ cơ sở kết luận dữ liệu mới giúp mô hình đạt accuracy cao hơn V2**. Độ đa dạng bề mặt ngôn ngữ tính theo tiền tố tám từ giảm đáng kể; phân bổ kỹ năng mới chỉ là giả thuyết thiết kế, chưa được hiệu chỉnh bằng thử nghiệm mô hình. Do đó, dữ liệu phù hợp để **thực hiện một thí nghiệm fine-tune có kiểm soát**, không nên coi đây là chất lượng tối ưu cuối cùng.

Mô hình dữ liệu là **một corpus duy nhất 200.000 câu**, không có tập 40K chuyên biệt ghép riêng. Các trường hợp ngôn ngữ thường, không gian, điểm lấy, sửa chỉ dẫn và gây nhiễu được đan xen xuyên suốt tập.

## 2. Kiến trúc và đầu ra

### 2.1. Bộ sinh

- **Bản mã:** synth_weighted_v4.py.
- **Nguồn nền:** bản sao byte-identical của V2, tên synth_v2_clone.py. SHA-256 bản gốc và bản sao trùng nhau.
- **Chế độ v2:** giữ nguyên hành vi V2 theo seed và holdout; đã đối chiếu thành công trên các mẫu thử.
- **Chế độ weighted:** tạo câu theo trọng số nhiệm vụ; cho phép sinh mới bằng seed khác.
- **Tương thích:** nhãn goal, via, urgent, fragile; tám đầu phân loại goal_mode, goal_type, goal_anchor, via_mode, via_type, via_anchor, urgent, fragile.
- **Nguyên tắc:** lấy mẫu trực tiếp từ bộ sinh thống nhất, **không ghép lại các tệp 40K/200K cũ**.

### 2.2. Tệp dữ liệu chính

**Corpus huấn luyện:**  
D:\phenikaa\results\NLP_V4_R2\unified_weighted_200k_20261010_a\train_weighted_unified_200000_seed2026101060.jsonl

**Số bản ghi:** 200.000, JSONL.  
**Seed:** 2026101060.  
**SHA-256:** da03ac6bb3498b177ed238855489212c23df895beb4b675b7920f5ea15b1645a

Các tệp bổ trợ dùng cho truy xuất nguồn gốc và QA (không phải dữ liệu huấn luyện bổ sung):

- train_weighted_unified_200000_roles_seed2026101060.jsonl
- train_weighted_unified_200000_manifest.json
- independent_weighted_200000_quality_report.json
- additional_holdout_sampling_audit_v2.json

## 3. Phân bổ trọng số theo nhiệm vụ

| Trọng tâm tạo dữ liệu | Tỷ trọng | Số câu |
|---|---:|---:|
| Ngôn ngữ V2 tổng quát, nhiễu và cách diễn đạt phong phú | 42% | 84.000 |
| Nhiệm vụ kết hợp nhiều điều kiện/ngữ cảnh | 18% | 36.000 |
| Phân biệt có/không có điểm lấy hàng (via) | 15% | 30.000 |
| Quan hệ không gian, mốc và hướng | 12% | 24.000 |
| Hủy hoặc đính chính chỉ dẫn cũ | 9% | 18.000 |
| Phủ định, địa điểm gây nhiễu và thông tin không hiệu lực | 4% | 8.000 |
| **Tổng** | **100%** | **200.000** |

Đây là **trọng tâm sinh câu**, không phải các lớp ngữ nghĩa tách biệt: một câu thuộc nhóm không gian vẫn có thể chứa via, phủ định, sửa lệnh và cờ hàng dễ vỡ.

**Tính đồng đều trong file:** mười đoạn liên tiếp, mỗi đoạn 20.000 câu, đều được kiểm chứng có đúng phân bổ 8.400 / 3.600 / 3.000 / 2.400 / 1.800 / 800 theo thứ tự bảng trên. Không dồn tất cả câu khó vào một khối riêng.

**Lưu ý:** Bộ trọng số 42/18/15/12/9/4 xuất phát từ ưu tiên kỹ thuật; **chưa phải trọng số tối ưu được chứng minh bằng thực nghiệm**.

## 4. Độ phủ dữ liệu thực tế

| Đặc điểm ngữ nghĩa | Số câu / lượt đề cập | Tỷ lệ trên 200K khi áp dụng |
|---|---:|---:|
| Có via thực sự | 113.803 | 56,90% |
| Goal có quan hệ không gian | 137.344 | 68,67% |
| Via có quan hệ không gian | 33.743 | 16,87% |
| Hủy/thay đổi đích giao trước đó | 35.815 | 17,91% |
| Hủy/thay đổi điểm lấy trước đó | 23.241 | 11,62% |
| Vừa sửa đích giao vừa có via | 18.796 | 9,40% |
| Vai trò địa điểm gây nhiễu | 85.429 lượt | Không quy đổi thành tỷ lệ câu độc lập |
| Có via nhưng không xuất hiện “trước” | 73.302 | 36,65% toàn tập |
| Không có via nhưng xuất hiện “lấy” | 30.319 | 15,16% toàn tập |
| urgent=True | 91.341 | 45,67% |
| fragile=True | 85.377 | 42,69% |
| Có dấu tiếng Việt | 64.925 | 32,46% |

**Độ dài câu:** trung vị 63 token; P90 = 88; P99 = 108; tối đa 149 token. Toàn bộ nằm trong giới hạn của encoder.

**Không gian tổ hợp nhãn:** đã quan sát đủ **384/384** tổ hợp ở mức (goal_mode × via_mode × urgent × fragile). Trong số đó **126 tổ hợp có dưới 50 ví dụ**, ít hơn cấu hình V2 thuần được đối chiếu (198 tổ hợp); tổ hợp hiếm nhất có 16 ví dụ, so với 3 ở V2 thuần.

## 5. Ưu điểm

### 5.1. Đúng trọng tâm suy luận của bài toán courier

- **Goal/via và thứ tự hành động:** có cả điểm lấy thực, không có điểm lấy, hàng đã nhận tại nơi xuất phát và các diễn đạt lấy hàng không sử dụng từ khóa “trước”.
- **Đính chính lệnh:** phân biệt địa điểm cũ bị hủy với đích giao hoặc điểm lấy còn hiệu lực.
- **Ngữ cảnh gây nhiễu:** hỗ trợ các vai trò như địa điểm cũ, không cần ghé, không liên quan, đóng cửa, người nhận đã chuyển.
- **Tham chiếu không gian:** bao phủ near/far, hướng, mốc anchor và cực trị trong các nhãn hiện được V2/V4 hỗ trợ.
- **Nhiệm vụ kết hợp:** nhiều yếu tố trên có thể cùng xuất hiện trong một câu, tăng độ khó suy luận so với câu một mệnh đề.

### 5.2. Giảm học tắt theo từ khóa

Dữ liệu có số lượng đáng kể các trường hợp phản ví dụ: **73.302 câu có via không cần từ “trước”** và **30.319 câu không có via mặc dù chứa “lấy”**. Điều này hữu ích cho việc chống dự đoán nhãn chỉ dựa vào một từ nổi bật.

### 5.3. Cân bằng các tổ hợp nhãn tốt hơn

Cả V2 thuần và bản mới có 384/384 tổ hợp cơ bản, nhưng bản mới có **ít ô rất thưa hơn** (126 so với 198 ô dưới 50 mẫu). Đây là cải thiện về **mật độ phủ nhãn**, không phải bằng chứng trực tiếp cho accuracy.

### 5.4. Chất lượng kỹ thuật và khả năng tái lập

- **200.000/200.000** câu đọc được, đúng schema và mã hóa được tám đầu nhãn.
- **Không phát hiện vi phạm** trong các hợp đồng ngữ nghĩa và mẫu lỗi ngữ pháp đã mã hóa.
- **Không có câu trùng nguyên văn sau chuẩn hóa** bên trong tập; các nguồn tham chiếu cũ đã được dùng để lọc trùng khi sinh.
- Giới hạn **không quá 35 lần cho cùng tiền tố tám từ** nhằm ngăn một số khuôn mở đầu lấn át dữ liệu.
- Tập **5.000 câu holdout** sinh riêng có **0 trùng nguyên văn sau chuẩn hóa** với tập train; việc này không loại trừ trùng họ mẫu hoặc câu gần nghĩa.
- Trainer hiện tại nạp đủ 200.000 câu, tạo tám tensor nhãn có đủ 200.000 phần tử mỗi đầu; một batch 256 câu mã hóa thành công với kích thước **[256, 123, 24]**.
- Mã gốc V2 và checkpoint không bị thay đổi.

## 6. Hạn chế và rủi ro

### 6.1. Độ đa dạng bề mặt thấp hơn V2 thuần

So sánh ở cùng quy mô 200.000 câu:

| Chỉ số | V2 thuần | Bản weighted mới |
|---|---:|---:|
| Tiền tố tám từ khác nhau | **146.819** | 91.784 |
| Tiền tố bốn từ khác nhau | **51.806** | 27.169 |
| Tỷ lệ câu thuộc nhóm tiền tố tám từ xuất hiện hơn một lần | 34,86% | **62,28%** |
| Tần suất lớn nhất của cùng tiền tố tám từ | 172 | **35** |

Giới hạn tần suất đã giảm những mở đầu quá phổ biến, **nhưng không bù lại sự sụt giảm về số kiểu mở đầu khác nhau**. Các ngân hàng diễn đạt ngữ nghĩa mới vẫn có nguy cơ lặp khuôn. Cần bổ sung nhiều cấu trúc câu độc lập hơn, thay vì chỉ đổi tên địa điểm và danh từ.

### 6.2. Tỷ trọng chưa được tối ưu bằng kết quả mô hình

Tập weighted có via trong **56,90% câu**, cao hơn V2 thuần (~50,05%) và bộ tham chiếu 40K (~47,85%). Điều này có thể giúp nhận diện via, nhưng cũng có nguy cơ đẩy mô hình về phía dự đoán via quá thường xuyên. Cần theo dõi **false-positive via**, không chỉ recall.

Phần ngữ nghĩa khó chiếm tỷ trọng đáng kể; hiện **chưa có ablation** xác nhận 42/18/15/12/9/4 tốt hơn các tỷ lệ khác. Không nên xem việc tăng tỷ trọng câu sửa lệnh là luôn có lợi.

### 6.3. Dữ liệu có dấu còn hạn chế

**32,46%** câu có dấu và **67,54%** không có dấu. Đây là cải thiện so với V2 thuần trong mẫu đối chiếu (0% có dấu), nhưng chưa xác nhận phù hợp phân phối văn bản thực tế hoặc bộ đánh giá mục tiêu. Nếu đầu vào thực tế phần lớn có dấu, phân phối này có thể chưa tối ưu.

### 6.4. Độ dài và mức phức tạp tăng

So với V2 thuần, độ dài câu tăng từ **trung vị 51 → 63** token và **P90 71 → 88** token. Điều này giúp tăng số tình huống nhiều mệnh đề nhưng có thể làm giảm hiệu quả trên câu ngắn nếu fine-tune thiếu kiểm soát.

### 6.5. Không thể chứng minh ngữ nghĩa tuyệt đối chỉ bằng kiểm tra tự động

QA trên 200K kiểm chứng cấu trúc nhãn, các mốc, chứng cứ từ ngữ và những mẫu lỗi đã biết; **không thể chứng minh từng câu diễn đạt tự nhiên hoặc hàm ý đúng hoàn toàn**. Có 333 câu được chọn phân tầng trải rộng toàn tập; 27 câu được đọc trực tiếp trong lượt rà cuối. Chưa có gán nhãn đánh giá độc lập bởi con người trên một mẫu ngẫu nhiên đủ lớn.

### 6.6. Rủi ro tổng quát hóa và leakage dạng họ mẫu

Kiểm tra trùng câu sau chuẩn hóa **không phát hiện được** trường hợp cùng một template chỉ thay địa điểm, từ đồng nghĩa hoặc câu gần nghĩa rơi vào cả train và holdout. Cần tách holdout theo **họ kịch bản/template**, không chỉ theo seed và text.

### 6.7. Chưa đo cải thiện accuracy

**Chưa có thử nghiệm huấn luyện và đánh giá mô hình** cho corpus 200K này. Vì vậy chưa thể kết luận:
- Accuracy tổng thể tốt hơn V2;
- Goal/via thực tế cải thiện bao nhiêu;
- Các cấu trúc khó có gây suy giảm hiệu suất trên câu thường hoặc câu ngắn hay không;
- Tăng dữ liệu lên 250K/300K có cần thiết không.

## 7. Quyết định sử dụng và kiến nghị

**Mức chấp nhận hiện tại:** Đạt **data-side automated QA**, phù hợp cho một **thí nghiệm fine-tune có kiểm soát**. Chưa chứng minh là tập tối ưu hoặc cho điểm Kaggle cao hơn.

**Ưu tiên 1 — Thử nghiệm trước khi tăng quy mô.** Dùng cùng checkpoint V2 và quy trình validation/holdout để so:
- V2 thuần 200K và weighted 200K;
- Goal exact, via exact, tất cả đầu nhãn chính xác;
- Via false-positive, đặc biệt câu có “lấy” nhưng via=null;
- Nhóm câu ngắn, có dấu/không dấu, không gian, sửa chỉ dẫn, nhiều địa điểm phủ định;
- Các hard-case đã biết mà V4 R1 từng suy giảm.

**Ưu tiên 2 — Tăng đa dạng ngôn ngữ thay vì tăng số dòng ngay.** Tạo thêm ngân hàng câu có cấu trúc độc lập cho các kịch bản khó; đánh giá tiền tố bốn/tám từ, họ template và near-duplicate. Đây là điểm yếu đo được nổi bật nhất.

**Ưu tiên 3 — Chỉ hiệu chỉnh tỷ trọng hoặc tăng quy mô khi có bằng chứng.** Nếu các nhóm hiếm vẫn thiếu hoặc kết quả cho thấy underfitting, có thể cân nhắc thêm ví dụ có mục tiêu. **Không khuyến nghị tăng đại trà lên 300K** chỉ bằng cách sinh thêm từ cùng các ngân hàng mẫu hiện tại.

**Đánh giá cuối cùng:** Bộ sinh mới **tốt hơn về độ phủ suy luận và cân bằng nhãn**, nhưng **chưa tốt hơn toàn diện về đa dạng ngôn ngữ**; ảnh hưởng thực tế lên mô hình vẫn là giả thuyết cần kiểm chứng.

## 8. Tệp và trạng thái

**Thư mục:** D:\phenikaa\results\NLP_V4_R2\unified_weighted_200k_20261010_a

**Mã bộ sinh:** synth_weighted_v4.py  
**Corpus duy nhất:** train_weighted_unified_200000_seed2026101060.jsonl  
**Dữ liệu vai trò phụ trợ:** train_weighted_unified_200000_roles_seed2026101060.jsonl  
**Manifest:** train_weighted_unified_200000_manifest.json  
**QA độc lập:** independent_weighted_200000_quality_report.json  
**Kiểm tra holdout:** additional_holdout_sampling_audit_v2.json

**Trạng thái:** Đã sinh và kiểm tra dữ liệu; **chưa train**, chưa chỉnh sửa trọng số mô hình, chưa submit Kaggle.