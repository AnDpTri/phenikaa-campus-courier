# Báo cáo bàn giao và kế hoạch nâng cấp NLP

**Dự án:** Phenikaa Campus Courier 2026

**Ngày lập:** 08/10/2026

**Phạm vi:** Phân tích câu lệnh giao hàng tiếng Việt, không bao gồm CV và chiến thuật solver

> **Yêu cầu bắt buộc đối với bên tiếp quản:** mọi số liệu, kết luận và giả định trong tài liệu này phải được kiểm chứng lại trên môi trường của bên thực hiện trước khi dùng để thay mô-đun hiện tại hoặc tạo bài nộp. Không coi validation 100% là bằng chứng mô-đun sẽ đạt 100% trên test.

## 1. Mục tiêu bàn giao

Xây dựng mô-đun NLP có khả năng tổng quát hóa tốt hơn parser dựa trên luật hiện tại, đặc biệt với:

- cách diễn đạt chưa xuất hiện trong train;
- câu không dấu, thiếu dấu câu hoặc có lỗi gõ;
- nhiều địa điểm được nhắc trong cùng yêu cầu;
- câu phủ định, đính chính, “tin cũ/tin mới”;
- mô tả không nêu trực tiếp loại đích mà dùng quan hệ không gian.

Đầu ra NLP phải tương thích với pipeline hiện có và biểu diễn được:

```text
goal_type, goal_reference, goal_anchor,
via_type, via_reference, via_anchor,
urgent, fragile
```

Mô-đun cuối cùng phải chạy hoàn toàn local khi dự đoán test và toàn bộ hệ thống phải nằm dưới giới hạn 200 triệu tham số.

## 2. Trạng thái hiện tại

NLP hiện tại là parser dựa trên từ điển, biểu thức/quy tắc và một bộ sửa lỗi chính tả theo unigram/bigram. Đây không phải mô hình ngôn ngữ đã được fine-tune.

Các thành phần chính:

- `src/courier/nlp/parser.py`: phân tích câu thành `TargetSpec` cho goal và via;
- `src/courier/nlp/text.py`: chuẩn hóa chữ và sửa một số lỗi gõ;
- `src/courier/nlp/lexicon.py`: từ/cụm từ và alias tổng quát;
- `src/courier/nlp/resolver.py`: ánh xạ `TargetSpec` vào landmark thực tế trên bản đồ;
- `src/courier/nlp/vocab.json`: thống kê unigram/bigram dựng từ train;
- `scripts/nlp/evaluate_nlp.py`: đánh giá NLP bằng landmark chuẩn;
- `tests/nlp/test_parser.py`: unit test hiện có.

Resolver hiện có thể thay goal khi parser không tìm thấy đích hoặc loại đích không xuất hiện trong landmark. Via không giải được có thể bị bỏ. Vì vậy pipeline chạy thành công không đồng nghĩa NLP đã hiểu đúng yêu cầu.

## 3. Số liệu đã quan sát

Đánh giá dùng landmark chuẩn từ `scenes.json`, nhờ đó tách lỗi NLP khỏi CV.

| Chỉ số | Train, 2.000 cảnh | Validation, 300 cảnh |
| --- | ---: | ---: |
| Node đích | 100% | 100% |
| Loại đích | 100% | 100% |
| Loại tham chiếu đích | 100% | 100% |
| Node ghé qua | 99,95% | 100% |
| Urgent | 100% | 100% |
| Fragile | 100% | 100% |
| Đúng toàn bộ | 99,95% | 100% |

18/18 unit test NLP đã đạt trong lần kiểm tra. Vocabulary hiện tại và vocabulary dựng lại từ train có cùng nội dung: 241 unigram và 1.333 bigram.

Lỗi duy nhất quan sát được trên train là cảnh `train-01552`. Cụm “ghé khu thể tao lấy bộ vợt trước” có nhãn via là `sports`, nhưng parser không sửa từ hợp lệ “tao” thành “thao”, dẫn đến mất via. Không nên khắc phục bằng luật riêng `thể tao -> thể thao`; cần xử lý sửa lỗi theo cụm/ngữ cảnh và đo nguy cơ sửa nhầm.

Stress test có kiểm soát, giữ nguyên nhãn:

| Biến thể | Train đúng toàn bộ | Validation đúng toàn bộ |
| --- | ---: | ---: |
| Văn bản gốc | 99,95% | 100% |
| Bỏ dấu tiếng Việt bằng phép chuẩn hóa hiện tại | 99,95% | 100% |
| Bỏ `. ! ? ; ( ) [ ]` | 92,45% | 91,00% |
| Thêm lỗi xóa/đảo ký tự với xác suất 5% cho mỗi từ đủ dài | 99,65% | 100% |

Stress test là phép thử nhân tạo, chỉ dùng một cấu hình/seed và **không phải ước lượng accuracy trên test**. Việc bỏ dấu câu có thể làm mất tín hiệu phạm vi thật, nên kết quả 91% chỉ chứng minh parser nhạy với cách phân đoạn câu, không chứng minh test sẽ giảm đúng chừng đó.

Audit suy luận test trước đây ghi nhận:

- 96/1.200 cảnh không có parsed goal;
- 245/1.200 cảnh dùng goal thay thế;
- trong 245 cảnh đó có 148 cảnh mà loại địa điểm được parser yêu cầu không xuất hiện trong landmark do CV cung cấp;
- 9 via bị bỏ.

Test không có nhãn chuẩn, nên các con số trên **không phải tỷ lệ sai của NLP**. Nguyên nhân có thể thuộc NLP, CV, hoặc tương tác giữa hai mô-đun. Không được đọc nội dung từng câu test để bổ sung luật.

Chi tiết lần đo hiện tại nằm trong thư mục local `results/nlp_check_20261008/`, gồm `REPORT.md`, `metrics.json`, danh sách lỗi và script stress test. Thư mục `results` đang bị Git ignore; bên tiếp quản cần tự chạy lại thay vì chỉ dựa vào artifact local này.

## 4. Ràng buộc cuộc thi

Theo mục 8 của `Phenikaa_Campus_Courier_2026_v3/DE_BAI.md`:

1. Được dùng mô hình ngôn ngữ nhỏ, luật viết tay hoặc phương pháp kết hợp.
2. Khi dự đoán không được gọi dịch vụ AI bên ngoài, bao gồm API mô hình ngôn ngữ hoặc đa phương thức.
3. Tổng số tham số của các mô hình dùng khi dự đoán không vượt quá 200 triệu.
4. Không dùng nội dung test để gán nhãn, huấn luyện, tự học không nhãn, bổ sung từ điển hoặc viết luật.
5. Test chỉ được dùng để chạy dự đoán.

Việc dùng API AI ở giai đoạn phát triển để sinh dữ liệu tổng hợp từ train/schema không bị câu chữ trên cấm trực tiếp, nhưng cách hiểu này **cần được xác nhận lại với ban tổ chức trước khi sử dụng chính thức**. Nếu chưa có xác nhận, chỉ dùng tăng cường dữ liệu local/tất định và mô hình pretrained công khai theo quy định.

Không gửi câu hoặc ảnh test vào bất kỳ API nào. Không đưa API key vào mã nguồn, log, artifact hoặc Git.

## 5. Kiến trúc đề xuất

Sử dụng mô hình lai, trong đó một encoder tiếng Việt local làm parser chính và luật hiện tại làm lớp kiểm tra/fallback có giám sát.

### 5.1. Mô hình học máy

Dùng một encoder không làm tổng tham số toàn pipeline vượt 200 triệu. Một ứng viên có thể là PhoBERT-base hoặc một encoder nhỏ hơn, nhưng phải đo số tham số thực tế cùng toàn bộ mô hình CV trước khi chốt.

Trên biểu diễn câu, gắn các đầu phân loại:

- `goal_type`: 10 loại landmark và trạng thái không xác định;
- `goal_ref`: none/north/south/west/east/near/far/north_most/south_most/west_most/east_most/anchor_near;
- `goal_anchor`: 10 loại landmark và none;
- ba đầu tương ứng cho via;
- hai đầu nhị phân urgent và fragile.

Danh sách loại landmark và quan hệ không gian là ontology cố định của đề, không phải luật được suy ra từ test. Cần xác minh lại mapping giữa output của model, `TargetSpec` và `SpatialRef` hiện tại.

### 5.2. Kết hợp với parser luật

Đề xuất kết hợp theo từng trường thay vì thay toàn bộ kết quả một lần:

- model và parser đồng ý: chấp nhận;
- model vượt ngưỡng confidence đã khóa trên dữ liệu phát triển: ưu tiên model;
- model không chắc nhưng parser có kết quả hợp lệ: dùng parser và ghi nguồn fallback;
- hai bên mâu thuẫn hoặc đều không chắc: đánh dấu unresolved và ghi nguyên nhân.

Không được dùng resolver fallback như bằng chứng NLP đúng. Báo cáo phải tách `parsed`, `resolved`, `fallback reason` và kết quả cuối.

## 6. Xây dựng dữ liệu

### 6.1. Dữ liệu thật

Chuyển toàn bộ `mission` trong train và validation thành bảng nhãn cấu trúc. Trước khi huấn luyện phải kiểm tra:

- mọi giá trị thuộc ontology;
- goal luôn tồn tại;
- via và via_ref nhất quán;
- anchor chỉ xuất hiện với quan hệ phù hợp;
- phân phối từng lớp và tổ hợp hiếm;
- không rò cùng khuôn câu giữa train và holdout.

### 6.2. Tăng cường local

Tạo biến thể có seed và lưu provenance:

- bỏ dấu tiếng Việt;
- thay đổi dấu câu;
- lỗi xóa, chèn, đảo và thay ký tự;
- thay đổi khoảng trắng/viết hoa;
- đổi thứ tự các mệnh đề nhưng giữ nghĩa;
- chèn câu nhiễu không làm thay đổi goal/via;
- tạo phủ định và đính chính theo template tổng quát.

Mọi phép biến đổi phải có unit test chứng minh nhãn vẫn đúng. Không dùng phép biến đổi nếu việc bỏ dấu câu hoặc đổi thứ tự làm câu trở nên nhập nhằng.

### 6.3. Tăng cường bằng API, nếu được xác nhận hợp lệ

API chỉ nhận schema, nhãn cấu trúc và câu train hoặc yêu cầu sinh câu mới từ cấu trúc. Yêu cầu output JSON theo schema cố định, ví dụ gồm `text`, `goal`, `goal_ref`, `goal_anchor`, `via`, `via_ref`, `via_anchor`, `urgent`, `fragile`.

Cần:

- lưu model API, phiên bản prompt, tham số sinh, thời gian và hash đầu vào;
- loại trùng theo văn bản chuẩn hóa;
- kiểm tra JSON và ontology tự động;
- kiểm tra ngẫu nhiên thủ công trên dữ liệu sinh, không phải test;
- cân bằng tổ hợp hiếm thay vì sinh đồng đều;
- loại mẫu mà câu và nhãn không thể xác minh là nhất quán;
- không phụ thuộc API khi chạy submission.

## 7. Chiến lược đánh giá

Không chọn mô hình chỉ dựa trên random split hoặc accuracy train.

### 7.1. Các tập đánh giá cần có

1. **Official validation:** báo cáo khả năng trên phân phối cuộc thi; không dùng lặp đi lặp lại để tối ưu từng thay đổi nhỏ.
2. **Template/group holdout:** nhóm theo khuôn/cụm diễn đạt đã chuẩn hóa, để khuôn gần nhau không nằm ở cả hai phía.
3. **Challenge set:** không dấu, thiếu dấu câu, typo, nhiều thực thể, phủ định, đính chính.
4. **Regression set:** toàn bộ lỗi đã tìm thấy trên train/validation cùng các phản ví dụ dễ sửa nhầm.

### 7.2. Chỉ số bắt buộc

- exact match cho từng trường;
- exact match toàn bộ mission;
- `goal_nodes` và `via_nodes` sau resolver dùng landmark chuẩn;
- tỷ lệ unresolved;
- tỷ lệ và nguyên nhân fallback;
- calibration/confidence theo từng đầu;
- accuracy hành động của solver khi chỉ thay NLP;
- thời gian, RAM/VRAM và số tham số khi inference.

Kết quả end-to-end phải được tách tối thiểu thành:

- oracle CV + NLP mới + solver;
- CV hiện tại + NLP mới + solver;
- parser cũ và NLP mới trên cùng một artifact CV/solver.

### 7.3. Điều kiện chấp nhận sơ bộ

Các ngưỡng dưới đây là đề xuất và cần được đội tiếp quản xác nhận trước khi huấn luyện:

- không giảm official validation so với parser hiện tại;
- tăng rõ ràng trên template/group holdout;
- challenge bỏ dấu câu cao hơn đáng kể mức 91% hiện quan sát;
- xử lý đúng lỗi `train-01552` và có phản ví dụ chứng minh không sửa nhầm mọi từ “tao”;
- không tăng số goal/via unresolved hoặc fallback ngoài ngưỡng đã thống nhất;
- end-to-end validation không giảm;
- pipeline dự đoán local, tái lập được và dưới 200 triệu tham số.

Không thay parser hiện tại chỉ vì train đạt 100%.

## 8. Trình tự triển khai đề xuất

### Giai đoạn A — tái lập baseline

1. Cài môi trường sạch từ repository.
2. Chạy lại 18 unit test NLP.
3. Chạy `evaluate_nlp.py` trên train và validation.
4. Dựng lại vocabulary từ train và so sánh nội dung.
5. Chạy lại stress test bằng seed đã ghi.
6. Ghi phiên bản Python, thư viện, commit Git và hash dữ liệu.

Không tiếp tục nếu không tái lập được baseline hoặc phải giải thích rõ chênh lệch.

### Giai đoạn B — bộ dữ liệu và evaluator

1. Viết exporter nhãn cấu trúc.
2. Viết group split và challenge generator.
3. Thêm metric unresolved/fallback/calibration.
4. Thêm regression test cho `train-01552` và phản ví dụ.
5. Khóa tập đánh giá trước khi chọn model.

### Giai đoạn C — mô hình local

1. Huấn luyện baseline encoder với các multi-head.
2. So sánh frozen encoder, unfreeze một phần và fine-tune toàn bộ.
3. Dùng early stopping trên development holdout, không trên test.
4. Chạy nhiều seed và báo trung bình/độ lệch, tránh chọn may mắn.
5. Calibrate confidence trên tập riêng.

### Giai đoạn D — dữ liệu tổng hợp

1. Thử augmentation local trước.
2. Nếu được BTC xác nhận, sinh dữ liệu API chỉ từ train/schema.
3. Kiểm tra chất lượng và provenance.
4. Chạy ablation: không augmentation, local augmentation, API augmentation.
5. Chỉ giữ nguồn dữ liệu chứng minh được cải thiện trên holdout.

### Giai đoạn E — tích hợp và đóng băng

1. Tích hợp model nhưng giữ parser cũ để đối chứng.
2. Đo riêng NLP và end-to-end.
3. Đo tổng tham số, thời gian và tài nguyên.
4. Khóa model, tokenizer, threshold và mã nguồn.
5. Huấn luyện artifact cuối trên dữ liệu đã được quy định trước.
6. Chạy test ở chế độ inference-only, không gọi API.
7. Lưu submission mới ở thư mục riêng; không ghi đè kết quả cũ.

## 9. Deliverable yêu cầu từ bên tiếp quản

- mã nguồn tạo dataset và split;
- prompt/API schema và log provenance nếu dùng API;
- model local và tokenizer;
- script huấn luyện có seed;
- script đánh giá riêng NLP và end-to-end;
- báo cáo số tham số và tài nguyên;
- bảng ablation;
- regression test;
- tài liệu cách tái lập;
- file submission mới trong thư mục version riêng;
- báo cáo xác nhận không dùng test để huấn luyện/viết luật.

Không commit API key, dữ liệu test đã trích xuất, model nhị phân lớn hoặc submission nếu chính sách repository hiện tại vẫn ignore các thành phần này.

## 10. Các điểm bắt buộc phải kiểm chứng lại

Trước khi chấp nhận kết quả, bên tiếp quản phải xác minh tối thiểu:

1. Số liệu 99,95% train, 100% validation và 18/18 unit test có tái lập được không.
2. Stress test 91% có đúng cách biến đổi, đúng seed và không vô tình đổi nghĩa không.
3. Vocabulary có thật sự chỉ dựng từ train không.
4. Group holdout có rò rỉ template/paraphrase không.
5. Nhãn dữ liệu tổng hợp có đúng ngữ nghĩa không.
6. Threshold confidence có được chọn mà không nhìn test không.
7. API augmentation có được ban tổ chức cho phép hay không.
8. Test có hoàn toàn vắng mặt khỏi training, tuning và rule development không.
9. Tổng số tham số của toàn bộ CV + NLP + các mô hình liên quan có dưới 200 triệu không.
10. Bản submission có thể tái lập từ commit, artifact và lệnh chạy được ghi lại không.
11. Cải thiện NLP có còn tồn tại ở end-to-end, hay bị CV/solver che mất.
12. Mọi fallback có được thống kê thay vì tính như parse thành công không.

## 11. Kết luận bàn giao

Parser hiện tại rất tốt trên train/validation gốc nhưng còn dấu hiệu phụ thuộc vào khuôn câu và dấu câu. Chưa có bằng chứng hợp lệ để khẳng định NLP đạt 100% trên test. Hướng nâng cấp phù hợp là học ánh xạ câu sang schema cố định bằng mô hình local, dùng dữ liệu tổng hợp có kiểm soát, giữ luật làm lớp kiểm tra và đánh giá trên group holdout/challenge set.

Toàn bộ hướng trên là kế hoạch kỹ thuật cần thực nghiệm. Bên tiếp quản không nên coi bất kỳ kiến trúc, ngưỡng confidence, mô hình pretrained hay cách dùng API nào là đã được xác nhận trước khi chạy lại đầy đủ các bước kiểm chứng nêu trong tài liệu.
