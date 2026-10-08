# Phenikaa Campus Courier · Vòng private test (bộ dữ liệu mới)

> **Đây là đề của vòng private test.** Bài toán, định dạng dữ liệu và cách nộp bài **giống hệt vòng trước** (mô tả đầy đủ ở các mục bên dưới). Bộ dữ liệu là **bộ mới hoàn toàn**: train, validation và test đều mới, sinh bằng khóa mới. Mô hình huấn luyện trên bộ cũ cần được **huấn luyện lại trên train mới**.

## 0. Những gì khác vòng trước

1. **Chiến thuật ngầm của 10 robot đã thay đổi và phức tạp hơn nhiều.**
   - Tên và gợi ý của robot giữ nguyên, nhưng trọng số, điều kiện và cách phá hòa đều mới.
   - Hành vi có thể phụ thuộc vào nhiều thông tin trên ảnh và trong câu hơn trước.
   - Mọi chiến thuật vẫn **cố định và tất định**, và đều suy ra được từ dữ liệu có nhãn của train mới.
2. **Quy tắc phá hòa không còn dùng chung.** Mỗi robot có quy tắc phá hòa riêng. Quy tắc chung số 4 ở mục 5 được thay bằng: *"Nếu nhiều bước đi tốt ngang nhau, mỗi robot phá hòa theo quy tắc riêng của nó; quy tắc này không được công bố."*
3. **Yêu cầu tiếng Việt khó hơn.**
   - Nơi giao có thể **chỉ được mô tả qua bản đồ**, không nêu tên loại địa điểm. Ví dụ: "địa điểm nằm cao nhất trên bản đồ", "nơi cạnh thư viện nhất". Khi đó phải đọc bản đồ mới biết nơi giao, và mọi robot tới đúng địa điểm được mô tả.
   - Câu có thể chứa **đính chính** ("tin trước ghi nhầm là …") hoặc **đối chiếu** ("không phải X mà là Y"). Nơi giao luôn là địa điểm đúng sau cùng.
   - Câu nhiễu, lỗi gõ và câu không dấu xuất hiện nhiều hơn.
   - Trong `scenes.json`, các mô tả qua bản đồ có `goal_ref.kind` thuộc `north_most`, `south_most`, `west_most`, `east_most`, `anchor_near`. Kèm theo là `goal_ref.rc`, tọa độ đúng của nơi giao.
4. **Ảnh khó hơn:** bản đồ có nhiều địa điểm hơn (8–14 địa điểm, nhiều loại có hai bản), nhiều ảnh bị xoay, mờ, nén JPEG hơn, kể cả ở train.
5. **Xếp hạng vòng loại chỉ dùng private test này: 10 đội đứng đầu vào chung kết.** Bảng của vòng công khai trước chỉ để tham khảo.
6. **Kích thước, định dạng và cách chấm không đổi.** Nộp `predictions.json` gồm đúng 12.000 số cho `test/observations.json` **của bộ mới**. Bài nộp này dùng mục **"Campus Courier · Private test"** trên website (mã `COURIER2`), tách riêng với bảng xếp hạng của vòng trước.

---

# (Đề gốc) Phenikaa Campus Courier: Robot sẽ đi hướng nào?

*Bài thi AI Hackathon kết hợp **thị giác máy tính (CV)**, **xử lý ngôn ngữ tự nhiên (NLP)** và **học chiến thuật từ dữ liệu**.*

## 1. Bối cảnh

Hệ thống giao nhận nội bộ của một khu campus giả lập có **10 robot**. Mỗi robot có một **chiến thuật di chuyển cố định**. Mỗi lượt, hệ thống nhận được:

- một **ảnh sơ đồ** khu campus: đường đi, trạng thái từng đoạn đường, các địa điểm, vị trí và hướng mặt của robot, thời tiết;
- một **yêu cầu bằng tiếng Việt**, ví dụ: *"Ghé căn tin phía bắc lấy khay cơm trước, rồi mang bưu kiện tới KTX. Hàng dễ vỡ, đi cẩn thận."*

Đội thi cần dự đoán **bước đi đầu tiên** của robot `robot_id`: đi **lên, xuống, trái hay phải**.

> Sơ đồ là bản đồ mô phỏng. Đây không phải bản đồ đi lại thật của Đại học Phenikaa.

## 2. Đầu vào và đầu ra

Mỗi dòng dữ liệu gồm:

| trường | ý nghĩa |
|---|---|
| `id` | mã dòng, ví dụ `test-00012-R7` |
| `robot_id` | số nguyên từ 0 đến 9 |
| `image` | đường dẫn ảnh (PNG hoặc JPEG), tính từ thư mục của split |
| `mission` | yêu cầu bằng tiếng Việt |

Đầu ra của mỗi dòng là **một số nguyên**: `0 = UP`, `1 = DOWN`, `2 = LEFT`, `3 = RIGHT`. Đây là hướng tuyệt đối trên ảnh, theo hàng và cột của lưới đường. `UP` nghĩa là đi sang giao lộ ở hàng ngay phía trên. Hướng này không phụ thuộc robot đang quay mặt về đâu.

Mỗi **cảnh** (một ảnh và một yêu cầu) có đúng 10 dòng liên tiếp, lần lượt cho robot 0 đến 9. Các dòng này dùng chung ảnh và yêu cầu; chỉ `robot_id` và nhãn thay đổi. Các cảnh độc lập với nhau: không có lịch sử, không có phần thưởng, không có tương tác.

## 3. Trong ảnh có gì?

- **Lưới giao lộ:**
  - Số hàng và số cột thay đổi theo từng ảnh: từ 5 đến 9; phần lớn ảnh train có từ 5 đến 8.
  - Một số giao lộ có thể không tồn tại (ví dụ chỗ đó là hồ nước).
  - Không phải cặp giao lộ kề nhau nào cũng có đường nối.
  - Giao lộ có thể lệch khỏi vị trí thẳng hàng một chút. Ảnh có thể bị xoay nhẹ, mờ hoặc nén JPEG như ảnh chụp hay ảnh quét.
- **Trạng thái đoạn đường:** đường thường, đường đông người, đường có mái che, đường đóng (không đi được).
- **Ký hiệu thêm:**
  - bậc thang: chỉ robot có chân đi qua được;
  - đường một chiều: đi theo chiều mũi tên;
  - robot **R**: có mũi chỉ hướng mặt đang quay;
  - biểu tượng thời tiết: nắng, nhiều mây hoặc mưa.
- **Địa điểm:** mỗi ảnh có từ 5 đến 9 địa điểm thuộc 10 loại dưới đây. Nhiều loại có thể xuất hiện **hai lần** trên cùng bản đồ.

| khóa trong `scenes.json` | ký hiệu | tên |
|---|---|---|
| `library` | TV | Thư viện |
| `dorm` | KTX | Ký túc xá |
| `sports` | TT | Nhà thể thao |
| `clinic` | YT | Trạm y tế |
| `canteen` | CA | Căn tin |
| `parking` | XE | Bãi xe |
| `lecture` | GĐ | Giảng đường |
| `lab` | TN | Phòng thí nghiệm |
| `office` | HC | Phòng hành chính |
| `gate` | CT | Cổng trường |

- **Bốn kiểu vẽ.** Mỗi kiểu có riêng:
  - bảng màu và cách biểu thị trạng thái đường (màu, nét đứt, nét đôi);
  - cách vẽ địa điểm (thẻ viết tắt, biểu tượng hình, nhãn tên đầy đủ, huy hiệu);
  - phông chữ và vị trí chú giải.

  Kích thước ảnh cũng thay đổi.
- **Chú giải là quy ước của chính ảnh đó.** Trong một số ảnh, màu hoặc kiểu nét của đường thường, đường đông và đường có mái che bị **hoán đổi** so với thói quen của kiểu vẽ. Ví dụ, "Đường đông người" có thể được vẽ bằng màu xanh. Chỉ chú giải của ảnh cho biết đúng quy ước.
- Ảnh PNG được lưu dạng bảng màu (palette). Hãy chuyển mọi ảnh sang RGB khi đọc, ví dụ `Image.open(p).convert("RGB")`.

## 4. Trong yêu cầu có gì?

- **Nơi cần giao (goal).** Yêu cầu có thể gọi thẳng tên, dùng cách gọi khác ("nơi mượn giáo trình", "khu nội trú") hoặc chỉ nhắc tới người nhận ("thủ thư đang chờ…").
- **Điểm ghé giữa đường (via), không bắt buộc.** Robot phải tới đó trước rồi mới tới nơi giao. Điểm ghé có thể được nhắc trước hoặc sau nơi giao trong câu, ví dụ "Trước khi mang … tới A, nhớ ghé B".
- **Tham chiếu không gian.** Khi một loại địa điểm có hai bản trên bản đồ, yêu cầu có thể chỉ rõ bản nào:
  - **phía bắc / nam / tây / đông** (hay phía trên / dưới / trái / phải bản đồ): bản nằm xa hơn về phía đó. Bắc là phía trên ảnh, theo mũi tên "B".
  - **gần X hơn / xa X hơn**, với X là một địa điểm chỉ có một bản trên bản đồ: bản có khoảng cách đường chim bay tới X nhỏ hơn / lớn hơn.

  Chỉ những tham chiếu rõ ràng mới được dùng: hai bản cách nhau ít nhất 2 hàng hoặc 2 cột, hoặc chênh lệch khoảng cách tới X ít nhất 1 ô.
- **Mức độ gấp** và **hàng dễ vỡ.** Hai thông tin này có thể bị phủ định, ví dụ "không gấp đâu" hay "hàng chắc chắn, không dễ vỡ".
- **Địa điểm gây nhiễu:** tên những nơi không phải đích, ví dụ "không cần ghé…", "đừng nhầm với…".
- Một số yêu cầu **viết không dấu**, viết thường hoặc **gõ sai chính tả**.
- `validation` và `test` có **cách diễn đạt không xuất hiện trong `train`**: tên gọi địa điểm, khung câu và cụm từ mới.

## 5. Mười robot

Thông tin công khai của các robot nằm trong `robots.json`.

| robot_id | tên | gợi ý |
|---|---|---|
| 0 | Tia Chớp | Chỉ muốn tới nơi bằng ít đoạn đường nhất. |
| 1 | Yên Tĩnh | Rất ngại những đoạn đường đông người. |
| 2 | Mái Hiên | Thích đi dưới mái che, dù phải vòng một chút. |
| 3 | Mây Mưa | Cách chọn đường phụ thuộc vào thời tiết trên bản đồ. |
| 4 | Sơn Dương | Robot có chân, là robot duy nhất đi được bậc thang. |
| 5 | Thẳng Tắp | Không thích đổi hướng; hướng mũi robot trên ảnh rất quan trọng. |
| 6 | Hỏa Tốc | Đọc kỹ mức độ khẩn cấp trong yêu cầu. |
| 7 | Nâng Niu | Cách đi phụ thuộc vào món hàng đang chở. |
| 8 | Lề Phải | Có thói quen riêng khi đổi hướng ở giao lộ. |
| 9 | Tham Lam | Chỉ nhìn một bước trước mắt. |

**Quy tắc chung (công khai):**

1. Không robot nào đi vào đường đóng hoặc đi ngược đường một chiều. Chỉ Sơn Dương (robot 4) đi được bậc thang.
2. Nếu yêu cầu có điểm ghé, robot tới điểm ghé trước rồi mới tới nơi giao.
3. Nếu yêu cầu chỉ rõ một địa điểm bằng tham chiếu không gian, **mọi robot tới đúng địa điểm đó**. Nếu không chỉ rõ và loại địa điểm có hai bản, robot chọn bản hợp với chiến thuật của nó.
4. Mỗi robot có chiến thuật **cố định và tất định**: cùng một cảnh luôn cho cùng một bước đi. **Vòng private test:** nếu nhiều bước đi tốt ngang nhau, mỗi robot phá hòa theo **quy tắc riêng** của nó (xem mục 0).
5. Trọng số, điều kiện và quy tắc phá hòa **không được công bố**. Đội thi cần tự suy ra chúng từ dữ liệu có nhãn.

## 6. Dữ liệu

```
delivery_public/
  robots.json
  train/        observations.json  labels.json  scenes.json  dataset_card.json  images/
  validation/   observations.json  labels.json  scenes.json  dataset_card.json  images/
  test/         observations.json  sample_submission.json    dataset_card.json  images/
starter/
  starter.py    đọc dữ liệu, chấm validation giống cách chấm test, baseline đơn giản,
                vẽ lại chú thích lên ảnh (python starter.py --show 0)
```

- **Validation và test khó hơn train:**
  - bản đồ lớn hơn và dày đặc hơn;
  - nhiều điểm ghé và tham chiếu không gian hơn;
  - ảnh nhiễu hơn, nhiều ảnh bị hoán đổi chú giải hơn;
  - yêu cầu dài và nhiễu hơn;
  - các robot bất đồng với nhau thường xuyên hơn.

  Mọi hiện tượng trong test đều đã có trong train, chỉ với tần suất thấp hơn. Hãy dùng validation để đo khả năng tổng quát hóa.
- `labels.json` là danh sách số nguyên, cùng thứ tự với `observations.json`.
- `scenes.json` (chỉ có ở train và validation) là **chú thích phụ** cho từng ảnh. Có thể dùng nó để huấn luyện riêng phần đọc ảnh và phần đọc yêu cầu. Mỗi phần tử ứng với một cảnh, theo đúng thứ tự cảnh trong `observations.json`, và gồm:
  - `scene_id`, `image`, `width`, `height`: mã cảnh, đường dẫn ảnh và kích thước ảnh;
  - `grid`: số hàng và số cột;
  - `nodes`: mỗi giao lộ gồm `rc = [hàng, cột]` và `xy` là tọa độ pixel tâm giao lộ trên ảnh;
  - `edges`: đoạn đường giữa hai giao lộ kề nhau `a` và `b`, gồm `status` (`normal` / `crowded` / `covered` / `closed`), `stairs` (true/false) và `oneway_to` (giao lộ được phép đi tới, hoặc `null`);
  - `landmarks`: địa điểm, gồm `type` và `rc`;
  - `robot`: vị trí `rc` và `heading`;
  - `weather`: `rain` hoặc `dry`;
  - `mission`: `text`, `goal`, `goal_ref`, `via` (hoặc `null`), `via_ref`, `urgent`, `fragile`. Trường `goal_ref` / `via_ref` là `null` hoặc `{kind, anchor, rc}`, với `kind` ∈ north/south/west/east/near/far, `anchor` là loại địa điểm mốc và `rc` là địa điểm được chỉ tới;
  - `legend`: từng dòng chú giải, gồm `kind`, `text`, và khung pixel `swatch` (mẫu ký hiệu) và `label` (chữ);
  - `weather_box`: khung pixel của biểu tượng thời tiết;
  - `road_look`: trạng thái nào được vẽ bằng kiểu thông thường của trạng thái nào. Ví dụ `{"crowded": "covered"}` nghĩa là đường đông được vẽ giống đường có mái che thông thường;
  - `degradation`: độ xoay, độ mờ, chất lượng JPEG;
  - `style`: kiểu vẽ.
- **Test không có `scenes.json`.** Khi dự đoán, hệ thống chỉ được dùng ảnh, yêu cầu và `robot_id`.
- Mạng đường của mỗi bản đồ test **không trùng với bất kỳ bản đồ nào trong train hoặc validation**. Vị trí địa điểm, vị trí robot, trạng thái đường và cách vẽ cũng được sinh mới cho từng ảnh.

Kích thước bộ dữ liệu: train 2.000 cảnh (20.000 dòng), validation 300 cảnh (3.000 dòng), test 1.200 cảnh (12.000 dòng).

## 7. Nộp bài và chấm điểm

- Nộp `predictions.json` trên website cuộc thi (https://aihackathon.phenikaa-uni.edu.vn): danh sách số nguyên từ 0 đến 3, **cùng độ dài và cùng thứ tự** với `test/observations.json`. `sample_submission.json` là ví dụ đúng định dạng. Phải dự đoán cho **mọi dòng**. File sai định dạng bị từ chối ngay và không tính lượt nộp.
- **Điểm = trung bình accuracy của 10 robot** (macro accuracy). Mỗi robot có trọng số như nhau.
- **Không phải cảnh test nào cũng được chấm.** Test gồm cảnh được chấm trộn lẫn với cảnh không chấm, và không có cách phân biệt hai loại.
  - Bảng xếp hạng công khai trong thời gian thi dùng khoảng 30% số cảnh được chấm.
  - **Xếp hạng cuối cùng dùng các cảnh được chấm còn lại** và chỉ được công bố sau khi kết thúc. Với mỗi đội, hệ thống tự động dùng **bài nộp có điểm công khai cao nhất** (nếu bằng điểm, lấy bài nộp sau cùng).
- Mỗi đội được nộp tối đa **5 lần mỗi ngày** (theo giờ Việt Nam).
- **Đây là vòng loại (giai đoạn 1) của PHENIKAA AI Hackathon 2026.** **10 đội có xếp hạng cuối cùng cao nhất ở private test** vào vòng chung kết (giai đoạn 2), gồm hai bài toán mới được công bố khi vòng chung kết bắt đầu. Trước khi công bố danh sách, BTC có thể yêu cầu các đội này nộp mã nguồn để kiểm tra (mục 8.5).

## 8. Quy định

1. Được dùng mọi phương pháp: xử lý ảnh truyền thống, CNN, mô hình ngôn ngữ nhỏ, luật viết tay hoặc kết hợp.
2. Dự đoán phải do hệ thống tự động của đội sinh ra, **không gọi dịch vụ AI bên ngoài** (API của mô hình ngôn ngữ hoặc mô hình đa phương thức).
3. Tổng số tham số của các mô hình dùng khi dự đoán **không vượt quá 200 triệu**. Được dùng mô hình pretrained công khai trong giới hạn này, ví dụ PhoBERT, ResNet hoặc MobileNet.
4. **Không dùng nội dung test để huấn luyện hay viết luật.** Cụ thể:
   - không gán nhãn tay;
   - không đọc câu hay xem ảnh test để bổ sung từ điển hoặc luật;
   - không huấn luyện, kể cả tự học không nhãn, trên dữ liệu test.

   Test chỉ dùng để chạy dự đoán.
5. BTC có quyền yêu cầu các đội có thứ hạng cao nộp mã nguồn và hướng dẫn chạy để kiểm tra lại kết quả. Đội không tái lập được kết quả hoặc vi phạm mục 4 sẽ bị loại.

## 9. Gợi ý hướng tiếp cận

Có thể tách bài thành 3 khâu và kiểm tra từng khâu trên validation bằng `scenes.json`:

```
ảnh ──CV──► đồ thị (giao lộ, đoạn đường, ký hiệu, địa điểm, robot, thời tiết) ─┐
      (đọc chú giải!)                                                          ├─► chiến thuật của robot_id ─► hướng đi
yêu cầu ──NLP──► (goal, via, tham chiếu, urgent, fragile) ─────────────────────┘
```

- **CV:**
  - dùng `nodes[].xy` để huấn luyện một mạng nhỏ tìm tâm giao lộ (heatmap), rồi gán hàng và cột;
  - cắt một mảnh ảnh dọc mỗi cặp giao lộ kề nhau và cho một CNN nhỏ phân loại trạng thái đường, bậc thang và chiều mũi tên;
  - cắt một mảnh ảnh tại mỗi giao lộ để nhận ra địa điểm và robot (kèm hướng mũi robot);
  - dùng `legend` để học cách đọc chú giải, rồi so khớp màu hoặc kiểu nét của từng đoạn đường với chú giải của chính ảnh đó.
- **NLP:** huấn luyện bộ phân loại hoặc gán nhãn chuỗi nhỏ, ví dụ TF-IDF + hồi quy logistic, BiLSTM hoặc PhoBERT. Cần chú ý: phủ định, thứ tự "ghé… trước, rồi…", tham chiếu không gian, địa điểm gây nhiễu, câu không dấu, lỗi gõ và cách diễn đạt mới.
- **Chiến thuật:**
  - mô phỏng đường đi trên đồ thị với nhiều giả thuyết chi phí (quãng đường, đường đông, mái che, số lần rẽ, …) và học xem giả thuyết nào khớp với từng robot, trong điều kiện nào;
  - hoặc huấn luyện một mạng nơ-ron đồ thị (GNN) nhỏ.
  Nên so sánh các robot trên cùng một cảnh; đó là manh mối rất tốt.

Không cần học tăng cường và không cần tương tác online. Trọng tâm của bài là đọc đúng ảnh mới, hiểu đúng yêu cầu mới và suy ra đúng chiến thuật của từng robot.
