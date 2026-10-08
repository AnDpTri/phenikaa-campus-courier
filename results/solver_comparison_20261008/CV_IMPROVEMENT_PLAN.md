# Kế hoạch cải thiện CV sau bản solver candidate

Phân tích ngày 2026-10-08, chỉ dùng train/validation và mã nguồn.
Đây là kế hoạch thí nghiệm; chưa thay CV đang dùng để sinh submission v2.

## Bằng chứng

- Train có 493 print, 477 classic, 520 night, 510 sketch: không có lệch lớn về số mẫu.
- Long side trung vị của cả bốn style là 1120 px, cả train lẫn validation.
  Nhận định trước rằng print thường lớn hơn các style khác chưa có cơ sở.
- Print blur trung vị 0,7 trên cả train và validation; classic/night là 0.
- Print JPEG quality trung vị: train 65, validation 51 (chỉ tính ảnh JPEG).
- Khoảng cách giao lộ print sau resize 512: trung vị 52,25 px ở train,
  45,77 px ở validation. Grid dày hơn làm ký hiệu nhỏ tương đối và gần nhau hơn.
- Detector chỉ dùng gain/bias ngẫu nhiên; chưa có blur/JPEG/affine augmentation.
- Checkpoint được chọn theo focal loss tổng; chưa chọn theo node-set/grid hay
  chất lượng graph sau pipeline.
- Print validation: grid shape 69/74, node set 55/74, edge set 56/74,
  robot heading 68/74, scene exact 49/74. Lỗi grid shape chỉ là một phần:
  nhiều cảnh có đúng số hàng/cột nhưng vẫn sai tập node.

## Thí nghiệm ưu tiên

1. Giữ checkpoint hiện tại, thử inference 512/768 và grid fitting chắc hơn trên
   validation. Model convolution có thể xử lý kích thước khác, nhưng kết quả
   phải đo thực tế vì phân phối kích thước đối tượng đã thay đổi.
2. Train detector mới ở 768 với width 24 trước, để tách tác dụng độ phân giải
   khỏi tác dụng số tham số. Thêm blur/JPEG nhẹ đến mạnh, affine nhỏ; ảnh và
   nhãn tọa độ/heatmap/box phải biến đổi nhất quán.
3. Nếu còn thiếu khả năng biểu diễn, so width 24 với 32 ở cùng input 768.
   Chỉ thử width 48 nếu VRAM/time và kết quả validation cho phép.
4. Chọn checkpoint bằng node-set và grid-shape theo style, rồi xác nhận
   scene-exact và độ chính xác hành động end-to-end. Không dùng loss tổng
   làm chỉ số quyết định duy nhất.
5. Cải thiện grid fit: tận dụng tính thẳng hàng, khoảng cách lưới và confidence
   để loại peak lạc hàng/cột. Chỉ thêm node thiếu khi có bằng chứng ảnh.
   Không nối cạnh chỉ vì solver cần một tuyến đi tới đích.
6. Robot heading: huấn luyện với crop có sai lệch tâm đo được trên train,
   blur/JPEG và lượng xoay nhỏ; nếu flip/rotate thì phải đổi nhãn hướng đúng.
   Giữ EdgeNet hiện tại làm baseline; đánh giá lại trên crop detector thực.

## Tổ chức và tiêu chí

- Mỗi thí nghiệm có thư mục riêng trong results/cv_experiments; lưu model,
  cấu hình, seed, split, số liệu theo style, thời gian và VRAM.
- Không ghi vào artifacts CV baseline hay predictions của v1/v2.
- Không train CV đồng thời với quá trình sinh file test trên GPU 4 GB.
- Lưu một validation holdout hoặc hard split train trước khi thử nhiều cấu hình;
  validation hiện tại đã được dùng để tinh chỉnh nên điểm là benchmark phát triển.
- Mục tiêu thử nghiệm: print node-set từ 74,32% lên ít nhất 90%; hạn chế giảm
  ở các style còn lại; giảm fallback và cải thiện macro validation với solver
  cố định. Đây là mục tiêu, không phải kết quả đã đạt.
- Test chỉ dùng inference; không đọc nội dung test để bổ sung luật/augmentation.

## Lưu ý khả năng tái lập

Cache detector hiện chỉ kiểm tra tên heatmap. Khi thay độ phân giải, sigma,
augmentation hoặc dữ liệu, cần cache riêng và fingerprint cấu hình/dữ liệu.
Độ phân giải input cũng phải được lưu trong metadata artifact và dùng nhất quán
khi inference. Dùng AMP và batch nhỏ để thử vừa VRAM 4 GB; batch validation hiện
hard-code 32 cần cấu hình riêng.
