# V9 — nghiên cứu tiếp về `via` false-positive, 10/10/2026

**Quyết định: chưa đủ điều kiện triển khai V9. Không có thay đổi nào vào source, artifacts, checkpoint V5 chính hay submission.** Tất cả script, report, checkpoint mới chỉ nằm trong workspace nghiên cứu.

## Bối cảnh

Nhánh D1 (học đồng thời via/urgent, frozen backbone/goal/fragile) cải thiện `urgent` trên V8 family nhưng làm tỷ lệ false-positive `via` tăng. Cần giữ phần hữu ích mà không phá độ đặc hiệu.

## Thí nghiệm 1 — Ghép theo đầu dự đoán (H)

Từ `experiments/v9_restricted_B0_control_v1.pt` và `experiments/v9_restricted_D1_focused_v1.pt`, xác nhận backbone, goal, fragile giống nhau; mô hình in-memory H giữ toàn bộ trọng số B0, chỉ thay các tensor `attention.urgent` và `output.urgent` bằng D1. Không huấn luyện hoặc tạo checkpoint H để triển khai.

- Trên family challenge 1.200 câu: exact-all B0 28,92% → H **31,33%**, tăng **2,42 điểm %**; urgent accuracy +4,17 điểm %. `via` FPR giữ **32,33%** so với B0, thay vì 34,50% nếu dùng cả D1.
- Trên v2/weighted/hard synthetic seed mới, exact-all H bằng B0 hoặc rất gần, không có cải thiện đáng kể.
- Trên focused holdout 192 câu, H thấp hơn B0 0,52 điểm %.
- Báo cáo: `checkpoints/v9_split_head_B0_D1_H_evaluation_v1.json`.

## Chẩn đoán sự tự tin của sai số via

Kiểm tra raw `via_mode` trên 1.200 câu family challenge (batch 64): có **233 ca false positive ở head mode**, **179/233 ca confidence >= 0,90**, confidence median ~0,997. Các câu sai thường khẳng định "đã nhận đủ tại kho", "bước lấy hoàn tất tại nguồn", "địa điểm bị hủy" mà vẫn dự đoán via cụ thể. Do vậy tăng confidence threshold đơn thuần khó giải quyết.

## Thí nghiệm 2 — Ưu tiên nhãn via=None khi huấn luyện head-only

- Khởi tạo cả hai nhánh từ mô hình H (B0 via/goal/fragile, D1 urgent).
- Giữ backbone và mọi head ngoài via cố định, assert bất biến trọng số.
- Cùng **6.400 câu train** gồm train thật 2.000, V9 cũ 1.000, synthetic weighted 2.800 và đối chứng sửa lỗi 600; **2 epoch, cùng seed/optimizer/hyperparams**, chỉ khác trọng số `via_mode` cho class NONE: E0 = 1,0, E25 = 2,5.
- Ba tập synthetic với seed mới khác các lần đánh giá trước và hai challenge (đã được dùng trong nghiên cứu), không mở official validation/test.

### So E25 với E0 (cùng data, cùng training budget)

| Tập | Δ exact-all | Δ via FPR | Δ via FNR |
|---|---:|---:|---:|
| V2 fresh n1500 | −0,27 pp | 0,00 pp | +1,25 pp |
| Weighted fresh n1500 | +0,07 pp | **−3,84 pp** | +4,14 pp |
| Hard fresh n1000 | −1,10 pp | **−2,20 pp** | +3,85 pp |
| V8 family n1200 | +0,92 pp | **−4,50 pp** | +2,33 pp |
| Focused192 | +2,08 pp | **−4,10 pp** | +1,43 pp |

**E25 làm ít dự đoán nhầm via hơn nhưng lại bỏ sót via nhiều hơn**; hard exact-all giảm 1,10 điểm %. `via` FPR family vẫn 28,33%, cao. Trên weighted mới E25 FPR 8,68%, FNR 27,10%. Chưa đủ tiêu chí bảo toàn recall và OOD. Không promote.

Báo cáo đầy đủ: `checkpoints/v9_via_none_weighted_AB_report_v1.json`.
Checkpoints nghiên cứu: `experiments/v9_via_E0_normal_control_v1.pt`, `experiments/v9_via_E25_none_weight_v1.pt`.

## Tồn đọng

1. Cần sửa lỗi ngữ nghĩa chứ không chỉ confidence calibration: negative `via` khi đã nhận hàng đủ/đã hủy chặng vẫn bị dự đoán rất chắc chắn.
2. Giữ giải pháp split-head H làm **giả thuyết nghiên cứu** vì tránh làm FPR nặng thêm, nhưng chưa có bằng chứng human-authored/blind.
3. Nên dùng loss/đối chứng làm rõ sự kiện hành động `pickup active` so với mentions đã hủy, kiểm duyệt nhãn thủ công, phối hợp recall/FPR; tránh liên tục tuning trên family challenge đã được xem.
4. Cần ít nhất nhiều seed và heldout ngôn ngữ viết độc lập, chưa tham gia chọn model, rồi mới đề xuất phê duyệt.
5. Không dùng nội dung test hoặc điểm public leaderboard để chọn ngưỡng/huấn luyện. V5 #1182 70,39% được giữ nguyên.

**Trạng thái V9: ĐÃ TIẾN THÊM, CHƯA HOÀN THIỆN, CHƯA PHÊ DUYỆT.**
