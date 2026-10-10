# Semantic V9 — Kiểm định A/B ngày 10/10/2026

**Kết luận: V9 cải thiện rõ trên synthetic weighted/hard trong pilot, chưa chứng minh đã sửa triệt để V5.**

## An toàn và phạm vi
- Tất cả file mới trong D:\phenikaa\results\NLP_Data_Research_20261010_a.
- Không thay thế/sửa source, dữ liệu và checkpoint gốc.
- Không mở official validation/test; chỉ dùng real train 2.000 câu và synthetic probes.
- Không dùng family challenge để train.

## Sửa dữ liệu
- 3.184 câu V8 được bổ sung căn cứ thuộc tính hàng hóa cho nhãn fragile; thay vì suy ra từ trạng thái quy định xử lý hàng.
- V9 pilot 3.200 câu: via 32,72%; urgent 38,22%; fragile 39,19%; có dấu 70,63%; trung vị 49 token. Real train tương ứng 32,70%, 38,20%, 39,20%, 70,60% và 46 token.
- Dữ liệu vẫn synthetic và chưa có human audit độc lập. Tỷ lệ >=70 token V9 là 10,47%, cao hơn real train 1,75%.
- Code: prepare_semantic_v9_pilot_fixed.py
- Dữ liệu: experiments\semantic_v9_label_repaired_full.jsonl; experiments\semantic_v9_real_like_pilot.jsonl
- Manifest: checkpoints\semantic_v9_repair_and_sampling_manifest.json

## Thiết kế A/B
- Chạy từ cùng Scratch seed 1 epoch 5 (frozen input checkpoint).
- A: 10.000 câu weighted synthetic mới + 2.000 real train.
- B: giữ toàn bộ cấu hình, thay 1.000/10.000 câu synthetic bằng V9 + cùng 2.000 real train.
- Mỗi arm 3 epoch, 94 steps/epoch, tổng 282 optimizer steps. Cùng seed, AdamW, OneCycleLR peak 0,00015, micro64 tích lũy2.
- Dùng đúng cùng một tập synthetic V2 (n1000), weighted (n1000), hard (n600), V8 family challenge (n1200) để đối chiếu.
- Đây là pilot tiếp tục học từ một model đã train, KHÔNG phải retrain toàn bộ 3 seed từ đầu.

## Kết quả exact-all
| Tập | Ban đầu | A dữ liệu cũ | B có V9 | B - A |
|---|---:|---:|---:|---:|
| V2 n1000 | 48,50% | 49,60% | 50,20% | +0,60 điểm % |
| Weighted n1000 | 36,90% | 37,30% | 48,60% | +11,30 điểm % |
| Hard n600 | 33,33% | 31,67% | 52,67% | +21,00 điểm % |
| Family n1200 | 17,83% | 16,75% | 23,92% | +7,17 điểm % |

## Các lỗi còn lại
- Hard fragile accuracy A 55,17% -> B 84,50% (+29,33 điểm %).
- Hard via bỏ sót A 42,57% -> B 26,73% (giảm 15,84 điểm %).
- Nhưng hard via false positive A 8,42% -> B 11,79% (tăng 3,37 điểm %).
- Weighted via false positive tăng từ 8,42% lên 11,16%.
- Family challenge urgent accuracy vẫn 50% ở cả A và B.
- Gain V2 rất nhỏ; V2 goal exact có thể giảm khi thêm V9.
- Pilot chỉ có một seed, ngắn, toàn bộ các probe là synthetic có thể chia sẻ từ vựng/họ cấu trúc. Không khẳng định thống kê chắc chắn ngoài các tập này, cũng không khẳng định mô hình đã generalize trên văn bản người viết.

## Kết quả và mô hình
- Báo cáo chi tiết: checkpoints\v9_controlled_ab_pilot_results.json
- Log: checkpoints\v9_controlled_ab_pilot_metrics.jsonl
- Config/selection: experiments\v9_controlled_ab_train_sampling_manifest.json
- Checkpoint A: experiments\v9_pilot_A_continuation.pt
- Checkpoint B: experiments\v9_pilot_B_continuation.pt
- Script: train_controlled_v9_ab_pilot.py

## Quyết định
Có đủ bằng chứng để tiếp tục kiểm chứng V9. Chưa đủ cơ sở thay mô hình chính. Bước tiếp: kiểm duyệt nhãn thủ công 100–200 câu; kiểm tra via false-positive và urgent; lặp 3 seeds; kiểm định bằng các câu độc lập do người viết trước khi quyết định triển khai.
