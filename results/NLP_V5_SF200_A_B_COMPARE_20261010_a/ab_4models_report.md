# Đối chiếu A/B: V2 seed 0 · V2 ensemble 3 seed · V4 R1 · V5-SF200

Tất cả checkpoint **chỉ đọc**, không train, không đụng tập test. Chỉ đánh giá neural parser, không áp dụng rule/hybrid.

## All exact accuracy

| Bộ kiểm thử (N) | V2 seed 0 | V2 ensemble 3 | V4 R1 | V5-SF200 |
|---|---:|---:|---:|---:|
| real_validation (300) | 64.33% | 68.33% | 63.00% | 63.67% |
| v2_final_holdout (3,000) | 43.53% | 47.90% | 56.27% | 53.13% |
| weighted_final_holdout (2,000) | 24.25% | 24.75% | 32.10% | 33.85% |
| hard_final_holdout (1,000) | 15.10% | 15.30% | 17.90% | 22.30% |
| handwritten_challenge (50) | 96.00% | 96.00% | 94.00% | 96.00% |

## Via exact accuracy

| Bộ kiểm thử | V2 seed 0 | V2 ensemble 3 | V4 R1 | V5-SF200 |
|---|---:|---:|---:|---:|
| real_validation | 89.33% | 92.33% | 82.33% | 85.33% |
| v2_final_holdout | 80.53% | 88.70% | 83.50% | 83.83% |
| weighted_final_holdout | 50.80% | 54.10% | 59.10% | 63.40% |
| hard_final_holdout | 40.00% | 41.40% | 46.50% | 51.90% |
| handwritten_challenge | 98.00% | 98.00% | 96.00% | 98.00% |

## Goal exact accuracy

| Bộ kiểm thử | V2 seed 0 | V2 ensemble 3 | V4 R1 | V5-SF200 |
|---|---:|---:|---:|---:|
| real_validation | 90.00% | 91.67% | 93.67% | 93.00% |
| v2_final_holdout | 61.07% | 66.43% | 75.43% | 72.67% |
| weighted_final_holdout | 54.15% | 57.80% | 63.75% | 69.65% |
| hard_final_holdout | 53.40% | 57.10% | 57.90% | 71.50% |
| handwritten_challenge | 98.00% | 100.00% | 98.00% | 96.00% |

## Paired V5 gain/loss: all exact

| Baseline | Tập kiểm thử | V5 đúng/base sai | V5 sai/base đúng | Δ all (điểm %) |
|---|---|---:|---:|---:|
| V2_seed0 | real_validation | 14 | 16 | -0.67 |
| V2_seed0 | v2_final_holdout | 467 | 179 | +9.60 |
| V2_seed0 | weighted_final_holdout | 242 | 50 | +9.60 |
| V2_seed0 | hard_final_holdout | 82 | 10 | +7.20 |
| V2_seed0 | handwritten_challenge | 0 | 0 | +0.00 |
| V2_ensemble3 | real_validation | 14 | 28 | -4.67 |
| V2_ensemble3 | v2_final_holdout | 446 | 289 | +5.23 |
| V2_ensemble3 | weighted_final_holdout | 257 | 75 | +9.10 |
| V2_ensemble3 | hard_final_holdout | 85 | 15 | +7.00 |
| V2_ensemble3 | handwritten_challenge | 1 | 1 | +0.00 |
| V4_R1 | real_validation | 8 | 6 | +0.67 |
| V4_R1 | v2_final_holdout | 169 | 263 | -3.13 |
| V4_R1 | weighted_final_holdout | 126 | 91 | +1.75 |
| V4_R1 | hard_final_holdout | 57 | 13 | +4.40 |
| V4_R1 | handwritten_challenge | 1 | 0 | +2.00 |

## Hạn chế diễn giải
- V5 chỉ được chốt theo tập dev; final synthetic holdout dùng seed khác nhưng không tách hoàn toàn các họ template.
- Validation thực tế nhỏ và có thể đã tham gia lựa chọn mô hình trong lịch sử; không đại diện cho hidden test.
- Ensemble 3 mô hình có chi phí suy luận lớn hơn mô hình đơn.
- Bộ 49 câu khó cũ chỉ được đánh giá nếu có tập gán nhãn chính thức; phiên bản này không tự tạo/giả mạo 49 câu.
- Các mẫu V5 đúng/sai tương đối so với baseline nằm trong ab_4models_discordant_examples.jsonl.

## Đầu ra
- ab_4models_results.json: toàn bộ chỉ số, nhóm tình huống, CI và checkpoint SHA.
- ab_4models_discordant_examples.jsonl: ví dụ sai/đúng bất đồng đã gắn nhãn.
- Tổng thời gian đánh giá: 69.6 giây.
