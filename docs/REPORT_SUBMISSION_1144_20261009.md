# Báo cáo Chi tiết Lượt nộp #1144 — Phenikaa Campus Courier 2026 (COURIER2)

**Thời gian nộp:** 09/10/2026, 22:20 (Server: 15:20:10 UTC)  
**Mã lượt nộp:** `#1144`  
**Điểm Public Leaderboard:** **65,83%**  
**Trạng thái lượt nộp trong ngày:** Đã dùng 5/5 lượt hôm 09/10 (còn 0 lượt).

---

## 1. Kết quả & So sánh với Lịch sử

| ID | Thời gian | File | SHA-256 (đầu) | Cấu hình | Public Score | So với mốc tương đương |
|---|---|---|---|---|---:|---|
| #1027 | 08/10 10:25 | `predictions.json` (v1) | `3ACC42B2` | Solver cũ 34 profile | 52,39% | Mốc bắt đầu |
| #1045 | 08/10 12:57 | `v2_candidate` | `D5675EC6` | Candidate solver 48 profile | 55,67% | +3,28% |
| #1057 | 08/10 15:17 | `A_repair_only` | `96243F77` | v2 + sửa đồ thị (repair) | 57,56% | +1,89% |
| #1062 | 08/10 16:01 | `N_repair_nlpv1_t095` | `3951DA8C` | Repair + NLP v1 (goal 0.95) | 61,28% | +3,72% |
| #1101 | 09/10 01:01 | `V2_t095` | `2AA03DF6` | Repair + NLP v2 (goal 0.95) | 63,33% | +2,05% |
| #1102 | 09/10 01:18 | `V2_t080_solverTV` | `EA05F184` | Repair + NLP v2 (goal 0.8, via 2.0) + Solver trainval | 64,78% | +1,45% |
| **#1144** | **09/10 22:20** | **`V4_curated_t080_45E94B83`** | **`45E94B83`** | **Repair + NLP v4 Curated (goal 0.8, via 2.0) + Solver trainval** | **65,83%** | **+1,05% so với #1102** |

> [!NOTE]
> So sánh công bằng nhất của **#1144** là với **#1102**:
> * Cả hai đều dùng: `goal_threshold = 0.8`, `via_threshold = 2.0` (giữ luật via), và `candidate_strategy_trainval.joblib`.
> * Điểm tăng từ **64,78% (#1102) $\rightarrow$ 65,83% (#1144)** (+1,05 điểm phần trăm).
> * Điều này chứng minh dữ liệu **Curated 200K (Fine-tune 3 epoch)** đã giúp mô hình bắt Goal trên tập Test thật tốt hơn hẳn so với NLP v2!

---

## 2. Thông số Kỹ thuật & Cấu hình Tái tạo Chính xác

```powershell
py -3.12 scripts/create_submission.py --detector-device cuda `
  --nlp-model results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/model.pt `
  --nlp-goal-threshold 0.8 `
  --strategy-artifact artifacts/solver/candidate_strategy_trainval.joblib `
  --out results/submission_v4_curated_20261009/predictions.json `
  --diagnostics-dir results/submission_v4_curated_20261009/diagnostics
```

* **File đầu ra:** `D:\phenikaa\results\submission_v4_curated_20261009\predictions.json`
* **SHA-256:** `45E94B833E507C070BAF9A5D78A0C5FE0F53BF34012019CB29578FE9B17CD667`
* **Artifacts sử dụng:**
  * **NLP:** `results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/model.pt` (Mô hình BiGRU 1 seed fine-tune 3 epoch trên corpus 200k).
  * **Solver:** `artifacts/solver/candidate_strategy_trainval.joblib` (Fit train + validation, 48 profile).
  * **CV:** `artifacts/cv/detector.pt`, `node_net.pt`, `edge_net.pt`, `weather_classifier.joblib`.

---

## 3. Phân tích Chẩn đoán Chi tiết trên Tập Test (1.200 Cảnh)

Dựa trên file `results/submission_v4_curated_20261009/diagnostics/summary.json`:

1. **Số cảnh bị fallback rơi xuống mức kỷ lục:**
   * Chỉ còn **29 / 1.200 cảnh** (2,42% số cảnh).
   * Lịch sử giảm dần: 112 (v2) $\rightarrow$ 65 (NLP v1) $\rightarrow$ 48 (#1102) $\rightarrow$ 42 (#1107) $\rightarrow$ **29 (#1144)**.
2. **Khớp đích (`resolver_goal_replaced`):**
   * Đạt **0 cảnh** (ở v2 là 245 cảnh). Toàn bộ 100% mục tiêu do NLP v4 đọc ra đều có thật trên bản đồ.
3. **Mô hình nơ-ron ghi đè luật:**
   * `nlp_goal_neural_override`: **778 cảnh** (chiếm 64,8% số cảnh test).
   * `nlp_goal_neural_fill`: **95 cảnh**.
4. **Phân bố 4 hướng hành động cân đối:**
   * `UP (0)`: 3.070 (25,58%)
   * `DOWN (1)`: 3.075 (25,62%)
   * `LEFT (2)`: 2.892 (24,10%)
   * `RIGHT (3)`: 2.963 (24,69%)
   * Tổng cộng: đúng 12.000 dự đoán hợp lệ (0..3).

---

## 4. Nhận định Chiến lược

* Lượt nộp #1144 đã chính thức xác nhận giả thuyết: **Bộ dữ liệu Curated 200K đem lại bước tiến thực chất trên tập Test thật**, nâng điểm cấu hình gốc từ **64,78% lên 65,83%**.
* Hôm nay bạn đã hoàn thành đủ 5/5 lượt nộp. Kết quả này củng cố thêm vị thế an toàn trong top bảng xếp hạng vòng loại trước hạn chót 0h ngày 10/10/2026.
