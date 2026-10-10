# Báo cáo Chi tiết Kỷ lục Lượt nộp #1182 — Phenikaa Campus Courier 2026 (COURIER2)

**Thời gian nộp:** 10/10/2026, 08:54:59 AM (Server: 01:54:59 UTC)  
**Mã lượt nộp:** `#1182`  
**Điểm Public Leaderboard:** **70,39%** 🚀 *(CHÍNH THỨC PHÁ VỠ MỐC 70% — KỶ LỤC LỊCH SỬ MỚI)*  
**Trạng thái lượt nộp trong ngày:** Đã dùng 2/5 lượt hôm 10/10 (**còn 3 lượt**).

---

## 1. Kết quả & Toàn cảnh Lịch sử Nộp bài

| ID | Thời gian | File | SHA-256 (đầu) | Cấu hình | Public Score | So với mốc tương đương |
|---|---|---|---|---|---:|---|
| #1027 | 08/10 10:25 | `predictions.json` (v1) | `3ACC42B2` | Solver cũ 34 profile | 52,39% | Mốc bắt đầu giải đấu |
| #1045 | 08/10 12:57 | `v2_candidate` | `D5675EC6` | Candidate solver 48 profile | 55,67% | +3,28% |
| #1057 | 08/10 15:17 | `A_repair_only` | `96243F77` | v2 + sửa đồ thị (repair) | 57,56% | +1,89% |
| #1062 | 08/10 16:01 | `N_repair_nlpv1_t095` | `3951DA8C` | Repair + NLP v1 (goal 0.95) | 61,28% | +3,72% |
| #1101 | 09/10 01:01 | `V2_t095` | `2AA03DF6` | Repair + NLP v2 (goal 0.95) | 63,33% | +2,05% |
| #1102 | 09/10 01:18 | `V2_t080_solverTV` | `EA05F184` | Repair + NLP v2 + Solver trainval (via=2.0) | 64,78% | Khóa via bằng luật |
| #1144 | 09/10 22:20 | `V4_curated_t080` | `45E94B83` | Repair + NLP v4 Curated + Solver trainval (via=2.0) | 65,83% | +1,05% so với #1102 |
| #1180 | 10/10 08:24 | `V5_scratch_t080` | `D10193A8` | Repair + NLP v5 SF200 + Solver trainval (via=2.0) | 65,28% | Bản khóa via buổi sáng |
| #1107 | 09/10 09:08 | `V2_t080_via095_solverTV` | `F892E0CD` | Repair + NLP v2 (goal 0.8, via 0.95) + Solver trainval | 68,78% | Kỷ lục cũ của V2 |
| **#1182** | **10/10 08:54** | **`V5_scratch_via095`** | **`1B400996`** | **Repair + NLP V5 SF200 Scratch Ensemble (goal 0.8, via 0.95) + Solver trainval** | **70,39%** | **+1,61% so với kỷ lục #1107 (+5,11% so với #1180)** 🏆 |

---

## 2. Minh chứng Kết quả trên Hệ thống Chấm thi

![Nộp bài thành công lượt 2 ngày 10/10](/C:/Users/andan/.gemini/antigravity-ide/brain/b6d1673a-b4b3-4fa4-b837-07a2b9cab736/.user_uploaded/media_1791597296248.png)

![Bảng điểm Dashboard Phenikaa xác nhận điểm số kỷ lục 70.39%](/C:/Users/andan/.gemini/antigravity-ide/brain/b6d1673a-b4b3-4fa4-b837-07a2b9cab736/.user_uploaded/media_1791597321786.png)

---

## 3. Cấu hình Kỹ thuật Tái tạo Tuyệt đối

```powershell
py -3.12 scripts/create_submission.py --detector-device cuda `
  --nlp-model artifacts/nlp/neural_parser_v5_sf200_scratch_via095.pt `
  --nlp-goal-threshold 0.8 `
  --strategy-artifact artifacts/solver/candidate_strategy_trainval.joblib `
  --out results/submission_v5_via095_20261010/predictions.json `
  --diagnostics-dir results/submission_v5_via095_20261010/diagnostics
```

* **Đường dẫn file đầu ra:** `D:\phenikaa\results\submission_v5_via095_20261010\predictions.json`
* **Kích thước file:** 24.001 bytes (đúng 12.000 số nguyên $0 \dots 3$).
* **Mã băm SHA-256:** `1B400996A598AA21BEE3B727C563288CF86E0B4AFA53FC9BC28BFD7B785B2E9B`.
* **Artifacts sử dụng:**
  * **NLP Model:** `artifacts/nlp/neural_parser_v5_sf200_scratch_via095.pt` (V5 Semantic Fusion 200K, Scratch 3-seed Ensemble 5 Epochs, metadata ngưỡng `goal=0.8, via=0.95, flags=2.0`).
  * **Solver:** `artifacts/solver/candidate_strategy_trainval.joblib` (48 profiles tối ưu + cơ chế tự động sửa đồ thị `repair_unreachable`).
  * **Computer Vision:** SharedDetector 512 heatmap (CUDA), NodeNet, EdgeNet, WeatherClassifier (CV pipeline chuẩn, loại bỏ print 768).

---

## 4. Phân tích Nguyên nhân Kỹ thuật Tạo nên Điểm số 70,39%

Thành công vượt bậc của lượt nộp #1182 đến từ 2 yếu tố quyết định:

### 4.1. Bật đúng ngưỡng Via nơ-ron (`via = 0.95` mang lại +5,11% so với #1180)
* Ở bản #1180 (65,28%), mô hình bị khóa cứng `via = 2.0`, khiến mạng nơ-ron không được can thiệp vào bất kỳ điểm ghé nào.
* Sang #1182, khi mở `via = 0.95`, mạng nơ-ron V5 đã ghi đè thành công **342 điểm ghé Via** mà bộ luật regex cũ bị bỏ sót hoặc đoán nhầm.
* Tổng số bước đi thay đổi so với bản #1180 là **1.148 / 12.000 dự đoán (9,57%)**, giúp điểm tăng vọt từ **65,28% lên 70,39%** (+5,11 điểm phần trăm).

### 4.2. V5 Scratch Ensemble vượt trội hơn hẳn V2 cũ (+1,61% so với #1107)
So sánh trực tiếp 188 cảnh bất đồng giữa V5 và bản kỷ lục cũ V2 (#1107 — 68,78%):
1. **Khả năng chống bẫy phủ định:** 
   * V2 thường xuyên bị lừa bởi các câu cấm hoặc địa chỉ cũ (ví dụ Scene 21: câu *"Phòng học lớn không nhận hàng"* khiến V2 đoán nhầm thành `lecture`, trong khi V5 đoán chuẩn xác đích là `gate`).
   * Scene 30: V2 bị lừa bởi cụm *"Sân vận động là địa chỉ cũ"* thành `sports`, V5 đoán đúng quầy cơm sinh viên là `canteen`.
2. **Khả năng bắt Via dạng cấu trúc ghép:**
   * V2 bỏ sót hoàn toàn các câu có cấu trúc *"Lộ trình: qua X trước, sau đó mới tới Y"* (Scene 14, 15, 22), đoán Via là `None`.
   * V5 được train trên 200k câu Semantic Fusion nên bắt trọn vẹn các điểm ghé này kèm cả phương hướng địa lý (phía tây / bên trái).
3. **Độ an toàn hệ thống cao hơn:**
   * Số cảnh Fallback lỗi đồ thị trên Test giảm từ 42 cảnh (ở V2) xuống còn đúng **29 cảnh** (mức an toàn kỷ lục).
   * Số cảnh bỏ rơi điểm ghé (`via_dropped`) giảm từ 9 cảnh xuống chỉ còn **1 cảnh**.

---

## 5. Phân bố Dự đoán & Thống kê Chi tiết trên 1.200 Cảnh Test

* **Tổng số quan sát:** 12.000 lượt (10 robot $\times$ 1.200 cảnh).
* **Phân bố 4 hướng hành động (Cân bằng lý tưởng):**
  * `0 (UP)`: 2.986 (24,88%)
  * `1 (DOWN)`: 3.052 (25,43%)
  * `2 (LEFT)`: 2.945 (24,54%)
  * `3 (RIGHT)`: 3.017 (25,14%)
* **Chẩn đoán Fallback:**
  * Tổng cảnh Fallback: 29 / 1.200 cảnh (2,42%).
  * Mục tiêu lệch khỏi map (`resolver_goal_replaced`): **0 cảnh** (100% mục tiêu NLP đều có thật trên sơ đồ).
  * Cảnh báo cấu trúc đồ thị: 0 cảnh.

---

## 6. Chiến lược Khai thác cho 3 Lượt nộp Còn lại trong Ngày (10/10)

Với điểm số **70,39%**, đội đã chính thức vươn lên nhóm dẫn đầu tuyệt đối và tạo khoảng cách an toàn rất lớn trước hạn chót đóng cổng. Chúng ta còn **3 lượt nộp**:

1. **Bảo toàn và Khóa phiên bản Vàng:**
   * File `submission_v5_via095_20261010/predictions.json` và artifact `neural_parser_v5_sf200_scratch_via095.pt` phải được lưu trữ bất biến làm phương án nộp chốt chặn cho Private Test.
2. **Định hướng Thử nghiệm Thăm dò (Còn 3 lượt):**
   * *Thử nghiệm 1:* Quét nhẹ ngưỡng `goal_threshold = 0.75` (hạ nhẹ ngưỡng Goal để kích hoạt thêm một số câu phủ định mà nơ-ron V5 đang tự tin ở mức 0.75–0.79).
   * *Thử nghiệm 2:* Khai thác kết quả từ đợt huấn luyện ngữ nghĩa chuyên sâu `train_controlled_v9_ab_pilot.py` (nếu hoàn tất sớm trong buổi sáng).
   * *Nguyên tắc bắt buộc:* Mọi file trước khi nộp đều phải chạy đối chiếu diff trực tiếp với bản **70,39% (`1B400996`)** và chỉ nộp khi độ lệch an toàn $\le 3 - 5\%$.
