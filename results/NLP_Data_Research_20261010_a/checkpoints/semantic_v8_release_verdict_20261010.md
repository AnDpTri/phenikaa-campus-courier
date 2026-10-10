# Semantic V8 — biên bản kiểm duyệt 10/10/2026

**QUYẾT ĐỊNH: KHÔNG PHÊ DUYỆT làm dữ liệu train chính. CHỈ PHÊ DUYỆT làm ứng viên thí nghiệm bổ trợ có giám sát.**

## Phạm vi
- V8: 9.600 candidate train / 3.200 family challenge, 1.600 cảnh đối chứng.
- Đối chiếu 2.000 câu train thực tế và 20.000 câu sampled Scratch synthetic (read-only).
- Không sử dụng official validation/test, không thay model, không sửa code nguồn bên ngoài workspace.
- Các báo cáo có thể kiểm tra lại: checkpoints/semantic_v8_quality_audit.json; semantic_v8_release_audit_v2.json; semantic_v8_unmonitored_shortcuts_v3.json.

## PASS
- Schema, token bounds, spec labels, cùng goal trong scene.
- 8 kết hợp via/urgent/fragile cho mỗi scene; nhãn 50/50.
- Không trùng normalized trong 12.800 câu V8, giữa train/challenge, hoặc với original Scratch epoch 1 và V6 train.
- 10 dấu hiệu từ khóa (không, hủy, nhận, lấy, trước, sau, gấp, khẩn, vỡ, nhạy) invariant trên 8 tổ hợp nhãn trong cùng scene.
- Train/challenge tách theo 6/2 họ template (nhưng vẫn chia sẻ nền ngữ nghĩa và địa danh).

## FAIL / BLOCKER
### 1. Sai lệch phân bố real-vs-synthetic

| Metric | V8 candidate train | Real train (2k) |
|---|---:|---:|
| Median tokens | 67 | 46 |
| P90 tokens | 78 | 60 |
| >=70 tokens | 36.97% | 1.75% |
| via=True | 50.00% | 32.70% |
| urgent=True | 50.00% | 38.20% |
| fragile=True | 50.00% | 39.20% |
| Accented | 54.67% | 70.60% |
| Contains “nhận” | 80.75% | 51.60% |
| Contains “lấy” | 83.33% | 19.85% |

V8 luôn ghép đủ bốn mệnh đề goal/via/urgent/fragile. Nó thiếu tác vụ ngắn và tác vụ không nêu cờ urgent/fragile (mặc định False), và chưa đại diện phân bố thực tế.
Original Scratch sample 20K: median tokens 61, >=70 tokens 34.92%, via=True 54.68%, accent 27.27%.

### 2. Tương quan từ vựng khác vẫn dự đoán nhãn
Dù 10 từ khóa invariant, audit toàn bộ unigram/bigram với support >=3% cho thấy:
- Từ “đủ” xuất hiện 1.600/9.600, P(via=True | đủ)=0%.
- Từ “ngay” xuất hiện 2.400/9.600, P(urgent=True | ngay)=100%.
- Cụm “chịu rung” xuất hiện 1.196/9.600, P(fragile=True | chịu rung)=0%.

Đây thường là dấu hiệu ngữ nghĩa hợp lệ, KHÔNG có nghĩa phải cân bằng tất cả từ; nhưng mô hình vẫn có thể học thuộc phrase-specific patterns thay vì hiểu phạm vi phủ định/chỉ dẫn cuối.

### 3. Chưa chứng minh chất lượng thật
- Chưa có human annotation độc lập 100–200 câu phân tầng; đã xem một số câu đối chứng, đa phần ngữ nghĩa hợp lý nhưng còn văn phong template và câu phủ định lồng.
- Family challenge có đúng 2 họ và vẫn chung ý nghĩa/lexicon, chưa là blind human test.
- Chưa có controlled A/B training chứng minh V8 tăng exact-all hoặc recall trên câu mới.

## Kế hoạch đạt điều kiện PASS
1. Tạo phiên bản kế tiếp với nhiều câu ngắn, không luôn đủ bốn mệnh đề. Chủ động tạo câu thiếu urgent/fragile có nhãn False và các pha chỉ goal, goal+via, goal+urgent/fragile.
2. Cân chỉnh tỷ lệ pha train theo real deployment; V8 hard cases chỉ nên thêm một phần nhỏ 5–10% khi thí nghiệm, không thay baseline V4.
3. Tăng split độc lập thực sự theo template và loại câu, tránh chỉ một vài khuôn.
4. Rà soát nhãn thủ công phân tầng tối thiểu 100–200 câu, nhất là named/spatial và các câu hủy/đính chính.
5. Sau đó thử nghiệm A/B cùng model, seeds, steps và LR; giữ V8 family challenge riêng, đo all exact, via FPR/FNR, urgent/fragile recall và case consistency.

**Không tạo hay chỉnh sửa mã nguồn gốc, data/model/checkpoint ngoài vùng nghiên cứu.**
