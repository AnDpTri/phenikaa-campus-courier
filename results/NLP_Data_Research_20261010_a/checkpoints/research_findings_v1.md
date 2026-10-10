# Báo cáo nghiên cứu lần 1 — V5-SF200-Scratch overfit

*Thư mục độc lập:* `D:\phenikaa\results\NLP_Data_Research_20261010_a`.

## Bằng chứng đo được (không phải suy đoán)

1. **Heldout lexicon `fragile` cực kỳ hẹp.** Bộ sinh chia phrase bank theo chỉ số `i % 5 == 4`; bank FRAGILE_POS/NEG gồm 6 câu, nên tập hard holdout chỉ dùng duy nhất một câu chứng cứ `fragile=True` và một câu `fragile=False` ở index 4. Trên 750 mẫu hard chẩn đoán, có 333 mẫu `fragile=True`, 417 mẫu `fragile=False`. Cả checkpoint Scratch E1 và E2 đều dự đoán `fragile=False` cho 750/750 mẫu: **recall true = 0%, specificity false = 100%, accuracy = 55.6%**. Đây là dấu hiệu lexical-shortcut mạnh.

2. **Thay duy nhất chứng cứ lexical fragile bằng câu cùng nghĩa đã thấy trong train** (không thay nhãn và nội dung còn lại): `fragile accuracy 55.6% → 100%` ở cả E1/E2; `all exact E1 31.07% → 51.07%; E2 31.20% → 52.80%`. Đây là *controlled ablation*, không phải bằng chứng cải thiện do đào tạo.

3. **Lệch prior via/genre:** train 200,000 mẫu có 56.99% `via`; holdout V2 51.60%, weighted 52.13%, hard 51.47%. Regime `pickup_reasoning` trong train có 69.95% `via`, rõ ràng không cân bằng. Train chỉ 32.55% câu có dấu tiếng Việt, hard holdout có 57.73%, gây domain/lexical shift.

4. **Họ câu lặp:** 200K câu train chỉ 91,910 prefix 8-token khác nhau (~45.96%). Đây là chỉ báo lặp cách mở câu, không tự nó chứng minh duplicate nguyên văn hay thiếu đa dạng ngữ nghĩa.

5. **Cặp đối chứng via:** 600 cặp khác nhau về vai trò `via` active-vs-cancelled. Old Scratch E1: `via=61.50%`, cả cặp đúng 23.0%; E2: `via=76.58%`, cả cặp đúng 53.17%. Tuy nhiên, E2 trên 600 mẫu active-via đúng 66.83%, thấp hơn 68.50% của E1. Khả năng phát hiện `via` thực sự còn yếu dù nhận ra hủy lệnh tốt hơn.

6. **Đã tạo 3 thử nghiệm generator ứng viên, chưa dùng vào train:** V2 và V3 lộ các shortcut từ vựng, V4 giữ 1000 cặp active-vs-cancelled, 2000 ví dụ, 50% `via=True`, 50% `urgent=True`, 50% `fragile=True`, 75% có dấu. Với cách tách từ bằng regex xử lý dấu câu, đo được `P(via|lấy)=P(via|nhận)=P(via|hủy)=P(via|trước)=0.5` trong candidate V4. *Giới hạn:* chỉ trường hợp đích/via named, câu đang khá khuôn mẫu, chưa được đánh giá bởi người và chưa kiểm tra tác dụng khi train.

## Nguyên nhân được xếp ưu tiên

1. **Lexical evidence leakage / heldout phrase-bank brittleness:** *đã có bằng chứng ablation trực tiếp*.
2. **Shortcut vai trò pickup + lệch phân bố via và phủ định/đính chính:** *có bằng chứng confusion / matched pairs*, cần thêm kiểm chứng.
3. **Giới hạn họ template và thiếu biến thể cú pháp:** *giả thuyết hợp lý*, cần đánh giá template-family split.
4. **Lịch LR và số epoch:** có thể khuếch đại confidence sai, nhưng không thay thế việc sửa dữ liệu.

## Hướng cải tiến cần thử nghiệm độc lập

- Tạo nhiều **cặp counterfactual** có cùng goal / flag / địa danh, chỉ thay ý nghĩa `via` bằng chỉ dẫn có hiệu lực, hủy, lịch sử và phủ định. Cân bằng các dấu từ `lấy`, `nhận`, `trước`, `hủy` giữa nhãn qua ngữ cảnh thực sự chứ không chỉ đảo từ.
- `fragile/urgent`: đa dạng paraphrase true/false với phủ định trong cùng họ từ, xen kẽ câu không nêu flags; bắt buộc QA label và test phép thay từ giữ nhãn.
- Đánh giá theo **semantic/template family holdout** bên cạnh lexical holdout; một heldout duy nhất cho mỗi bank nhỏ không đủ xác định độ khái quát.
- Kiểm nghiệm nhiều mức trộn candidate (ví dụ 5%, 10%, 20%) trên những run **mới riêng biệt** khi được phép, không làm ô nhiễm / sửa tiến trình train 3×5 hiện tại.
- Có thể triển khai CheckList-style minimal pairs/invariance + direction tests dựa trên các trường goal/via/urgent/fragile. 

## Giới hạn và nguyên tắc

- Không sử dụng official validation/test.
- Chưa thực hiện controlled retraining với candidate; hiện chỉ kiểm định dữ liệu và checkpoint cũ E1/E2 bằng CPU.
- Không chỉnh code, checkpoint, output hay tiến trình ở ngoài workspace này.
- Mọi nghiên cứu mới lưu checkpoints độc lập, có seeds, metadata và code.

## Chứng cứ đã lưu

- `checkpoints/corpus_audit_v1.json`
- `checkpoints/counterfactual_via_v1.json`
- `checkpoints/fragile_lexical_ablation_v1.json`
- `checkpoints/counterfactual_candidate_v2_audit.json`
- `checkpoints/counterfactual_candidate_v3_audit.json`
- `checkpoints/counterfactual_candidate_v4_audit.json`
- `checkpoints/round_000.json` (vòng đầu của nghiên cứu 4 giờ)

## Nguồn phương pháp

- Ribeiro et al., *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*, ACL 2020: https://aclanthology.org/2020.acl-main.442/
- Kaushik, Hovy & Lipton, *Learning the Difference That Makes a Difference with Counterfactually-Augmented Data*, ICLR 2020: https://openreview.net/forum?id=Sklgs0NFvr

Các tài liệu này hỗ trợ cách kiểm thử behavior/minimal-pair và thiết kế augmentation, không phải bằng chứng rằng candidate hiện tại tăng điểm.
