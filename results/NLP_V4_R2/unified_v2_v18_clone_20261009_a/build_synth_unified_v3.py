"""Create-only V3 revision of integrated generator, based on inspected semantic errors."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\unified_v2_v18_clone_20261009_a")
s=(P/"synth_unified_v2.py").read_text(encoding="utf-8")
fixes=[
 ('f"{place} là địa điểm cực {w} trong nhóm cùng loại"',
  'f"{place} ở cực {w} trong nhóm cùng loại"'),
 ('"Tại {loc} đang có {pick} cần nhận để mang cùng chuyến."',
  '"Điểm lấy bổ sung {pick} là địa chỉ sau: {loc}. Robot cần nhận hàng ở đó."'),
 ('"Ở {loc} có {pick} đang chờ; robot nhận món đó rồi đi giao."',
  '"Địa chỉ nhận {pick} như sau: {loc}. Robot lấy hàng tại điểm đó rồi đi giao."'),
 ('"Có thêm chặng lấy {pick} ở {loc}; khi đã lấy xong mới đi giao."',
  '"Cần nhận {pick} tại địa điểm sau: {loc}. Lấy xong mới thực hiện chặng giao."'),
 ('"Đơn này yêu cầu ghé {loc} lấy {pick}; đích cuối đã nêu riêng."',
  '"Chặng nhận {pick} có địa chỉ: {loc}. Đích giao cuối cùng được nêu riêng."'),
 ('"Hàng trong thùng rất mong manh, không được làm rơi."',
  '"Hàng đang chuyển rất mong manh, không được làm rơi."'),
]
for a,b in fixes:
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
out=P/"synth_unified_v3.py"
with out.open("x",encoding="utf-8",newline="\n") as f:f.write(s)
print("REVISED_GENERATOR_CREATED",out,flush=True)
