"""Creates V2 unified clone revision without modifying existing generator revisions."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\unified_v2_v18_clone_20261009_a")
src=(P/"synth_unified.py").read_text(encoding="utf-8")
changes=[
 ('"Địa điểm {old} từng là nơi nhận, nhưng không dùng nữa."',
  '"Nơi nhận cũ được ghi tại {old}, nhưng giờ không dùng nữa."'),
 ('"Đích {old} trong bản nháp là sai; bỏ địa chỉ đó."',
  '"Chỉ dẫn giao tới {old} trong bản nháp là sai; bỏ địa chỉ đó."'),
 ('"Chặng giao {old} bị xóa khỏi phiếu; dùng đích mới."',
  '"Chặng giao đến {old} bị xóa khỏi phiếu; dùng đích mới."'),
 ('"Khu {loc} không tiếp nhận hàng trong chuyến này."',
  '"Bộ phận tại {loc} không tiếp nhận hàng trong chuyến này."'),
 ('"Điểm {loc} đang ngừng hoạt động, không đến giao."',
  '"Khu vực tại {loc} đang ngừng hoạt động, không đến giao."'),
 ('"Nơi {loc} chỉ là thông tin loại trừ, không phải nơi bàn giao."',
  '"Thông tin {loc} chỉ là nội dung loại trừ, không phải nơi bàn giao."'),
 ('f"điểm sát {place} nhất, nhưng không lấy chính mốc đó"',
  'f"điểm sát {place} nhất, không tính chính {place}"'),
 ('f"địa điểm ngoài {place} có khoảng cách nhỏ nhất đến {place}"',
  'f"địa điểm cách {place} ngắn nhất, không tính chính {place}"'),
 ('f"điểm gần {place} nhất trong các điểm còn lại"',
  'f"điểm gần {place} nhất trong các điểm còn lại, không tính chính {place}"'),
 ('f"vị trí có cự ly tới {place} ngắn nhất sau khi bỏ chính mốc này"',
  'f"vị trí có cự ly tới {place} ngắn nhất, không tính chính {place}"'),
 ('f"địa danh không phải {place} nhưng nằm gần {place} nhất"',
  'f"địa danh gần {place} nhất, không tính chính {place}"'),
 ('f"điểm có khoảng cách ngắn nhất tới {place}, ngoại trừ mốc đó"',
  'f"điểm có khoảng cách ngắn nhất tới {place}, không tính chính {place}"'),
]
for a,b in changes:
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
out=P/"synth_unified_v2.py"
with out.open("x",encoding="utf-8",newline="\n") as f:f.write(src)
print("CREATED",out,flush=True)
