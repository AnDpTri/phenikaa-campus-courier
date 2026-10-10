"""Sixth pilot: compositional grammar compatibility and compactness; create only."""
from pathlib import Path
BASE=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
src=(BASE/"pilot_generator.py").read_text(encoding="utf-8")
changes=[
 ('N=2000\nSEED=202610091','N=2000\nSEED=202610096'),
 ('OUTPUT=OUT/"pilot_new_specialized_2000.jsonl"','OUTPUT=OUT/"pilot_new_specialized_2000_v6.jsonl"'),
 ('REPORT=OUT/"pilot_quality_report.json"','REPORT=OUT/"pilot_quality_report_v6.json"'),
 ('n_dist=R.choices([0,1,2,3],[.12,.27,.40,.21])[0]','n_dist=R.choices([0,1,2,3],[.43,.51,.06,0.0])[0]'),
 ('if R.random()<.94:clauses.append(R.choice(URGENT_TRUE if urgent else URGENT_FALSE))','if urgent or R.random()<.50:clauses.append(R.choice(URGENT_TRUE if urgent else URGENT_FALSE))'),
 ('if R.random()<.90:clauses.append(R.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE))','if fragile or R.random()<.50:clauses.append(R.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE))'),
 ('if R.random()<.60:clauses.append(R.choice(NEUTRAL))','if R.random()<.10:clauses.append(R.choice(NEUTRAL))'),
 ('clauses.append(R.choice(CLOSE))','if R.random()<.60:clauses.append(R.choice(CLOSE))'),
 ('"Đây là đơn khẩn, xử lý ngay khi nhận."','"Ban đầu bảo không vội, giờ đã đổi thành giao khẩn."'),
 ('"Đồ gốm bên trong có thể vỡ nếu rơi."','"Trước nói hàng bền; đính chính: đồ gốm dễ vỡ."'),
 ('"Không cần ưu tiên hỏa tốc."','"Tin cũ báo khẩn đã hủy, hiện không gấp."'),
 ('"Kiện hàng chắc chắn, không thuộc loại dễ vỡ."','"Bản trước ghi dễ vỡ, đính chính: hàng rất bền."'),
 ('f"{place} nằm ở {R.choice(DIRECTION[mode])} trên sơ đồ"','f"{place} nằm ở {R.choice(DIRECTION[mode])} của bản đồ"'),
 ('"north":("phía bắc","phía trên","về hướng bắc"),','"north":("phía bắc","nửa phía trên","khu vực phía bắc"),'),
 ('"south":("phía nam","phía dưới","về hướng nam"),','"south":("phía nam","nửa phía dưới","khu vực phía nam"),'),
 ('"west":("phía tây","bên trái","về hướng tây"),','"west":("phía tây","bên trái","khu vực phía tây"),'),
 ('"east":("phía đông","bên phải","về hướng đông"),','"east":("phía đông","bên phải","khu vực phía đông"),'),
 ('"giao kiện {item} cho người trực ở {goal}."','"giao {item} cho người trực ở {goal}."'),
 ('f"trong hai {place}, chọn nơi có khoảng cách tới {anchor} ngắn hơn"','f"{place} có khoảng cách tới {anchor} ngắn hơn điểm cùng loại còn lại"'),
 ('f"{place} phía gần {anchor} hơn chỗ thứ hai"','f"{place} nằm gần {anchor} hơn điểm cùng loại thứ hai"'),
 ('f"trong hai {place}, chọn nơi ở xa {anchor} hơn"','f"{place} ở xa {anchor} hơn điểm cùng loại còn lại"'),
 ('f"{place} phía xa {anchor} hơn chỗ thứ hai"','f"{place} nằm xa {anchor} hơn điểm cùng loại thứ hai"'),
]
for before,after in changes:
 assert src.count(before)==1,(before,src.count(before))
 src=src.replace(before,after)
namespace={"__name__":"quality_pilot_v6_inmemory","__file__":str(BASE/"pilot_generator.py")}
exec(compile(src,str(BASE/"pilot_generator.py"),"exec"),namespace)
namespace["main"]()
