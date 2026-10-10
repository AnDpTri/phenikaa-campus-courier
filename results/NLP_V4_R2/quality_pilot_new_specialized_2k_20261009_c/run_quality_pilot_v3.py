"""Third quality pilot: corrects missing evidence for true flags, preserves fixed old files."""
from pathlib import Path
BASE=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
src=(BASE/"pilot_generator.py").read_text(encoding="utf-8")
changes=[
 ('N=2000\nSEED=202610091','N=2000\nSEED=202610093'),
 ('OUTPUT=OUT/"pilot_new_specialized_2000.jsonl"','OUTPUT=OUT/"pilot_new_specialized_2000_v3.jsonl"'),
 ('REPORT=OUT/"pilot_quality_report.json"','REPORT=OUT/"pilot_quality_report_v3.json"'),
 ('n_dist=R.choices([0,1,2,3],[.12,.27,.40,.21])[0]','n_dist=R.choices([0,1,2,3],[.57,.38,.05,0.0])[0]'),
 ('if R.random()<.94:clauses.append(R.choice(URGENT_TRUE if urgent else URGENT_FALSE))','if urgent or R.random()<.50:clauses.append(R.choice(URGENT_TRUE if urgent else URGENT_FALSE))'),
 ('if R.random()<.90:clauses.append(R.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE))','if fragile or R.random()<.50:clauses.append(R.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE))'),
 ('if R.random()<.60:clauses.append(R.choice(NEUTRAL))','if R.random()<.10:clauses.append(R.choice(NEUTRAL))'),
 ('"Đây là đơn khẩn, xử lý ngay khi nhận."','"Ban đầu bảo không vội, giờ đã đổi thành giao khẩn."'),
 ('"Đồ gốm bên trong có thể vỡ nếu rơi."','"Trước nói hàng bền; đính chính: đồ gốm dễ vỡ."'),
 ('"Không cần ưu tiên hỏa tốc."','"Tin cũ báo khẩn đã hủy, hiện không gấp."'),
 ('"Kiện hàng chắc chắn, không thuộc loại dễ vỡ."','"Bản trước ghi dễ vỡ, đính chính: hàng rất bền."'),
]
for before,after in changes:
 assert src.count(before)==1,(before,src.count(before))
 src=src.replace(before,after)
namespace={"__name__":"quality_pilot_v3_inmemory","__file__":str(BASE/"pilot_generator.py")}
exec(compile(src,str(BASE/"pilot_generator.py"),"exec"),namespace)
namespace["main"]()
