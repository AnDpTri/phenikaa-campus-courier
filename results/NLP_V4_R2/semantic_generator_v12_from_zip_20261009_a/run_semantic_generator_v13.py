"""V13 semantic brevity pilot; uses new-in-memory source transforms, never modifies V12."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
src=(P/"role_semantic_generator_v12.py").read_text(encoding="utf-8")
edits=[
 ("SEED=202610091212","SEED=202610091313"),
 ("pilot_2000_v12_role_semantics.jsonl","pilot_2000_v13_role_semantics.jsonl"),
 ("pilot_2000_v12_provenance.jsonl","pilot_2000_v13_provenance.jsonl"),
 ("pilot_2000_v12_generation_report.json","pilot_2000_v13_generation_report.json"),
 ('f"sem_v12_{num:06d}"','f"sem_v13_{num:06d}"'),
 ('weights=(.36,.55,.09)','weights=(.57,.41,.02)'),
 ('r.random()<.59:blocks.append(r.choice(URGENCY[urgent]))','r.random()<.35:blocks.append(r.choice(URGENCY[urgent]))'),
 ('r.random()<.59:blocks.append(r.choice(FRAGILITY[fragile]))','r.random()<.35:blocks.append(r.choice(FRAGILITY[fragile]))'),
 ('r.random()<.24 else None','r.random()<.13 else None'),
 ('if r.random()<.22:blocks.append(r.choice(NEUTRAL_END))','if r.random()<.06:blocks.append(r.choice(NEUTRAL_END))'),
 ('role=r.choice(tuple(NEGATIVE_ROLE))','role=r.choice(("historical","not_related") if goal.type is None else tuple(NEGATIVE_ROLE))'),
 ('"Trước đó robot được yêu cầu mang {parcel} tới {old}. Đính chính: không được giao theo lệnh cũ."',
  '"Lệnh chuyển {parcel} tới {old} đã bị hủy; dùng đích mới."'),
 ('"Phiếu đầu tiên yêu cầu giao {parcel} tới {old}. Chỉ thị ấy đã bị hủy."',
  '"Phiếu cũ giao {parcel} tại {old}, nhưng nay đã hủy."'),
 ('"Hãy xóa khỏi lộ trình yêu cầu giao đến {old}; nơi giao chính thức nằm trong lệnh sau."',
  '"Hủy đích giao {old}; thực hiện lệnh mới ở câu sau."'),
 ('"Nơi giao đúng của đơn hiện tại: {goal}"','"Đích mới: {goal}"'),
 ('"Cập nhật đích đến đã được chấp nhận: {goal}"','"Địa chỉ thay thế: {goal}"'),
]
for a,b in edits:
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
mod={"__name__":"pilot_v13_inmemory","__file__":str(P/"role_semantic_generator_v12.py")}
exec(compile(src,str(P/"role_semantic_generator_v12.py"),"exec"),mod)
mod["main"]()
