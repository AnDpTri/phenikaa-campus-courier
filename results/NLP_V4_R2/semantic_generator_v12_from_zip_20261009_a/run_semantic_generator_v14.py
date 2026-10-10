"""V14: further semantic and Vietnamese grammar repairs, immutable earlier pilots."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
w=(P/"run_semantic_generator_v13.py").read_text(encoding="utf-8")
for a,b in [
 ("202610091313","202610091414"),
 ("pilot_2000_v13_role_semantics.jsonl","pilot_2000_v14_role_semantics.jsonl"),
 ("pilot_2000_v13_provenance.jsonl","pilot_2000_v14_provenance.jsonl"),
 ("pilot_2000_v13_generation_report.json","pilot_2000_v14_generation_report.json"),
 ('sem_v13_{num:06d}','sem_v14_{num:06d}'),
]:
 assert w.count(a)==1,(a,w.count(a));w=w.replace(a,b)
extra=[
 ('base.rng=r','base.rng=r\nbase.VIA_ACTION=tuple(x for x in base.VIA_ACTION if "Bưu kiện {item}" not in x)'),
 ('"Địa chỉ giao hiện có hiệu lực là: {goal}"','"Nội dung giao mới được xác nhận: {goal}"'),
 ('"Lệnh giao cuối cùng cần làm là: {goal}"','"Robot thực hiện chỉ dẫn mới này: {goal}"'),
 ('"Cập nhật đích đến đã được chấp nhận: {goal}"','"Chỉ dẫn thay thế đã có hiệu lực: {goal}"'),
 ('"Bỏ địa chỉ cũ; thực hiện chỉ dẫn giao mới: {goal}"','"Không theo địa chỉ cũ; làm đúng lệnh giao mới: {goal}"'),
 ('"Nơi giao đúng của đơn hiện tại: {goal}"','"Chặng giao chính thức của đơn này như sau: {goal}"'),
]
needle="]\nfor a,b in edits:"
assert w.count(needle)==1
content="".join(" ("+repr(a)+","+repr(b)+"),\n" for a,b in extra)
w=w.replace(needle,content+needle)
ctx={"__name__":"new_v14_readonly_source","__file__":str(P/"run_semantic_generator_v13.py")}
exec(compile(w,str(P/"run_semantic_generator_v13.py"),"exec"),ctx)
