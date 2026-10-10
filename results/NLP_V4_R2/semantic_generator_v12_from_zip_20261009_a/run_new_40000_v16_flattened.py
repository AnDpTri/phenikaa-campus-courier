"""V16 40K. Flatten v12->v13->v14->v15 validated patches before runtime; CREATE-ONLY."""
import ast,re
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
src=(P/"role_semantic_generator_v12.py").read_text(encoding="utf-8")
tree=ast.parse((P/"run_semantic_generator_v13.py").read_text(encoding="utf-8"))
edits=None
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="edits" for x in node.targets):
  edits=ast.literal_eval(node.value)
assert edits and len(edits)>10
for before,after in edits:
 assert src.count(before)==1,(before,src.count(before))
 src=src.replace(before,after)
# V14 grammar fixes
p='base.rng=r'
assert src.count(p)==1
src=src.replace(p,p+'\nbase.VIA_ACTION=tuple(x for x in base.VIA_ACTION if not fold(x).startswith("buu kien"))')
revised='''EXPLICIT_GOAL=(
 "Lệnh giao mới có hiệu lực: {goal}",
 "Yêu cầu mới như sau: {goal}",
 "Thay cho lệnh cũ, robot thực hiện: {goal}",
 "Đây là chỉ dẫn cuối cùng: {goal}",
 "Sau khi hủy đích cũ, robot làm theo lệnh: {goal}",
)'''
assert len(re.findall(r'(?s)EXPLICIT_GOAL=\(.*?\n\)',src))==1
src=re.sub(r'(?s)EXPLICIT_GOAL=\(.*?\n\)',lambda _:revised,src,count=1)
# V15 grammar and cargo-consistency fixes
assert src.count("Ngoài kiện {parcel}")==1
src=src.replace("Ngoài kiện {parcel}","Ngoài {parcel}")
before='if fragile or r.random()<.35:blocks.append(r.choice(FRAGILITY[fragile]))'
after='if fragile or r.random()<.35:blocks.append(r.choice(tuple(q for q in FRAGILITY[fragile] if not fragile or ("linh kien" not in fold(q) and "do thuy tinh" not in fold(q)))))'
assert src.count(before)==1,(before,src.count(before))
src=src.replace(before,after)
# V16: large new corpus, shorter spatial descriptions and no schema changes.
edits16=[
 ("N=2000","N=40000"),
 ("SEED=202610091313","SEED=202610091616"),
 ("pilot_2000_v13_role_semantics.jsonl","specialized_new_40000_v16_role_semantics.jsonl"),
 ("pilot_2000_v13_provenance.jsonl","specialized_new_40000_v16_provenance.jsonl"),
 ("pilot_2000_v13_generation_report.json","specialized_new_40000_v16_generation_report.json"),
 ('sem_v13_{num:06d}','sem_v16_{num:06d}'),
]
for a,b in edits16:
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
short=r'''
_original_target=base.target
def _target_shorter(mode,excluded=()):
    spec,phrase,used=_original_target(mode,excluded)
    for a,b in (
      (" trong các địa điểm cùng loại, tính theo đường chim bay", " trong nhóm cùng loại"),
      (" trong số các địa điểm cùng loại", " trong nhóm cùng loại"),
      (" so với các địa điểm cùng loại", " so với những nơi cùng loại"),
      (" trong các địa điểm cùng loại trên bản đồ", " trong nhóm cùng loại"),
      (" trong các địa điểm cùng loại", " trong nhóm cùng loại"),
      (" trên toàn bản đồ", " trên bản đồ"),
      (" của toàn khuôn viên", " trong trường"),
      (" theo khoảng cách ô lưới", " theo ô lưới"),
      (" tính theo số ô lưới", " theo ô lưới"),
    ):phrase=phrase.replace(a,b)
    return spec,phrase,used
base.target=_target_shorter
'''
marker='base.VIA_ACTION=tuple(x for x in base.VIA_ACTION if not fold(x).startswith("buu kien"))'
assert src.count(marker)==1
src=src.replace(marker,marker+"\n"+short)
assert 'N=40000' in src and 'sem_v16_' in src
ctx={"__name__":"flattened_v16_40k","__file__":str(P/"role_semantic_generator_v12.py")}
exec(compile(src,str(P/"role_semantic_generator_v12.py"),"exec"),ctx)
ctx["main"]()
