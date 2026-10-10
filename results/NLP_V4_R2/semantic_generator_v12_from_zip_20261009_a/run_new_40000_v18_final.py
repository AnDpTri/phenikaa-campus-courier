"""V18 full 40k: natural mention-role grammar; excludes all earlier experimental datasets.
All outputs are newly created; existing files remain untouched.
"""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_new_40000_v16_flattened.py").read_text(encoding="utf-8")
for a,b in (
 ("202610091616","202610091818"),
 ("specialized_new_40000_v16_role_semantics.jsonl","specialized_new_40000_v18_role_semantics.jsonl"),
 ("specialized_new_40000_v16_provenance.jsonl","specialized_new_40000_v18_provenance.jsonl"),
 ("specialized_new_40000_v16_generation_report.json","specialized_new_40000_v18_generation_report.json"),
 ("sem_v16_{num:06d}","sem_v18_{num:06d}"),
 ("'sem_v16_' in src","'sem_v18_' in src"),
):
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
needle='ctx={"__name__"'
assert s.count(needle)==1
extra='''
old_awkward="Khu vực {loc} không liên quan đến việc giao nhận hiện tại."
natural="Thông tin về {loc} không liên quan đến chuyến giao này."
assert src.count(old_awkward)==1
src=src.replace(old_awkward,natural)
# Also reject exact folded duplicate texts in V16 and V17.
old_source='HERE.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl"]'
new_source='HERE.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl",\\n        HERE/"specialized_new_40000_v16_role_semantics.jsonl",\\n        HERE/"specialized_new_40000_v17_role_semantics.jsonl"]'
new_source=new_source.replace("\\\\n",chr(10)).replace("\\n",chr(10))
assert src.count(old_source)==1,(old_source,src.count(old_source))
src=src.replace(old_source,new_source)
'''
s=s.replace(needle,extra+"\n"+needle)
env={"__name__":"new_v18_40k","__file__":str(P/"run_new_40000_v16_flattened.py")}
exec(compile(s,str(P/"run_new_40000_v16_flattened.py"),"exec"),env)
