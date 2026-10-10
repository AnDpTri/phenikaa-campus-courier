"""V17 40K role-aware dataset; change seed and repair 'khu vực khu...' source grammar.
Reads all prior scripts, writes new V17 dataset only; refuses overwrite.
"""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
w=(P/"run_new_40000_v16_flattened.py").read_text(encoding="utf-8")
for a,b in [
 ("202610091616","202610091717"),
 ("specialized_new_40000_v16_role_semantics.jsonl","specialized_new_40000_v17_role_semantics.jsonl"),
 ("specialized_new_40000_v16_provenance.jsonl","specialized_new_40000_v17_provenance.jsonl"),
 ("specialized_new_40000_v16_generation_report.json","specialized_new_40000_v17_generation_report.json"),
 ("sem_v16_{num:06d}","sem_v17_{num:06d}"),
]:
 assert w.count(a)==1,(a,w.count(a))
 w=w.replace(a,b)
newpatch='''
old_awkward="Khu vực {loc} không liên quan đến việc giao nhận hiện tại."
new_natural="Địa điểm {loc} không liên quan đến chuyến giao này."
assert src.count(old_awkward)==1,(old_awkward,src.count(old_awkward))
src=src.replace(old_awkward,new_natural)
'''
marker='ctx={"__name__"'
assert w.count(marker)==1
w=w.replace(marker,newpatch+"\n"+marker)
ns={"__name__":"v17_generator_immutable","__file__":str(P/"run_new_40000_v16_flattened.py")}
exec(compile(w,str(P/"run_new_40000_v16_flattened.py"),"exec"),ns)
