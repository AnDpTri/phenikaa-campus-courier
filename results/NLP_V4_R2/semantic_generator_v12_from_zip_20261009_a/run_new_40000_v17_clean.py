"""V17 validated flattening wrapper; create-only."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_new_40000_v16_flattened.py").read_text(encoding="utf-8")
for a,b in (
 ("202610091616","202610091717"),
 ("specialized_new_40000_v16_role_semantics.jsonl","specialized_new_40000_v17_role_semantics.jsonl"),
 ("specialized_new_40000_v16_provenance.jsonl","specialized_new_40000_v17_provenance.jsonl"),
 ("specialized_new_40000_v16_generation_report.json","specialized_new_40000_v17_generation_report.json"),
 ("sem_v16_{num:06d}","sem_v17_{num:06d}"),
 ("'sem_v16_' in src","'sem_v17_' in src"),
):
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
needle='ctx={"__name__"'
assert s.count(needle)==1
injection='''# Fix the last observed recurrent noun-stacking construction.
old="Khu vực {loc} không liên quan đến việc giao nhận hiện tại."
assert src.count(old)==1
src=src.replace(old,"Địa điểm {loc} không liên quan đến chuyến giao này.")
'''
s=s.replace(needle,injection+"\n"+needle)
ctx={"__name__":"runner_v17_flattened","__file__":str(P/"run_new_40000_v16_flattened.py")}
exec(compile(s,str(P/"run_new_40000_v16_flattened.py"),"exec"),ctx)
