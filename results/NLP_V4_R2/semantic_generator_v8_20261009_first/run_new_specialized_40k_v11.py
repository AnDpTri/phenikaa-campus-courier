"""V11 final new 40k candidate from approved typed V10; create-only and read-only originals.
Variation: more independently marked distractors; natural cargo wording.
"""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
w=(P/"run_typed_semantic_v10.py").read_text(encoding="utf-8")
for old,new in [
 ("2026100920","2026100921"),
 ("pilot_v10_2000.jsonl","specialized_new_40000_v11.jsonl"),
 ("pilot_v10_stats.json","specialized_new_40000_v11_stats.json"),
]:
 assert w.count(old)==1,(old,w.count(old))
 w=w.replace(old,new)
needle="]\nfor a,b in changes:"
assert w.count(needle)==1
patches=[
 ('N=2000','N=40000'),
 ('f"sem_v8_{i:06d}"','f"sem_v11_{i:06d}"'),
 ('ct=rng.choices([0,1,2],weights=[.40,.57,.03])[0]','ct=rng.choices([0,1,2],weights=[.33,.63,.04])[0]'),
 ('"hộp lọ thuốc thử bằng kính"','"bộ lọ thuốc thử bằng thủy tinh"'),
]
insert="".join(" ("+repr(a)+","+repr(b)+"),\n" for a,b in patches)
w=w.replace(needle,insert+needle)
ctx={"__name__":"typed_semantic_v11","__file__":str(P/"run_typed_semantic_v10.py")}
exec(compile(w,str(P/"run_typed_semantic_v10.py"),"exec"),ctx)
