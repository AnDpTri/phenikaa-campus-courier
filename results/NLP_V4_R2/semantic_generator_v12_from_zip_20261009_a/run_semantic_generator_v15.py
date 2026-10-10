"""V15 grammar and cargo-consistency final pilot, create-only."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_semantic_generator_v14_corrected.py").read_text(encoding="utf-8")
for a,b in [
 ("202610091414","202610091515"),
 ("pilot_2000_v14_role_semantics.jsonl","pilot_2000_v15_role_semantics.jsonl"),
 ("pilot_2000_v14_provenance.jsonl","pilot_2000_v15_provenance.jsonl"),
 ("pilot_2000_v14_generation_report.json","pilot_2000_v15_generation_report.json"),
 ('sem_v14_{num:06d}','sem_v15_{num:06d}'),
]:
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
addition=r'''
src=src.replace("Ngo\u00e0i ki\u1ec7n {parcel}", "Ngo\u00e0i {parcel}")
old='if fragile or r.random()<.35:blocks.append(r.choice(FRAGILITY[fragile]))'
new='if fragile or r.random()<.35:blocks.append(r.choice(tuple(q for q in FRAGILITY[fragile] if not fragile or ("linh kien" not in fold(q) and "do thuy tinh" not in fold(q)))))'
assert src.count(old)==1
src=src.replace(old,new)
'''
marker='"""\nmarker='
assert s.count(marker)==1
s=s.replace(marker,addition+marker)
scope={"__name__":"v15_grammar_patch","__file__":str(P/"run_semantic_generator_v14_corrected.py")}
exec(compile(s,str(P/"run_semantic_generator_v14_corrected.py"),"exec"),scope)
