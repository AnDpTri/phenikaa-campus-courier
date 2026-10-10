"""Fix runtime self-assertion to refer to V17 ID; all old files remain unchanged."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_new_40000_v17_semantic_final_candidate.py").read_text(encoding="utf-8")
old="w=w.replace(a,b)"
assert s.count(old)==1
s=s.replace(old,old+'\nassert w.count("and \\'sem_v16_\\' in src")==1\nw=w.replace("and \\'sem_v16_\\' in src","and \\'sem_v17_\\' in src")')
env={"__name__":"v17_ref_fixed","__file__":str(P/"run_new_40000_v17_semantic_final_candidate.py")}
exec(compile(s,str(P/"run_new_40000_v17_semantic_final_candidate.py"),"exec"),env)
