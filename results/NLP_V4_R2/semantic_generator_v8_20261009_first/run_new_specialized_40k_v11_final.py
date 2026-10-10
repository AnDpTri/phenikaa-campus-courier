"""Corrected v11 runner; retain earlier runner scripts exactly as created."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
s=(P/"run_new_specialized_40k_v11_corrected.py").read_text(encoding="utf-8")
for a,b in [
("('SEED=2026100920','SEED=2026100921')","('SEED=2026100919','SEED=2026100921')"),
('OUTPUT=BASE/"pilot_v10_2000.jsonl"','OUTPUT=BASE/"pilot_v9_2000.jsonl"'),
('META=BASE/"pilot_v10_stats.json"','META=BASE/"pilot_v9_stats.json"'),
]:
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
scope={"__name__":"runner_v11_correct_source","__file__":str(P/"run_new_specialized_40k_v11_corrected.py")}
exec(compile(s,str(P/"run_new_specialized_40k_v11_corrected.py"),"exec"),scope)
