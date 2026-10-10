"""Fresh 3k pilot of integrated V3 source after role-grammar corrections."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\unified_v2_v18_clone_20261009_a")
s=(P/"pilot_qa_unified_v1.py").read_text(encoding="utf-8")
for a,b in [
 ('synth_unified.py','synth_unified_v3.py'),
 ('pilot_unified_hard_3000_v1.jsonl','pilot_unified_hard_3000_v3.jsonl'),
 ('pilot_unified_provenance_3000_v1.jsonl','pilot_unified_provenance_3000_v3.jsonl'),
 ('pilot_unified_audit_v1.json','pilot_unified_audit_v3.json'),
 ('2026100923','2026100927'),
]:
 assert s.count(a)>=1,(a,s.count(a));s=s.replace(a,b)
env={"__name__":"unified_hard_v3_pilot","__file__":str(P/"pilot_qa_unified_v1.py")}
exec(compile(s,str(P/"pilot_qa_unified_v1.py"),"exec"),env)
