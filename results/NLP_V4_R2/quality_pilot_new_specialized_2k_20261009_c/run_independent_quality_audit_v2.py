"""Run independent quality gates on original pilot series + v6, without rewriting audit v1."""
from pathlib import Path
D=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
src=(D/"audit_quality.py").read_text(encoding="utf-8")
changes=[
 ('"pilot_v3":D/"pilot_new_specialized_2000_v3.jsonl"}','"pilot_v3":D/"pilot_new_specialized_2000_v3.jsonl","pilot_v6":D/"pilot_new_specialized_2000_v6.jsonl"}'),
 ('OUT=D/"independent_quality_audit_v1.json"','OUT=D/"independent_quality_audit_v2.json"'),
 ('if k=="pilot_v3" else banks','if k in ("pilot_v3","pilot_v6") else banks'),
 ('"wrong_direction_phrasing":r"\\bnam o ve huong\\b"','"wrong_direction_phrasing":r"\\b(nam o ve huong|phia tren tren|phia duoi tren|giao kien the|o trong hai|den trong hai)\\b"'),
]
for a,b in changes:
 assert src.count(a)==1,(a,src.count(a))
 src=src.replace(a,b)
namespace={"__name__":"audit_quality_v2_inmemory","__file__":str(D/"audit_quality.py")}
exec(compile(src,str(D/"audit_quality.py"),"exec"),namespace)
namespace["main"]()
