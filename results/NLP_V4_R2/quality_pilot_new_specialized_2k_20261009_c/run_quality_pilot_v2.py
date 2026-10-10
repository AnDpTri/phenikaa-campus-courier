"""Second independent quality pilot. Never edits or overwrites the original pilot.
Source transforms are in-memory-only and fully asserted for reproducibility.
"""
import types
from pathlib import Path
BASE=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
src=(BASE/"pilot_generator.py").read_text(encoding="utf-8")
changes=[
 ('N=2000\nSEED=202610091','N=2000\nSEED=202610092'),
 ('OUTPUT=OUT/"pilot_new_specialized_2000.jsonl"','OUTPUT=OUT/"pilot_new_specialized_2000_v2.jsonl"'),
 ('REPORT=OUT/"pilot_quality_report.json"','REPORT=OUT/"pilot_quality_report_v2.json"'),
 ('n_dist=R.choices([0,1,2,3],[.12,.27,.40,.21])[0]','n_dist=R.choices([0,1,2,3],[.63,.32,.05,0.0])[0]'),
 ('if R.random()<.94:clauses.append(R.choice(URGENT_TRUE if urgent else URGENT_FALSE))','if R.random()<.72:clauses.append(R.choice(URGENT_TRUE if urgent else URGENT_FALSE))'),
 ('if R.random()<.90:clauses.append(R.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE))','if R.random()<.72:clauses.append(R.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE))'),
 ('if R.random()<.60:clauses.append(R.choice(NEUTRAL))','if R.random()<.15:clauses.append(R.choice(NEUTRAL))'),
]
for before,after in changes:
 assert src.count(before)==1,(before,src.count(before))
 src=src.replace(before,after)
namespace={"__name__":"quality_pilot_v2_inmemory","__file__":str(BASE/"pilot_generator.py")}
exec(compile(src,str(BASE/"pilot_generator.py"),"exec"),namespace)
namespace["main"]()
