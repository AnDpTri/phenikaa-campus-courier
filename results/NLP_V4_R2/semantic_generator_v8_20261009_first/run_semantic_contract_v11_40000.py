"""Whole 40k semantic contracts and 300 stratified samples; create-only.
Derived in memory from the earlier independent audit, including revised fragile vocabulary.
"""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
src=(P/"independent_semantic_contract_audit_v10.py").read_text(encoding="utf-8")
fixes=[
 ('DATA=P/"pilot_v10_2000.jsonl"','DATA=P/"specialized_new_40000_v11.jsonl"'),
 ('OUT=P/"semantic_contract_v10.json"','OUT=P/"semantic_contract_v11_40000.json"'),
 ('SAMPLE=P/"semantic_stratified_v10_180.jsonl"','SAMPLE=P/"semantic_stratified_v11_300.jsonl"'),
 ('assert len(rows)==2000','assert len(rows)==40000'),
 ('rng=random.Random(202610092031)','rng=random.Random(202610092132)'),
 ('min(12,len(candidates))','min(24,len(candidates))'),
 ('180-len(picked)','300-len(picked)'),
 ('lengths[1000]','lengths[20000]'),
 ('lengths[1800]','lengths[36000]'),
 ('FRAG=set(module.CARGO_FRAGILE);NORMAL=set(module.CARGO_DURABLE)',
  'FRAG=set(module.CARGO_FRAGILE);NORMAL=set(module.CARGO_DURABLE)\\nITEMS.add("bộ lọ thuốc thử bằng thủy tinh");FRAG.add("bộ lọ thuốc thử bằng thủy tinh")'.replace("\\n","\n")),
]
for old,new in fixes:
 assert src.count(old)==1,(old,src.count(old))
 src=src.replace(old,new)
env={"__name__":"whole_40k_audit_inmemory","__file__":str(P/"independent_semantic_contract_audit_v10.py")}
exec(compile(src,str(P/"independent_semantic_contract_audit_v10.py"),"exec"),env)
