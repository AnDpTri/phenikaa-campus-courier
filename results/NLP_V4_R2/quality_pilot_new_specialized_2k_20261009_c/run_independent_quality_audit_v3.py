"""Independent new-file-only audit for final candidate v7."""
from pathlib import Path
D=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
source=(D/"run_independent_quality_audit_v2.py").read_text(encoding="utf-8")
assert source.count("pilot_v6")==2
assert source.count("pilot_new_specialized_2000_v6.jsonl")==1
assert source.count('independent_quality_audit_v2.json')==1
source=source.replace("pilot_v6","pilot_v7")
source=source.replace("pilot_new_specialized_2000_v6.jsonl","pilot_new_specialized_2000_v7.jsonl")
source=source.replace("independent_quality_audit_v2.json","independent_quality_audit_v3.json")
context={"__name__":"audit_quality_v7_inmemory","__file__":str(D/"run_independent_quality_audit_v2.py")}
exec(compile(source,str(D/"run_independent_quality_audit_v2.py"),"exec"),context)
