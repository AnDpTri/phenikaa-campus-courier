"""Independent full 40k V17 audit + new natural-language regression tests.
Source v16 audit retained untouched. V17 outputs create only.
"""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
src=(P/"independent_full_40000_semantic_audit_v16.py").read_text(encoding="utf-8")
# All identifiers/paths now name V17; the audited source is a separate file.
assert src.count("v16")>=5
src=src.replace("v16","v17")
needle='(r"\\bdia diem diem\\b","double_dia_diem"),'
assert src.count(needle)==1
src=src.replace(needle,needle+'\n                  (r"\\bkhu vuc khu\\b","double_khu_vuc_khu"),')
# New cross-version collision check; no V16 rows used for generation.
needle='("v11",P.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl")):'
assert src.count(needle)==1
src=src.replace(needle,'("v11",P.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl"),\n                  ("prior_rejected_v16",P/"specialized_new_40000_v16_role_semantics.jsonl")):')
env={"__name__":"independent_full_semantic_audit_v17","__file__":str(P/"independent_full_40000_semantic_audit_v16.py")}
exec(compile(src,str(P/"independent_full_40000_semantic_audit_v16.py"),"exec"),env)
