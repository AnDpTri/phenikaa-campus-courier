"""Independent V18 all 40k audit, no earlier audit files modified."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
src=(P/"independent_full_40000_semantic_audit_v16.py").read_text(encoding="utf-8")
assert src.count("v16")>=5
src=src.replace("v16","v18")
needle='(r"\\bdia diem diem\\b","double_dia_diem"),'
assert src.count(needle)==1
src=src.replace(needle,needle+'\n                  (r"\\bkhu vuc khu\\b","double_khu_vuc_khu"),\n                  (r"\\bdia diem khu\\b","double_dia_diem_khu"),')
needle='("v11",P.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl")):'
assert src.count(needle)==1
src=src.replace(needle,'("v11",P.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl"),\n                  ("prior_v16",P/"specialized_new_40000_v16_role_semantics.jsonl"),\n                  ("prior_v17",P/"specialized_new_40000_v17_role_semantics.jsonl")):')
env={"__name__":"full_semantic_v18_audit","__file__":str(P/"independent_full_40000_semantic_audit_v16.py")}
exec(compile(src,str(P/"independent_full_40000_semantic_audit_v16.py"),"exec"),env)
