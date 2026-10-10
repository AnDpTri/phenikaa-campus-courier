"""Fresh V15 semantic-role contract audit; earlier artifacts remain unchanged."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_independent_role_audit_v14.py").read_text(encoding="utf-8")
assert s.count("v14")>=2
s=s.replace("v14","v15")
scope={"__name__":"independent_audit_v15","__file__":str(P/"run_independent_role_audit_v14.py")}
exec(compile(s,str(P/"run_independent_role_audit_v14.py"),"exec"),scope)
