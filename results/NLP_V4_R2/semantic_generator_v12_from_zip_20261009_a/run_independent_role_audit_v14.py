"""Run new independent semantic-role audit for V14 without editing previous reports."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
src=(P/"independent_role_audit_v13.py").read_text(encoding="utf-8")
assert src.count("v13")>=5
src=src.replace("v13","v14")
old='"huy","doi","khong con hieu luc"'
new='"huy","doi","thay dich giao","khong con hieu luc"'
assert src.count(old)==1,(old,src.count(old))
src=src.replace(old,new)
module={"__name__":"audit_independent_v14","__file__":str(P/"independent_role_audit_v13.py")}
exec(compile(src,str(P/"independent_role_audit_v13.py"),"exec"),module)
