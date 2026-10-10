"""V14 corrected runner: fixes clause type and a legacy ungrammatical via template."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_semantic_generator_v13.py").read_text(encoding="utf-8")
for a,b in [
 ("202610091313","202610091414"),
 ("pilot_2000_v13_role_semantics.jsonl","pilot_2000_v14_role_semantics.jsonl"),
 ("pilot_2000_v13_provenance.jsonl","pilot_2000_v14_provenance.jsonl"),
 ("pilot_2000_v13_generation_report.json","pilot_2000_v14_generation_report.json"),
 ('sem_v13_{num:06d}','sem_v14_{num:06d}'),
]:
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
injection=r"""
import re
src=src.replace('base.rng=r','base.rng=r'+chr(10)+'base.VIA_ACTION=tuple(x for x in base.VIA_ACTION if not fold(x).startswith("buu kien"))')
bank='''EXPLICIT_GOAL=(
 "L\u1ec7nh giao m\u1edbi c\u00f3 hi\u1ec7u l\u1ef1c: {goal}",
 "Y\u00eau c\u1ea7u m\u1edbi nh\u01b0 sau: {goal}",
 "Thay cho l\u1ec7nh c\u0169, robot th\u1ef1c hi\u1ec7n: {goal}",
 "\u0110\u00e2y l\u00e0 ch\u1ec9 d\u1eabn cu\u1ed1i c\u00f9ng: {goal}",
 "Sau khi h\u1ee7y \u0111\u00edch c\u0169, robot l\u00e0m theo l\u1ec7nh: {goal}",
)'''
assert len(re.findall(r'(?s)EXPLICIT_GOAL=\(.*?\n\)',src))==1
src=re.sub(r'(?s)EXPLICIT_GOAL=\(.*?\n\)',lambda _:bank,src,count=1)
"""
marker='mod={"__name__"'
assert s.count(marker)==1
s=s.replace(marker,injection+"\n"+marker)
context={"__name__":"v14_fixed_source","__file__":str(P/"run_semantic_generator_v13.py")}
exec(compile(s,str(P/"run_semantic_generator_v13.py"),"exec"),context)
