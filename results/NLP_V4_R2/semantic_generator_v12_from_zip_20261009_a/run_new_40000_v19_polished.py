"""V19 grammar-polished 40K: eliminate ambiguous anchor-near phrasing and repeated location predicate.
Create-only, excludes all prior R2 pilots/old curated/original.
"""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_new_40000_v16_flattened.py").read_text(encoding="utf-8")
for a,b in (
 ("202610091616","202610091919"),
 ("specialized_new_40000_v16_role_semantics.jsonl","specialized_new_40000_v19_role_semantics.jsonl"),
 ("specialized_new_40000_v16_provenance.jsonl","specialized_new_40000_v19_provenance.jsonl"),
 ("specialized_new_40000_v16_generation_report.json","specialized_new_40000_v19_generation_report.json"),
 ("sem_v16_{num:06d}","sem_v19_{num:06d}"),
 ("'sem_v16_' in src","'sem_v19_' in src"),
):
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
marker='ctx={"__name__"'
assert s.count(marker)==1
patch=r'''
# Role/distractor grammar.
old="Khu vực {loc} không liên quan đến việc giao nhận hiện tại."
assert src.count(old)==1
src=src.replace(old,"Thông tin về {loc} không liên quan đến chuyến giao này.")
# The supplemental pickup is an action, not a place-identification clause.
old="Chặng trung gian lấy {extra} nằm ở {via}."
assert src.count(old)==1
src=src.replace(old,"Chặng lấy {extra} diễn ra tại {via}.")
# Non-supplemental via text lives in the typed grammar bank.
bankmarker='base.VIA_ACTION=tuple(x for x in base.VIA_ACTION if not fold(x).startswith("buu kien"))'
assert src.count(bankmarker)==1
src=src.replace(bankmarker,bankmarker+chr(10)+
 'base.VIA_ACTION=tuple(x.replace("Chặng trung gian lấy {item} nằm ở {loc}.","Chặng lấy {item} diễn ra tại {loc}.") for x in base.VIA_ACTION)')
# A clear natural-language expression of anchor_near excludes the anchor itself.
natural=r"""
_prev_target=base.target
def _target_polished(mode, excluded=()):
    spec,phrase,used=_prev_target(mode,excluded)
    if mode=="anchor_near" and phrase.startswith("địa danh khác "):
        m=re.match(r"^địa danh khác (.*?) gần mốc (.*?) nhất theo ô lưới$",phrase)
        if m and m.group(1)==m.group(2):
            anchor=m.group(1)
            phrase=f"địa danh gần {anchor} nhất theo ô lưới, không tính chính {anchor}"
    return spec,phrase,used
base.target=_target_polished
"""
old="base.target=_target_shorter"
assert src.count(old)==1
src=src.replace(old,old+"\n"+natural)
# Exact normalized-text exclusion of prior experiments.
old='HERE.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl"]'
new='HERE.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl",\n        HERE/"specialized_new_40000_v16_role_semantics.jsonl",\n        HERE/"specialized_new_40000_v17_role_semantics.jsonl",\n        HERE/"specialized_new_40000_v18_role_semantics.jsonl"]'
assert src.count(old)==1
src=src.replace(old,new)
'''
s=s.replace(marker,patch+"\n"+marker)
env={"__name__":"new_40000_v19","__file__":str(P/"run_new_40000_v16_flattened.py")}
exec(compile(s,str(P/"run_new_40000_v16_flattened.py"),"exec"),env)
