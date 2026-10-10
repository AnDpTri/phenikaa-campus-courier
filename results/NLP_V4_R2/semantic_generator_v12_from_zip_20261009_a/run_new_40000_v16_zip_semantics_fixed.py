"""Fixed V16 full 40k runner, using V15 in-memory code chain; create-only."""
from pathlib import Path
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
s=(P/"run_semantic_generator_v15.py").read_text(encoding="utf-8")
for a,b in [
 ("202610091515","202610091616"),
 ("pilot_2000_v15_role_semantics.jsonl","specialized_new_40000_v16_role_semantics.jsonl"),
 ("pilot_2000_v15_provenance.jsonl","specialized_new_40000_v16_provenance.jsonl"),
 ("pilot_2000_v15_generation_report.json","specialized_new_40000_v16_generation_report.json"),
 ('sem_v15_{num:06d}','sem_v16_{num:06d}'),
]:
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
extra=r'''
src=src.replace("N=2000","N=40000")
# Shorten only redundant modifiers; target reference and anchors remain fixed.
custom=r"""
_base_target=base.target
def _target_shorter(mode, excluded=()):
    spec,phrase,used=_base_target(mode,excluded)
    for a,b in (
      (" trong các địa điểm cùng loại, tính theo đường chim bay", " trong nhóm cùng loại"),
      (" trong số các địa điểm cùng loại", " trong nhóm cùng loại"),
      (" so với các địa điểm cùng loại", " so với những nơi cùng loại"),
      (" trong các địa điểm cùng loại trên bản đồ", " trong nhóm cùng loại"),
      (" trong các địa điểm cùng loại", " trong nhóm cùng loại"),
      (" trên toàn bản đồ", " trên bản đồ"),
      (" của toàn khuôn viên", " trong trường"),
      (" theo khoảng cách ô lưới", " theo ô lưới"),
      (" tính theo số ô lưới", " theo ô lưới"),
    ):
        phrase=phrase.replace(a,b)
    return spec,phrase,used
base.target=_target_shorter
"""
assert src.count("base.rng=r")==1
src=src.replace("base.rng=r", "base.rng=r"+chr(10)+custom)
'''
old="s=s.replace(marker,addition+marker)"
assert s.count(old)==1
s=s.replace(old,"s=s.replace(marker,addition+"+repr(extra)+"+marker)")
env={"__name__":"new_full_40k_v16","__file__":str(P/"run_semantic_generator_v15.py")}
exec(compile(s,str(P/"run_semantic_generator_v15.py"),"exec"),env)
