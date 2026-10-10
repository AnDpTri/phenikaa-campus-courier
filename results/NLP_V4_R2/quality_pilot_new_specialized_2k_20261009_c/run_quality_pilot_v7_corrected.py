"""Pilot v7 sentence-quality repair, in-memory source customization; old files untouched."""
from pathlib import Path
D=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
wrapper=(D/"run_quality_pilot_v6.py").read_text(encoding="utf-8")
for a,b in [
 ("202610096","202610097"),
 ("pilot_new_specialized_2000_v6.jsonl","pilot_new_specialized_2000_v7.jsonl"),
 ("pilot_quality_report_v6.json","pilot_quality_report_v7.json"),
]:
 assert wrapper.count(a)==1,(a,wrapper.count(a))
 wrapper=wrapper.replace(a,b)
anchor="]\nfor before,after in changes:"
extra=(
 ' (\'"người nhận đợi kiện {item} tại {goal}."\',\'"người nhận chờ {item} tại {goal}."\'),\n'
 ' (\'"Hàng phụ {second} được lấy ở {via}; việc giao hàng là {goal}"\',\'"Món hàng bổ sung ({second}) được nhận tại {via}; tiếp theo {goal}"\'),\n'
)
assert wrapper.count(anchor)==1
wrapper=wrapper.replace(anchor,extra+anchor)
ctx={"__name__":"run_quality_pilot_v7_inmemory","__file__":str(D/"run_quality_pilot_v6.py")}
exec(compile(wrapper,str(D/"run_quality_pilot_v6.py"),"exec"),ctx)
