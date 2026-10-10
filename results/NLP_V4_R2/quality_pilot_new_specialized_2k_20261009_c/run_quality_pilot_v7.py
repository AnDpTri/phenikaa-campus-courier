"""Seventh pilot, revised after direct sentence-level review. No prior files changed."""
from pathlib import Path
D=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
src=(D/"run_quality_pilot_v6.py").read_text(encoding="utf-8")
changes=[
 ('202610096','202610097'),
 ('pilot_new_specialized_2000_v6.jsonl','pilot_new_specialized_2000_v7.jsonl'),
 ('pilot_quality_report_v6.json','pilot_quality_report_v7.json'),
 ('(\'f"{place} nằm xa {anchor} hơn chỗ thứ hai"\',\'f"{place} nằm xa {anchor} hơn điểm cùng loại thứ hai"\'),',
  '(\'f"{place} nằm xa {anchor} hơn chỗ thứ hai"\',\'f"{place} nằm xa {anchor} hơn điểm cùng loại thứ hai"\'),\\n (\'"người nhận đợi kiện {item} tại {goal}."\',\'"người nhận chờ {item} tại {goal}."\'),\\n (\'"Hàng phụ {second} được lấy ở {via}; việc giao hàng là {goal}"\',\'"Món hàng bổ sung ({second}) được nhận tại {via}; tiếp theo {goal}"\'),'),
]
for before,after in changes:
 assert src.count(before)==1,(before,src.count(before))
 src=src.replace(before,after)
ctx={"__name__":"run_quality_pilot_v7_inmemory","__file__":str(D/"run_quality_pilot_v6.py")}
exec(compile(src,str(D/"run_quality_pilot_v6.py"),"exec"),ctx)
