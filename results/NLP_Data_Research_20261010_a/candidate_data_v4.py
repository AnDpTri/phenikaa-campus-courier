"""Candidate V4 lexical-decorrelation via matched active/canceled old pickup clauses.
No modification of active train; output only within research workspace.
"""
import collections,datetime,hashlib,importlib.util,json,random,re,sys
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS
WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
spec=importlib.util.spec_from_file_location("candidate_v2_base_for_v4",WORK/"candidate_data_v2.py")
V=importlib.util.module_from_spec(spec);sys.modules[spec.name]=V;spec.loader.exec_module(V)
V.NEG[1]="Robot từng được yêu cầu ghé {via} lấy chứng từ trước khi đi giao, nay lệnh đã bị hủy. Đi thẳng để giao."
rows=V.generate(n_pairs=1000,seed=2026101002)
rng=random.Random(10111002)
skipped=0;ctr=collections.Counter();seen=set()
for i in range(0,len(rows),2):
    pos,neg=rows[i],rows[i+1]
    assert pos["via"] is not None and neg["via"] is None
    assert pos["pair_id"]==neg["pair_id"] and pos["goal"]==neg["goal"]
    excluded=(pos["goal"]["type"],pos["via"]["type"])
    old_type=rng.choice([name for name in V.ALIASES if name not in excluded])
    old_loc=rng.choice(V.ALIASES[old_type])
    pref=f"Đã hủy bước lấy ở {old_loc}. "
    if not pos["accented"]:pref=fold(pref)
    new_p=pref+pos["text"];new_n=pref+neg["text"]
    if max(len(tokenize(new_p)),len(tokenize(new_n)))>MAX_TOKENS:
        # Keep both unchanged so pair identity and label invariance hold.
        skipped+=1
    else:
        pos["text"]=new_p;neg["text"]=new_n
        for a in (pos,neg):a["provenance"]["cancelled_old_pickup_type"]=old_type
    for row in (pos,neg):
        t=fold(row["text"]);assert t not in seen;seen.add(t)
        is_via=row["via"] is not None
        ctr["rows"]+=1;ctr["via"]+=is_via
        markers=set(re.findall(r"[a-z0-9]+",t))
        for m in ("lay","nhan","huy","khong","truoc"):
            if m in markers:
                ctr["contains_"+m]+=1
                ctr["via_contains_"+m]+=is_via
report={"experiment":"counterfactual_semantic_balancing_candidate_v4",
    "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "rows":len(rows),"pairs":len(rows)//2,
    "via_rate":round(ctr["via"]/ctr["rows"],4),
    "pairs_skipped_common_canceled_old_prefix_due_length":skipped,
    "p_via_given_marker":{m:round(ctr["via_contains_"+m]/ctr["contains_"+m],4)
        if ctr["contains_"+m] else None for m in ("lay","nhan","huy","khong","truoc")},
    "all_pairs_goal_and_flags_equal":True,"all_rows_unique_folded":True,
    "qa":"Programmatically label-checked in v2 generator. No human linguistic audit, spatial targets not addressed.",
    "active_training_modified":False}
p=WORK/"experiments/counterfactual_candidate_v4.jsonl"
with p.open("x",encoding="utf-8") as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
report["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
o=WORK/"checkpoints/counterfactual_candidate_v4_audit.json"
with o.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("CANDIDATE_V4_COMPLETE",o,json.dumps(report,ensure_ascii=False),flush=True)
