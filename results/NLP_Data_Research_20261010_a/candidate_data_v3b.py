"""Candidate V3: balance 'before' marker and add active-via despite canceled old stop."""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,random,sys
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS
WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
spec=importlib.util.spec_from_file_location("candidate_v2_library",WORK/"candidate_data_v2.py")
V=importlib.util.module_from_spec(spec);sys.modules[spec.name]=V;spec.loader.exec_module(V)
V.NEG[1]="Robot từng được yêu cầu ghé {via} lấy chứng từ trước khi đi giao, nay lệnh đã bị hủy. Đi thẳng để giao."
rows=V.generate(n_pairs=1000,seed=2026101002)
rng=random.Random(2026101003)
c=collections.Counter()
for i in range(0,len(rows),2):
    pos,neg=rows[i],rows[i+1]
    assert pos["via"] is not None and neg["via"] is None
    assert pos["goal"]==neg["goal"] and pos["urgent"]==neg["urgent"] and pos["fragile"]==neg["fragile"]
    # 50% of positive cases now also contain a CANCELED former pickup
    # while still requiring the active new pickup; decouple 'hủy' from via=None.
    if (i//2)%2==0:
        current_goal=pos["goal"]["type"];current_via=pos["via"]["type"]
        other=rng.choice([k for k in V.ALIASES if k not in (current_goal,current_via)])
        old=rng.choice(V.ALIASES[other])
        prefix=f"Chặng lấy hàng trước đây tại {old} đã hủy; không ghé chỗ cũ nữa. "
        prefix=prefix if pos["accented"] else fold(prefix)
        candidate=prefix+pos["text"]
        if len(tokenize(candidate))<=MAX_TOKENS:
            pos["text"]=candidate
            pos["provenance"]["old_pickup_cancelled_type"]=other
            c["active_with_cancelled_old"]+=1
    if "truoc" in set(fold(pos["text"]).split()):
        c["pos_has_truoc"]+=1
    if "truoc" in set(fold(neg["text"]).split()):
        c["neg_has_truoc"]+=1
seen=set()
for r in rows:
    t=fold(r["text"]).strip()
    assert t not in seen,"Duplicate produced"
    seen.add(t)
    assert len(tokenize(r["text"]))<=MAX_TOKENS
    yes=r["via"] is not None
    c["total"]+=1;c["via"]+=yes
    w=set(t.split())
    for marker in ("lay","nhan","huy","khong","truoc"):
        if marker in w:
            c[f"with_{marker}"]+=1;c[f"via_with_{marker}"]+=yes
p=WORK/"experiments/counterfactual_candidate_v3.jsonl"
with p.open("x",encoding="utf-8") as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
report={"experiment":"candidate_counterfactual_v3","date_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pairs":1000,"examples":len(rows),"via_rate":c["via"]/len(rows),
        "active_via_with_cancelled_old":c["active_with_cancelled_old"],
        "p_via_given_marker":{a:round(c["via_with_"+a]/c["with_"+a],4) if c["with_"+a] else None
                              for a in ("lay","nhan","huy","khong","truoc")},
        "source":"candidate_data_v2.py read-only, evolved in memory only",
        "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
        "source_notes":"Label contrast + explicit canceled former pickup while genuine current pickup stays active.",
        "limitations":"Semantically reviewed by programmatic invariants, not human; named goals only; no active-run change."}
out=WORK/"checkpoints/counterfactual_candidate_v3_audit.json"
with out.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("CANDIDATE_V3_COMPLETE",out,json.dumps(report,ensure_ascii=False),flush=True)
