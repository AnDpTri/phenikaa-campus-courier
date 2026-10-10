"""Independent semantic contract and stratified review for V10, existing files read-only.
Writes only new audit and selection outputs, atomically create-only.
"""
from pathlib import Path
from collections import Counter,defaultdict
import importlib.util,random,json,re,hashlib
from courier.nlp.text import fold,tokenize
from courier.nlp.parser import TargetSpec
from courier.nlp.neural import spec_labels
from courier.nlp.synth import GOAL_MODES,VIA_MODES
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
GEN=P/"typed_semantic_generator.py"
DATA=P/"pilot_v10_2000.jsonl"
OLD=Path(r"D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review\incoming_40000.jsonl")
OUT=P/"semantic_contract_v10.json"
SAMPLE=P/"semantic_stratified_v10_180.jsonl"
for target in (OUT,SAMPLE):
 if target.exists():raise RuntimeError("Refuse overwrite "+str(target))
spec=importlib.util.spec_from_file_location("v8_semantic_generator_readonly",GEN)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ITEMS=set(module.CARGO_FRAGILE+module.CARGO_DURABLE)
FRAG=set(module.CARGO_FRAGILE);NORMAL=set(module.CARGO_DURABLE)
PLACES=module.PLACE
rows=[json.loads(s) for s in DATA.open(encoding="utf-8")]
orig=[json.loads(s) for s in OLD.open(encoding="utf-8")]
assert len(rows)==2000
issues=defaultdict(list);stats=Counter()
regex={
"double_noun":r"\b(ve dia diem|dia diem diem|dia diem vi tri|tai trong|den trong|o o|giao kien the)\b",
"temporal_nominal_mismatch":r"\b(ke do moi|sau do can|tiep theo|roi ket thuc bang viec)\s+(dia chi nhan|noi nhan|diem ket thuc|nguoi nhan cho|don nay giao)\b",
"awkward_buu_kien":r"\bbuu kien (tap|hop|bo|tui|xap|lo|khay|cuon|chong)\b",
"holdout_frame":r"\blay\b.{0,90}\bo\b.{0,90}\bxong thi\b",
"via_explicit":r"\btruoc\b",
"no_via_take":r"\blay\b",
"correction":r"\b(ban cu|ban nhap|huy|don khac|chuyen truoc|phieu cu|ghi chu cu|lenh moi|thay doi)\b",
"distractor":r"\b(don khac|chuyen truoc|don truoc|ghi chu cu|bo qua|khong thuoc|khong phai diem giao|khong lay hay giao)\b",
"spatial":r"\b(gan|xa|phia|cuc|duong chim bay|o luoi)\b",
}
rg={k:re.compile(v) for k,v in regex.items()}
def log_issue(name,i,desc):
 if len(issues[name])<30:issues[name].append({"id":i,"detail":desc})
 stats["error_"+name]+=1
anchors=0
for i,row in enumerate(rows):
 text=row["text"];t=fold(text);g=row["goal"];v=row["via"]
 stats["n"]+=1
 stats["via_present"]+=v is not None
 stats["accented"]+=any(ord(x)>127 for x in text)
 for name,p in rg.items():stats[name]+=bool(p.search(t))
 if v is not None and "truoc" not in t.split():stats["via_without_before"]+=1
 if v is None and "lay" in t.split():stats["none_with_pickup_verb"]+=1
 if g["ref"]=="anchor_near":
  anchors+=1
  if not any(s in t for s in ("khong tinh chinh","ngoai chinh","tru chinh","khong tinh moc")):
   log_issue("anchor_exclusion_missing",row["id"],t[:180])
 # Goal and via mode schemas, specificity of anchored modes
 goal=TargetSpec(g["type"],g["ref"],g["anchor"])
 via=None if v is None else TargetSpec(v["type"],v["ref"],v["anchor"])
 try:spec_labels(goal,via,row["urgent"],row["fragile"])
 except Exception as exc:log_issue("bad_schema",row["id"],str(exc))
 if (g["ref"] or "named") not in GOAL_MODES:log_issue("bad_goal_ref",row["id"],str(g))
 if v is not None and (v["ref"] or "named") not in VIA_MODES:log_issue("bad_via_ref",row["id"],str(v))
 if not 14<=len(tokenize(text))<=160:log_issue("bad_token_length",row["id"],len(tokenize(text)))
 used_cargo=[x for x in ITEMS if fold(x) in t]
 if len(used_cargo)!=1:log_issue("cargo_cardinality",row["id"],used_cargo)
 if len(used_cargo)==1:
  cargo=used_cargo[0]
  if (cargo in FRAG)!=row["fragile"]:log_issue("fragility_item_mismatch",row["id"],cargo)
 if g["type"]:
  if not any(fold(q) in t for q in PLACES[g["type"]]):
   log_issue("goal_missing_place",row["id"],str(g))
 if v and v["type"]:
  if not any(fold(q) in t for q in PLACES[v["type"]]):
   log_issue("via_missing_place",row["id"],str(v))
 if g["ref"] in ("near","far","anchor_near") and not g["anchor"]:log_issue("goal_missing_anchor",row["id"],str(g))
 if v is not None and v["ref"] in ("near","far") and not v["anchor"]:log_issue("via_missing_anchor",row["id"],str(v))
 if row["urgent"] and not any(fold(k) in t for k in ("giao gấp","ưu tiên","giao khẩn","hỏa tốc","hoàn tất ngay","xử lý sớm")):
  log_issue("urgent_true_missing",row["id"],t[-140:])
 if row["fragile"] and not any(fold(k) in t for k in ("dễ vỡ","dễ hỏng","nhạy va chạm","dễ bể")):
  log_issue("fragile_true_missing",row["id"],t[-140:])
 for k in ("double_noun","temporal_nominal_mismatch","awkward_buu_kien","holdout_frame"):
  if rg[k].search(t):log_issue(k,row["id"],t[:220])
# Explicitly sample varied strata without conditioning on outcome.
pred={
"goal_anchor_near":lambda r:r["goal"]["ref"]=="anchor_near",
"goal_near_far":lambda r:r["goal"]["ref"] in ("near","far"),
"goal_most":lambda r:r["goal"]["ref"] and r["goal"]["ref"].endswith("_most"),
"goal_direction":lambda r:r["goal"]["ref"] in ("north","south","east","west"),
"via_near_far":lambda r:r["via"] is not None and r["via"]["ref"] in ("near","far"),
"via_direction":lambda r:r["via"] is not None and r["via"]["ref"] in ("north","south","east","west"),
"via_named_no_before":lambda r:r["via"] is not None and r["via"]["ref"] is None and "truoc" not in fold(r["text"]).split(),
"via_none_take":lambda r:r["via"] is None and "lay" in fold(r["text"]).split(),
"both_flags":lambda r:r["urgent"] and r["fragile"],
"neither_flag":lambda r:not r["urgent"] and not r["fragile"],
"past_correction":lambda r:bool(rg["correction"].search(fold(r["text"]))),
"accents":lambda r:any(ord(c)>127 for c in r["text"]),
"no_accents":lambda r:all(ord(c)<=127 for c in r["text"]),
}
rng=random.Random(202610092031)
picked={};group_sizes={}
for group,fun in pred.items():
 candidates=[i for i,row in enumerate(rows) if fun(row)];group_sizes[group]=len(candidates)
 for j in rng.sample(candidates,min(12,len(candidates))):
  if j not in picked:picked[j]=group
# supplement from random
remaining=[i for i in range(len(rows)) if i not in picked]
for j in rng.sample(remaining, max(0,180-len(picked))):
 picked[j]="random"
stats["audit_sample_n"]=len(picked)
with SAMPLE.open("x",encoding="utf-8",newline="\n") as f:
 for j,group in sorted(picked.items()):
  f.write(json.dumps({"source_line":j+1,"stratum":group,**rows[j]},ensure_ascii=False)+"\n")
lengths=sorted(len(tokenize(x["text"])) for x in rows)
orig_lengths=sorted(len(tokenize(x["text"])) for x in orig)
report={
"status":"PROGRAMMATIC_CONTRACT_AUDIT_PENDING_SEMANTIC_REVIEW",
"population":len(rows),"quality_ref_rows":len(orig),
"counts":dict(stats),"issues":dict(issues),
"token_p50":lengths[1000],"token_p90":lengths[1800],
"original_p50":orig_lengths[20000],"original_p90":orig_lengths[36000],
"strata_counts":group_sizes,
"pointers":{"data":str(DATA),"sample":str(SAMPLE)},
"spec":"Schema + item fragility + map references + known grammar patterns checked, independent semantic judgement pending."
}
with OUT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("REPORT",str(OUT),flush=True)
print("SAMPLE",str(SAMPLE),flush=True)
print("SUMMARY",json.dumps({k:v for k,v in report.items() if k not in ("issues","strata_counts")},ensure_ascii=True),flush=True)
print("ISSUES",json.dumps({k:{"count":stats["error_"+k],"examples":v[:2]} for k,v in issues.items()},ensure_ascii=True),flush=True)
