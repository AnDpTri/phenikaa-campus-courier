"""Independent semantic role audit for V13 2k. Pure read of inputs; new report files only."""
from pathlib import Path
import json,re,random,collections,hashlib,statistics
from courier.nlp.text import fold,tokenize
from courier.nlp.parser import TargetSpec
from courier.nlp.neural import spec_labels,MAX_TOKENS
from courier.nlp.synth import GOAL_MODES,VIA_MODES
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
DATA=P/"pilot_2000_v13_role_semantics.jsonl"
PROV=P/"pilot_2000_v13_provenance.jsonl"
OUT=P/"independent_semantic_audit_v13_2000.json"
SAMPLES=P/"stratified_semantic_review_v13.jsonl"
for f in (OUT,SAMPLES):
 if f.exists():raise RuntimeError("Never overwrite existing: "+str(f))
rows=[json.loads(x) for x in DATA.open(encoding="utf-8")]
meta=[json.loads(x) for x in PROV.open(encoding="utf-8")]
assert len(rows)==len(meta)==2000
counts=collections.Counter()
errors=collections.defaultdict(list)
fn_regex={
 "bad_temporal_nominal":r"\b(ke do moi|sau do can|roi ket thuc bang viec)\s+(dia chi|noi nhan|diem ket thuc)\b",
 "double_nominal":r"\b(dia diem diem|ve dia diem|dia diem vi tri|buu kien xap)\b",
 "heldout_frame":r"\blay\b.{0,90}\bo\b.{0,90}\bxong thi\b",
}
def bad(kind,row,detail):
 counts["error_"+kind]+=1
 if len(errors[kind])<12:errors[kind].append({"id":row["id"],"detail":detail})
def contains_cue(text,seq):return any(fold(p) in fold(text) for p in seq)
urgent_yes=("giao ngay","hỏa tốc","khẩn cấp","giao gấp")
fragile_yes=("dễ vỡ","dễ bể","dễ hỏng","nhạy va đập")
for row,info in zip(rows,meta):
 if row["id"]!=info["id"]:bad("id_mismatch",row,row["id"]+" "+info["id"]);continue
 text=row["text"];norm=fold(text);g=row["goal"];v=row["via"];counts["n"]+=1
 for name,pattern in fn_regex.items():
  if re.search(pattern,norm):bad(name,row,pattern)
 gt=TargetSpec(g["type"],g["ref"],g["anchor"])
 vt=None if v is None else TargetSpec(v["type"],v["ref"],v["anchor"])
 try:
  assert (gt.ref or "named") in GOAL_MODES
  assert vt is None or (vt.ref or "named") in VIA_MODES
  assert len(spec_labels(gt,vt,row["urgent"],row["fragile"]))==8
 except Exception as ex:bad("bad_target_spec",row,str(ex))
 length=len(tokenize(text))
 if length>MAX_TOKENS:bad("truncated",row,str(length))
 if length<14:bad("too_short",row,str(length))
 if row["urgent"] and not contains_cue(text,urgent_yes):bad("urgent_positive_missing_cue",row,text[-170:])
 if row["fragile"] and not contains_cue(text,fragile_yes):bad("fragile_positive_missing_cue",row,text[-170:])
 if info["parcel"] not in text and fold(info["parcel"]) not in norm:bad("parcel_missing",row,info["parcel"])
 if info["supplemental_pickup_item"]:
  counts["supplemental_item"]+=1
  if v is None:bad("extra_item_but_no_via",row,info["supplemental_pickup_item"])
  if fold(info["supplemental_pickup_item"]) not in norm:bad("missing_extra_item",row,info["supplemental_pickup_item"])
 fg=info["former_destination"]
 if fg:
  counts["former_goal"]+=1
  if fold(fg) not in norm:bad("former_goal_not_stated",row,fg)
  if not any(p in norm for p in ("huy","doi","khong con hieu luc","khong duoc giao","khong dung diem ay","bo diem cu","bo dia chi cu","khong dung diem","khong dung dia chi","khong dung dia chi","nham roi")):
   # Includes 'nham roi' under natural language but must be explicit revocation.
   bad("former_goal_without_revocation",row,text[:180])
  if g["type"] is None:bad("former_goal_map_only",row,str(g))
  if v is not None:counts["former_goal_and_via"]+=1
 op=info["former_pickup"]
 if op:
  counts["former_pickup"]+=1
  if fold(op) not in norm:bad("former_pickup_missing",row,op)
  if not any(x in norm for x in ("huy","bo","khong con","rut lai","sai")):bad("former_pickup_revocation_missing",row,text[:180])
 for n in info["negative_mentions"]:
  counts["negative_mentions"]+=1
  if fold(n["lexeme"]) not in norm:bad("negative_lexeme_missing",row,n["lexeme"])
  if g["type"] is None and n["role"] not in ("historical","not_related"):
   bad("maponly_negative_conflict",row,n["role"])
  if n["type"] in (g["type"],g["anchor"],None if v is None else v["type"],None if v is None else v["anchor"]):
   bad("negative_same_target_type",row,str(n))
 if v is not None and "truoc" not in norm.split():counts["via_without_before"]+=1
 if v is None and bool(re.search(r"\blay\b",norm)):counts["no_via_with_take"]+=1
 if row["fragile"] and not info["parcel"] in ("gói hàng","thùng bưu phẩm","túi đồ","kiện hàng","hộp hàng","bưu kiện","hộp carton","túi chuyển phát","phong bì niêm phong","gói vật tư"):
  counts["fragile_named_physical"]+=1
# Stratification
groups={
 "revised_goal_with_via":lambda x,m:m["former_destination"] is not None and x["via"] is not None,
 "revised_goal_no_via":lambda x,m:m["former_destination"] is not None and x["via"] is None,
 "revised_pickup_has_via":lambda x,m:m["former_pickup"] is not None and x["via"] is not None,
 "revised_pickup_no_via":lambda x,m:m["former_pickup"] is not None and x["via"] is None,
 "goal_anchor_near":lambda x,m:x["goal"]["ref"]=="anchor_near",
 "goal_global_extreme":lambda x,m:bool(x["goal"]["ref"] and x["goal"]["ref"].endswith("_most")),
 "goal_near_far":lambda x,m:x["goal"]["ref"] in ("near","far"),
 "via_spatial":lambda x,m:x["via"] is not None and x["via"]["ref"] in ("near","far","north","south","east","west"),
 "supplementary_pickup":lambda x,m:m["supplemental_pickup_item"] is not None,
 "multiple_negatives":lambda x,m:len(m["negative_mentions"])==2,
 "negative_no_via":lambda x,m:x["via"] is None and bool(m["negative_mentions"]),
 "urgent_fragile_both":lambda x,m:x["urgent"] and x["fragile"],
 "neither_flags":lambda x,m:not x["urgent"] and not x["fragile"],
}
rng=random.Random(131313);picks={};group_count={}
for name,pred in groups.items():
 candidates=[i for i,(x,m) in enumerate(zip(rows,meta)) if pred(x,m)]
 group_count[name]=len(candidates)
 for i in rng.sample(candidates,min(9,len(candidates))):picks.setdefault(i,name)
remain=[i for i in range(len(rows)) if i not in picks]
for i in rng.sample(remain,max(0,140-len(picks))):picks[i]="random"
with SAMPLES.open("x",encoding="utf-8") as f:
 for i,group in sorted(picks.items()):
  f.write(json.dumps({"source_line":i+1,"review_group":group,"provenance":meta[i],**rows[i]},ensure_ascii=False)+"\n")
report={"source":"pilot_2000_v13_role_semantics.jsonl","status":"AUTOMATED_ROLE_CONTRACT_AUDIT_DONE; MANUAL_SEMANTIC_AUDIT_PENDING",
        "rows":len(rows),"role_counts":dict(counts),
        "issues":dict(errors),"issue_families":{k:counts["error_"+k] for k in errors},
        "sample_size":len(picks),"group_population":group_count,
        "limitations":["Rule-based consistency checks do not prove naturalness or all labels","Only 2k pilot has been assessed, not a new 40k","Dataset ZIP text validation was not incorporated into templates"]}
with OUT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("SUMMARY",json.dumps({"n":len(rows),"counts":dict(counts),"issues":{k:v[:3] for k,v in errors.items()},"sample_size":len(picks)},ensure_ascii=True),flush=True)
print("AUDIT",OUT,"SELECTED",SAMPLES,flush=True)
