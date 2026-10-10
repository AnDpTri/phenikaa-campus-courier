"""Independent full-population audit of SINGLE weighted 200K corpus (never modifies generator/data)."""
import json,hashlib,re,collections,random,sys,importlib.util,time
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.parser import TargetSpec
from courier.nlp.neural import spec_labels,MAX_TOKENS,encode_texts
from courier.nlp.synth import GOAL_MODES,VIA_MODES
ROOT=Path(r"D:\phenikaa")
P=ROOT/"results/NLP_V4_R2/unified_weighted_200k_20261010_a"
DATA=P/"train_weighted_unified_200000_seed2026101060.jsonl"
ROLES=P/"train_weighted_unified_200000_roles_seed2026101060.jsonl"
MANIFEST=P/"train_weighted_unified_200000_manifest.json"
OUT=P/"independent_weighted_200000_quality_report.json"
SAMPLES=P/"independent_weighted_200000_semantic_strata_280.jsonl"
for f in (OUT,SAMPLES):
 if f.exists():raise RuntimeError("Refuse overwrite "+str(f))
manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
sp=importlib.util.spec_from_file_location("qa_target_weighted_v4",P/"synth_weighted_v4.py")
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
counts=collections.Counter();recipe=collections.Counter();goal=collections.Counter();via=collections.Counter()
violations=collections.Counter();issue_ex=collections.defaultdict(list);pairs=[];seen=set()
first8=collections.Counter();lens=[]
rg_bad=re.compile(r"\b(khu vuc khu|dia diem diem|dia diem vi tri|buu kien buu kien|ngoai kien buu kien|ke do moi dia chi|sau do can dia chi|tai tai|o o)\b")
def issue(key,row,details):
 violations[key]+=1
 if len(issue_ex[key])<6:issue_ex[key].append({"id":row["id"],"detail":str(details)[:190]})
with DATA.open(encoding="utf-8") as f, ROLES.open(encoding="utf-8") as rolef:
 for i,(line,rline) in enumerate(zip(f,rolef),1):
  j=json.loads(line);p=json.loads(rline);txt=j["text"];t=fold(txt).strip()
  if j["id"]!=f"weighted_200k_{i:06d}" or p["id"]!=j["id"]:issue("bad_id",j,(i,p["id"]))
  if set(j)!={"id","text","goal","via","urgent","fragile"}:issue("unexpected_fields",j,list(j))
  if not isinstance(j["urgent"],bool) or not isinstance(j["fragile"],bool):issue("bad_boolean",j,(j["urgent"],j["fragile"]))
  g=j["goal"];v=j["via"]
  try:
   gg=TargetSpec(g["type"],g["ref"],g["anchor"])
   vv=None if v is None else TargetSpec(v["type"],v["ref"],v["anchor"])
   assert (gg.ref or "named") in GOAL_MODES
   assert vv is None or (vv.ref or "named") in VIA_MODES
   assert len(spec_labels(gg,vv,j["urgent"],j["fragile"]))==8
  except Exception as ex:issue("invalid_labels",j,str(ex))
  length=len(tokenize(txt));lens.append(length)
  if not (8<=length<=MAX_TOKENS):issue("token_range",j,length)
  if rg_bad.search(t):issue("known_grammar",j,t[:120])
  if t in seen:issue("duplicate",j,t[:120])
  seen.add(t);first8[" ".join(t.split()[:8])]+=1
  r=p["recipe"];recipe[r]+=1
  if r not in m.WEIGHTED_RECIPES:issue("invalid_recipe",j,r)
  if r=="v2_general" and p["mode"]!="v2":issue("legacy_provenance",j,p["mode"])
  if r!="v2_general":
   if p["mode"]!="hard":issue("role_provenance",j,p["mode"])
   if p["goal_mode"]!=(g["ref"] or "named"):issue("goal_mode_mismatch",j,p["goal_mode"])
   if p["via_mode"]!=("none" if v is None else (v["ref"] or "named")):issue("via_mode_mismatch",j,p["via_mode"])
   if p["former_destination"]:
    counts["revised_goal"]+=1
    if fold(p["former_destination"]) not in t:issue("former_goal_missing",j,p["former_destination"])
   if p["former_pickup"]:
    counts["revised_pickup"]+=1
    if fold(p["former_pickup"]) not in t:issue("former_pickup_missing",j,p["former_pickup"])
   if p["former_destination"] and v is not None:counts["revised_goal_via"]+=1
   for neg in p["negative_roles"]:
    counts["negative_mentions"]+=1
    if fold(neg["text"]) not in t:issue("negative_lexeme",j,neg)
    if neg["type"] in (g["type"],g["anchor"],None if v is None else v["type"],None if v is None else v["anchor"]):
     issue("negative_overlaps_positive",j,neg)
   if j["urgent"] and not any(fold(x) in t for x in m.URGENT_POS):issue("urgent_true_ungrounded",j,t[-180:])
   if j["fragile"] and not any(fold(x) in t for x in m.FRAGILE_POS):issue("fragile_true_ungrounded",j,t[-180:])
   if g["ref"]=="anchor_near" and not any(x in t for x in ("khong tinh","loai tru","ngoai","khong ke","tru moc")):issue("anchor_near_exclusion",j,t)
   if r=="pickup_reasoning" and (v is None and "lay" not in t.split() or v is not None and "truoc" in t.split()):
    issue("pickup_recipe_violation",j,r)
   if r=="spatial_grounding" and not (g["ref"] is not None or v is not None and v["ref"] is not None):
    issue("spatial_recipe_violation",j,r)
   if r=="instruction_revision" and not (p["former_destination"] or p["former_pickup"]):
    issue("revision_recipe_violation",j,r)
   if r=="distractor_negation" and not p["negative_roles"]:issue("negation_recipe_violation",j,r)
  counts["n"]+=1;counts["via"]+=v is not None
  counts["goal_spatial"]+=g["ref"] is not None
  counts["via_spatial"]+=v is not None and v["ref"] is not None
  counts["flag_urgent"]+=j["urgent"];counts["flag_fragile"]+=j["fragile"]
  counts["accented"]+=any(ord(x)>127 for x in txt)
  counts["via_without_before"]+=v is not None and not re.search(r"\btruoc\b",t)
  counts["no_via_with_lay"]+=v is None and bool(re.search(r"\blay\b",t))
  goal[g["ref"] or "named"]+=1
  via["none" if v is None else (v["ref"] or "named")]+=1
  if r!="v2_general":pairs.append((i,j,p))
  if i%50000==0:print("AUDIT_CHECKED",i,flush=True)
 if f.readline() or rolef.readline():raise RuntimeError("Unexpected extra data or role lines")
assert counts["n"]==200000
for key,v in m.WEIGHTED_RECIPES.items():
 if recipe[key]!=v*2000:issue("wrong_recipe_quota",{"id":"quota"},(key,recipe[key],v*2000))
if max(first8.values())>35:issue("excess_prefix_frequency",{"id":"prefix"},max(first8.values()))
lens.sort();N=len(lens)
# Reproducible, seeded semantic-stratified selection from entire mixed corpus.
rng=random.Random(2026101063);sub=collections.defaultdict(list)
for pos,row,p in pairs:
 if p["former_destination"] and row["via"] is not None:sub["revised_goal_via"].append((pos,row,p))
 if p["former_pickup"] and row["via"] is None:sub["removed_pickup_none"].append((pos,row,p))
 if p["former_pickup"] and row["via"] is not None:sub["removed_pickup_real_via"].append((pos,row,p))
 if row["goal"]["ref"]=="anchor_near":sub["anchor_near"].append((pos,row,p))
 if str(row["goal"]["ref"]).endswith("_most"):sub["global_extreme"].append((pos,row,p))
 if row["goal"]["ref"] in ("near","far"):sub["relative_goal"].append((pos,row,p))
 if row["via"] is not None and row["via"]["ref"] in ("near","far","north","south","east","west"):sub["spatial_via"].append((pos,row,p))
 if row["via"] is None and "lay" in fold(row["text"]).split():sub["no_via_with_lay"].append((pos,row,p))
 if row["via"] is not None and "truoc" not in fold(row["text"]).split():sub["via_without_before"].append((pos,row,p))
 if p["negative_roles"]:sub["negative_context"].append((pos,row,p))
 if row["urgent"] and row["fragile"]:sub["both_flags_true"].append((pos,row,p))
 if not row["urgent"] and not row["fragile"]:sub["both_flags_false"].append((pos,row,p))
selected={};n_groups={}
for name,group in sub.items():
 n_groups[name]=len(group)
 for item in rng.sample(group,min(15,len(group))):selected.setdefault(item[0],(name,*item[1:]))
for item in rng.sample(pairs, min(len(pairs), 280)):
 selected.setdefault(item[0],("random",*item[1:]))
chosen=sorted(selected.items())[:320]
with SAMPLES.open("x",encoding="utf-8",newline="\n") as f:
 for pos,(group,row,p) in chosen:
  f.write(json.dumps({"dataset_line":pos,"stratum":group,"provenance":p,**row},ensure_ascii=False)+"\n")
# Backward-compatible original V2 mode seed and holdout.
import courier.nlp.synth as original
def key(x):return(x.text,x.goal,x.via,x.urgent,x.fragile)
for flag in (False,True):
 if list(map(key,original.generate(256,918,flag)))!=list(map(key,m.generate(256,918,flag,mode="v2"))):
  issue("v2_compatibility",{"id":"legacy"},flag)
# Actual model encoding and trainer loader smoke.
spec=importlib.util.spec_from_file_location("trainer_readonly",ROOT/"scripts/nlp/train_neural_parser.py")
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
trainrows=trainer.labelled_prebuilt(DATA)
trainlabels=trainer.tensors(trainrows)
encoded=encode_texts([x[0] for x in trainrows[:256]])
if len(trainrows)!=200000:issue("trainer_count",{"id":"model_loader"},len(trainrows))
report={
 "status":"AUTOMATED_QA_COMPLETE_PENDING_MANUAL_SEMANTIC_REVIEW",
 "rows_checked":200000,
 "counts":dict(counts),"recipe_counts":dict(recipe),"goal_modes":dict(goal),"via_modes":dict(via),
 "token_lengths":{"p50":lens[N//2],"p90":lens[int(.9*N)],"p99":lens[int(.99*N)],"max":lens[-1]},
 "prefix8_unique":len(first8),"prefix8_max_frequency":max(first8.values()),
 "sample_size":len(chosen),"stratum_populations":dict(n_groups),
 "model_loader":{"rows":len(trainrows),"heads":{k:len(v) for k,v in trainlabels.items()},"encoded_batch":list(encoded.shape)},
 "error_counts":dict(violations),"error_examples":dict(issue_ex),
 "file_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
 "limitations":"Full-row checks are deterministic contracts, not exhaustive human entailment or model accuracy validation."
}
with OUT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("INDEPENDENT_FINAL",json.dumps({k:report[k] for k in ("rows_checked","counts","recipe_counts","token_lengths","prefix8_unique","prefix8_max_frequency","sample_size","model_loader","error_counts")},ensure_ascii=True),flush=True)
