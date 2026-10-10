"""INDEPENDENT exhaustive QA on every record of unified 240K output.

Read only training files; exclusive-write new QA report. No source edits, train, submissions.
"""
import sys,importlib.util,json,hashlib,re,collections,time,random
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import spec_labels,MAX_TOKENS
from courier.nlp.parser import TargetSpec
from courier.nlp.synth import GOAL_MODES,VIA_MODES
P=Path(r"D:\phenikaa\results\NLP_V4_R2\unified_v2_v18_clone_20261009_a")
V2=P/"train_v2_fresh_200000_seed2026100925.jsonl"
HARD=P/"train_specialized_fresh_40000_seed2026100926.jsonl"
BOTH=P/"train_unified_240000_interleaved_seed20261009.jsonl"
PROV=P/"train_specialized_roles_40000_seed2026100926.jsonl"
OUT=P/"independent_full_240000_quality_report.json"
SAMPLE=P/"independent_hard_semantic_stratified_240.jsonl"
for x in (OUT,SAMPLE):
 if x.exists():raise RuntimeError("Refuse overwrite "+str(x))
report={"type":"full-data QA with role-provenance cross-check, sample selection and model encoder smoke",
 "counts":collections.Counter(),"issues":collections.defaultdict(list),"modes":collections.Counter(),
 "goal_modes":collections.Counter(),"via_modes":collections.Counter(),"lengths":{}}
def error(code,row,details):
 report["counts"]["errors_"+code]+=1
 if len(report["issues"][code])<5:report["issues"][code].append({"id":row["id"],"detail":str(details)[:230]})
def enumerate_rows(fp):
 with fp.open(encoding="utf-8") as f:
  for line in f:yield json.loads(line)
def source_checks(fp,label,expected):
 c=collections.Counter();lens=[];seen=set();prefix=set()
 for i,row in enumerate(enumerate_rows(fp),1):
  c["n"]+=1
  if row["id"]!=f"fresh_{label}_{i:06d}":error("bad_id",row,i)
  if set(row.keys())!={"id","text","goal","via","urgent","fragile"}:error("bad_schema",row,row.keys())
  g=row["goal"];v=row["via"]
  try:
   gspec=TargetSpec(g["type"],g["ref"],g["anchor"])
   vspec=None if v is None else TargetSpec(v["type"],v["ref"],v["anchor"])
   assert (gspec.ref or "named") in GOAL_MODES
   assert vspec is None or (vspec.ref or "named") in VIA_MODES
   assert len(spec_labels(gspec,vspec,row["urgent"],row["fragile"]))==8
   assert isinstance(row["urgent"],bool) and isinstance(row["fragile"],bool)
  except Exception as exc:error("bad_labels",row,str(exc))
  t=fold(row["text"]);s=t.strip()
  n=len(tokenize(row["text"]));lens.append(n)
  if not 1<=n<=MAX_TOKENS:error("bad_token_length",row,n)
  if s in seen:error("duplicate_local",row,{"id":row["id"],"prefix":s[:90]})
  seen.add(s);prefix.add(" ".join(s.split()[:8]))
  c["via_present"]+=v is not None
  c["accented"]+=any(ord(ch)>127 for ch in row["text"])
  c["via_without_truoc"]+=v is not None and not re.search(r"\btruoc\b",t)
  c["none_with_lay"]+=v is None and bool(re.search(r"\blay\b",t))
  if re.search(r"\b(khu vuc khu|dia diem diem|buu kien buu kien|ngoai kien buu kien|ke do moi dia chi|sau do can dia chi)\b",t):
   error("known_bad_grammar",row,t[:200])
  if label=="hard":
   report["goal_modes"][g["ref"] or "named"]+=1
   report["via_modes"]["none" if v is None else (v["ref"] or "named")]+=1
  if i%60000==0:print("SOURCE_READ",label,i,flush=True)
 if c["n"]!=expected:raise ValueError(("wrong count",label,c["n"]))
 lens.sort()
 report["counts"].update({label+"_"+k:v for k,v in c.items()})
 report["lengths"][label]={"p50":lens[len(lens)//2],"p90":lens[int(len(lens)*.9)],"p99":lens[int(len(lens)*.99)],"max":lens[-1]}
 report["modes"][label+"_prefix8"]=len(prefix)
 return seen
t0=time.time()
v2_set=source_checks(V2,"v2",200000)
hard_set=source_checks(HARD,"hard",40000)
report["counts"]["cross_v2_hard_exact_overlap"]=len(v2_set&hard_set)
if v2_set&hard_set:error("cross_component_overlap",{"id":"cross"},"overlap")
# Verify combined interleaving (5 normal/1 hard) and that no record is omitted.
n=0;all_seen=set()
with BOTH.open(encoding="utf-8") as f:
 for line in f:
  row=json.loads(line);n+=1
  expected_mode="hard" if n%6==0 else "v2"
  if not row["id"].startswith(f"fresh_{expected_mode}_"):error("bad_interleave",row,n)
  all_seen.add(row["id"])
report["counts"]["combined_rows"]=n
if n!=240000 or len(all_seen)!=240000:raise ValueError(("combined count/unique id",n,len(all_seen)))
# Verify structural role metadata corresponds to individual hard examples.
pool=list(enumerate_rows(HARD))
provenance=list(enumerate_rows(PROV))
assert len(pool)==len(provenance)==40000
rng=random.Random(2026100945)
review=[];strata=collections.defaultdict(list)
for i,(row,meta) in enumerate(zip(pool,provenance)):
 t=fold(row["text"]);g=row["goal"];v=row["via"]
 if row["id"]!=meta["id"]:error("provenance_alignment",row,meta["id"])
 if meta["goal_mode"]!=(g["ref"] or "named"):error("goal_mode_alignment",row,meta["goal_mode"])
 if meta["via_mode"]!=("none" if v is None else (v["ref"] or "named")):error("via_mode_alignment",row,meta["via_mode"])
 if fold(meta["item"]) not in t:error("main_cargo_missing",row,meta["item"])
 if v is not None and fold(meta["pickup_item"]) not in t:error("via_cargo_missing",row,meta["pickup_item"])
 if meta["former_destination"]:
  report["counts"]["hard_former_goal"]+=1
  if fold(meta["former_destination"]) not in t:error("cancel_goal_unmentioned",row,meta["former_destination"])
  if v is not None:report["counts"]["hard_former_goal_with_via"]+=1
 if meta["former_pickup"]:
  report["counts"]["hard_former_pickup"]+=1
  if fold(meta["former_pickup"]) not in t:error("cancel_via_unmentioned",row,meta["former_pickup"])
 used={g["type"],g["anchor"]}
 if v is not None:used.update([v["type"],v["anchor"]])
 for nm in meta["negative_roles"]:
  report["counts"]["negative_role_mentions"]+=1
  if nm["type"] in used:error("negative_overlap",row,nm)
  if fold(nm["text"]) not in t:error("negative_lexeme_missing",row,nm)
 if g["ref"]=="anchor_near":
  if not any(y in t for y in ("khong tinh","loai tru","khong ke","ngoai","tru moc")):
   error("ambiguous_anchor_near",row,t[:190])
 strata["random"].append(i)
 if meta["former_destination"] and v is not None:strata["goal_revised_via"].append(i)
 if meta["former_pickup"] and v is None:strata["canceled_via_none"].append(i)
 if meta["former_pickup"] and v is not None:strata["canceled_via_active"].append(i)
 if g["ref"]=="anchor_near":strata["anchor_near"].append(i)
 if str(g["ref"]).endswith("_most"):strata["extreme"].append(i)
 if v is not None and v["ref"] in ("near","far","north","south","east","west"):strata["spatial_via"].append(i)
 if v is None and re.search(r"\blay\b",t):strata["none_lay"].append(i)
 if v is not None and not re.search(r"\btruoc\b",t):strata["via_without_before"].append(i)
 if row["urgent"] and row["fragile"]:strata["both_flags"].append(i)
 if i%10000==9999:print("ROLE_CHECKED",i+1,flush=True)
# Stratified 240 candidate cases incl random: no model test data included.
selected={}
for group,indices in strata.items():
 if group=="random":continue
 for i in rng.sample(indices,min(24,len(indices))):selected.setdefault(i,group)
rest=[i for i in range(len(pool)) if i not in selected]
for i in rng.sample(rest,max(0,240-len(selected))):selected[i]="random"
with SAMPLE.open("x",encoding="utf-8",newline="\n") as f:
 for i,group in sorted(selected.items()):
  f.write(json.dumps({"sample_row":i+1,"group":group,"roles":provenance[i],**pool[i]},ensure_ascii=False)+"\n")
# Train versus V2/hard HOLDOUT holdout=True. Never use this holdout in train.
s=importlib.util.spec_from_file_location("unified_qa_holdout",P/"synth_unified_v2.py")
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
for mode,target_set in (("v2",v2_set),("hard",hard_set)):
 for ex in m.generate(1000,20261009111,True,mode):
  if fold(ex.text).strip() in target_set:report["counts"]["holdout_"+mode+"_overlap"]+=1
issues={k:v for k,v in report["issues"].items()}
final={
 "status":"AUTOMATED_QA_COMPLETE_SEMANTIC_REVIEW_PENDING",
 "row_counts":{k:v for k,v in report["counts"].items()},
 "errors":dict(issues),
 "lengths":report["lengths"],"diversity":dict(report["modes"]),
 "goal_modes":dict(report["goal_modes"]),"via_modes":dict(report["via_modes"]),
 "samples_selected":len(selected),
 "source_sha256":hashlib.sha256((Path(r"D:\phenikaa\src\courier\nlp\synth.py")).read_bytes()).hexdigest(),
 "clone_sha256":hashlib.sha256((P/"synth_v2_clone.py").read_bytes()).hexdigest(),
 "seconds":round(time.time()-t0,1),
 "limitation":"Rule-based contracts and stratified reading do not prove 100% naturalness or downstream generalization."
}
assert final["source_sha256"]==final["clone_sha256"]
with OUT.open("x",encoding="utf-8") as f:json.dump(final,f,ensure_ascii=False,indent=2)
print("QA_240K",json.dumps({"errors":final["errors"],"lengths":final["lengths"],"counts":final["row_counts"],"samples_selected":len(selected),"duration":final["seconds"]},ensure_ascii=True),flush=True)
