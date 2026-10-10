"""Independent uniform full-range semantic audit sample and holdout collision smoke."""
from pathlib import Path
import json,random,hashlib,collections,re,importlib.util,sys
from courier.nlp.text import fold
P=Path(r"D:\phenikaa\results\NLP_V4_R2\unified_weighted_200k_20261010_a")
DATA=P/"train_weighted_unified_200000_seed2026101060.jsonl"
PROV=P/"train_weighted_unified_200000_roles_seed2026101060.jsonl"
OUT=P/"independent_semantic_strata_full_range_v2_320.jsonl"
REPORT=P/"additional_holdout_sampling_audit_v2.json"
for p in (OUT,REPORT):
 if p.exists():raise RuntimeError("Refuse overwrite "+str(p))
rng=random.Random(20261010980)
strata=("revision_both","revision_goal_via","revision_pickup_no_via",
 "goal_spatial","via_spatial","anchor_near","global_extreme","without_before",
 "no_via_with_lay","negative_roles","both_flags_true","both_flags_false",
 "accented","unaccented")
res={g:[] for g in strata};hits=collections.Counter()
uniform=[]
N=0;all_text=set()
with DATA.open(encoding="utf-8") as f, PROV.open(encoding="utf-8") as g:
 for line,p_line in zip(f,g):
  row=json.loads(line);meta=json.loads(p_line)
  N+=1
  oldg=bool(meta.get("former_destination"));oldv=bool(meta.get("former_pickup"))
  v=row["via"];t=fold(row["text"])
  classes={
    "revision_both":oldg and oldv,
    "revision_goal_via":oldg and v is not None,
    "revision_pickup_no_via":oldv and v is None,
    "goal_spatial":row["goal"]["ref"] is not None,
    "via_spatial":v is not None and v["ref"] is not None,
    "anchor_near":row["goal"]["ref"]=="anchor_near",
    "global_extreme":str(row["goal"]["ref"]).endswith("_most"),
    "without_before":v is not None and "truoc" not in t.split(),
    "no_via_with_lay":v is None and "lay" in t.split(),
    "negative_roles":bool(meta.get("negative_roles")),
    "both_flags_true":row["urgent"] and row["fragile"],
    "both_flags_false":not row["urgent"] and not row["fragile"],
    "accented":any(ord(c)>127 for c in row["text"]),
    "unaccented":all(ord(c)<=127 for c in row["text"]),
  }
  item={"dataset_line":N,"provenance":meta,**row}
  for name,truth in classes.items():
   if truth:
    hits[name]+=1
    q=res[name]
    if len(q)<12:q.append(item)
    else:
     k=rng.randrange(hits[name])
     if k<12:q[k]=item
  # Uniform random over ALL 200K.
  if len(uniform)<165:uniform.append(item)
  else:
   k=rng.randrange(N)
   if k<165:uniform[k]=item
  all_text.add(t.strip())
assert N==200000
selected={}
for name,lst in res.items():
 for item in lst:selected.setdefault(item["dataset_line"],(name,item))
for item in uniform:selected.setdefault(item["dataset_line"],("random_uniform",item))
with OUT.open("x",encoding="utf-8",newline="\n") as f:
 for line,(name,item) in sorted(selected.items()):
  f.write(json.dumps({"stratum":name,**item},ensure_ascii=False)+"\n")
# Verify fresh independent 5K holdout texts are not exact duplicates of this 200K train.
sp=importlib.util.spec_from_file_location("weighted_holdout_readonly_v2",P/"synth_weighted_v4.py")
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
overlap=0;hcount=5000
for e in m.generate(hcount,seed=2026101081,holdout=True,mode="weighted"):
 overlap+=fold(e.text).strip() in all_text
stat={"N":N,"sample_n":len(selected),"min_line":min(selected),"max_line":max(selected),
  "population_sizes":dict(hits),"exact_train_holdout_overlap":overlap,
  "holdout_n":hcount,"source_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
  "note":"Stratified reservoir + uniform sample over whole training set, not manually labeled ground truth."}
with REPORT.open("x",encoding="utf-8") as f:json.dump(stat,f,ensure_ascii=False,indent=2)
print("UNIFORM_FULL_RANGE",json.dumps(stat,ensure_ascii=True),flush=True)
