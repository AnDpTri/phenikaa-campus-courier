"""Generate ONE task-weighted 200K corpus via the unified cloned V2 generator.

Old V2 source untouched; no merging prebuilt 40K, no train/test examples copied.
ALL output files created exclusively (mode x). 6 skill quotas are interleaved.
"""
import sys,importlib.util,json,hashlib,random,re,collections,time
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import spec_labels,MAX_TOKENS
ROOT=Path(r"D:\phenikaa")
P=ROOT/"results/NLP_V4_R2/unified_weighted_200k_20261010_a"
OUT=P/"train_weighted_unified_200000_seed2026101060.jsonl"
ROLES=P/"train_weighted_unified_200000_roles_seed2026101060.jsonl"
MANIFEST=P/"train_weighted_unified_200000_manifest.json"
for f in (OUT,ROLES,MANIFEST):
 if f.exists():raise RuntimeError("Never overwrite existing: "+str(f))
name="weighted_200k_generator"
sp=importlib.util.spec_from_file_location(name,P/"synth_weighted_v4.py")
m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m)
weights=m.WEIGHTED_RECIPES;assert sum(weights.values())==100
GEN_SEED=2026101060
g=m.Generator(seed=GEN_SEED,holdout=False,mode="weighted")
rng=random.Random(2026101061)
# Protect original corpus + previously generated 240k from exact folded-text reuse.
sources={
 "specialist_reference":ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/incoming_40000.jsonl",
 "curated_reference":ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/nlp_v4_curated_200000_quality_v2_reindexed.jsonl",
 "V18_reference":ROOT/"results/NLP_V4_R2/semantic_generator_v12_from_zip_20261009_a/specialized_new_40000_v18_role_semantics.jsonl",
 "prior_240k":ROOT/"results/NLP_V4_R2/unified_v2_v18_clone_20261009_a/train_unified_240000_interleaved_seed20261009.jsonl",
}
reference=set()
for name,file in sources.items():
 n=0
 with file.open(encoding="utf-8") as stream:
  for line in stream:
   try:reference.add(fold(json.loads(line)["text"]).strip());n+=1
   except Exception as exc:raise ValueError((name,n)) from exc
 print("READ_REFERENCE",name,n,flush=True)
seen=set();prefix=collections.Counter();quotas=collections.Counter();stats=collections.Counter();rejected=collections.Counter()
token_mode=collections.defaultdict(list)
bad=re.compile(r"\b(khu vuc khu|dia diem diem|dia diem vi tri|buu kien buu kien|ngoai kien buu kien|ke do moi dia chi|sau do can dia chi|tai tai|o o)\b")
SEQUENCE=("v2_general","contextual","pickup_reasoning","spatial_grounding","instruction_revision","distractor_negation")
acc={k:0 for k in SEQUENCE}
t0=time.time()
# Entire ONE stream: weighted-fair interleaving (never one special contiguous block).
with OUT.open("x",encoding="utf-8",newline="\n") as outf, ROLES.open("x",encoding="utf-8",newline="\n") as rolefile:
 for i in range(200000):
  for name in SEQUENCE:acc[name]+=weights[name]
  recipe=max(SEQUENCE,key=lambda name:(acc[name],-SEQUENCE.index(name)))
  acc[recipe]-=100;quotas[recipe]+=1
  g._forced_recipe=recipe
  attempts=0
  while True:
   attempts+=1
   if attempts>600:raise RuntimeError(("filter rejects too many candidates",recipe,i))
   ex=g.example()
   t=fold(ex.text).strip()
   tok=len(tokenize(ex.text))
   if tok<8 or tok>MAX_TOKENS:
    rejected["token"]+=1;continue
   if t in reference or t in seen:
    rejected["duplicate"]+=1;continue
   if bad.search(t):
    rejected["grammar"]+=1;continue
   first=" ".join(t.split()[:8])
   if prefix[first]>=35:
    rejected["repeated_opening"]+=1;continue
   p=g.last_provenance.copy()
   if recipe!="v2_general":
    if ex.urgent and not any(fold(q) in t for q in m.URGENT_POS):raise AssertionError("urgent positive cue missing")
    if ex.fragile and not any(fold(q) in t for q in m.FRAGILE_POS):raise AssertionError("fragile positive cue missing")
    for ky in ("former_destination","former_pickup"):
     if p[ky] and fold(p[ky]) not in t:raise AssertionError(ky)
    for nm in p["negative_roles"]:
     if fold(nm["text"]) not in t:raise AssertionError(("negative role",nm))
    if ex.goal.ref=="anchor_near" and not any(x in t for x in ("khong tinh","loai tru","ngoai","khong ke","tru moc")):
     raise AssertionError("anchor_near not explicit")
   if len(spec_labels(ex.goal,ex.via,ex.urgent,ex.fragile))!=8:raise AssertionError("wrong label shape")
   seen.add(t);prefix[first]+=1;break
  def targ(x):return None if x is None else {"type":x.type,"ref":x.ref,"anchor":x.anchor}
  row={"id":f"weighted_200k_{i+1:06d}","text":ex.text,"goal":targ(ex.goal),
       "via":targ(ex.via),"urgent":ex.urgent,"fragile":ex.fragile}
  outf.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
  rolefile.write(json.dumps({"id":row["id"],"recipe":recipe,**p},ensure_ascii=False,separators=(",",":"))+"\n")
  stats["n"]+=1;stats["via"]+=ex.via is not None
  stats["accented"]+=any(ord(ch)>127 for ch in ex.text)
  stats["spatial_goal"]+=ex.goal.ref is not None
  stats["spatial_via"]+=ex.via is not None and ex.via.ref is not None
  stats["via_without_truoc"]+=ex.via is not None and not re.search(r"\btruoc\b",t)
  stats["none_with_lay"]+=ex.via is None and bool(re.search(r"\blay\b",t))
  stats["urgent"]+=ex.urgent;stats["fragile"]+=ex.fragile
  if p["mode"]=="hard":
   stats["revision_goal"]+=p["former_destination"] is not None
   stats["revision_via"]+=p["former_pickup"] is not None
   stats["distractor_roles"]+=len(p["negative_roles"])
  token_mode[recipe].append(tok)
  if i%25000==24999:print("GENERATED",i+1,"quota",json.dumps(dict(quotas)), "reject",json.dumps(dict(rejected)),flush=True)
assert stats["n"]==200000
expected={k:weights[k]*2000 for k in SEQUENCE}
assert dict(quotas)==expected,(quotas,expected)
def quant(a):
 a.sort();n=len(a)
 return {"n":n,"p50":a[n//2],"p90":a[int(.9*n)],"p99":a[int(.99*n)],"max":a[-1]}
manifest={
 "status":"CORPUS_CREATED_REQUIRES_INDEPENDENT_QA_AND_SEMANTIC_REVIEW",
 "rows":200000,"single_stream":True,
 "weighted_recipies":dict(weights),"actual_recipe_counts":dict(quotas),
 "generator_file":str(P/"synth_weighted_v4.py"),"seed":GEN_SEED,
 "reference_n_normalized":len(reference),"reject_counts":dict(rejected),
 "statistics":dict(stats),"token_lengths_by_recipe":{k:quant(v) for k,v in token_mode.items()},
 "unique_first8":len(prefix),"max_first8_frequency":max(prefix.values()),
 "bytes":{"jsonl":OUT.stat().st_size,"roles":ROLES.stat().st_size},
 "sha256":{"jsonl":hashlib.sha256(OUT.read_bytes()).hexdigest(),
           "roles":hashlib.sha256(ROLES.read_bytes()).hexdigest(),
           "v2_source":hashlib.sha256((ROOT/"src/courier/nlp/synth.py").read_bytes()).hexdigest(),
           "v2_clone":hashlib.sha256((P/"synth_v2_clone.py").read_bytes()).hexdigest()},
 "holdout_used_in_training":False,
 "training_executed":False,
 "elapsed_seconds":round(time.time()-t0,1),
}
with MANIFEST.open("x",encoding="utf-8") as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print("COMPLETED",json.dumps({"quotas":dict(quotas),"stats":dict(stats),"reject":dict(rejected),
 "unique_first8":len(prefix),"max_first8_frequency":max(prefix.values()),
 "elapsed_s":manifest["elapsed_seconds"]},ensure_ascii=True),flush=True)
