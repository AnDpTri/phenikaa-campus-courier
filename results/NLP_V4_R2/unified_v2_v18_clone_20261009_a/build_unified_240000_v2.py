"""Build a FRESH quality-filtered 200K V2 + 40K hard corpus from one cloned generator.

Outputs are exclusively new 'x'-created files; no old datasets/scripts changed.
Train language banks only. Heldout mode never mixed into train.
No model training or submission.
"""
import sys,importlib.util,json,re,hashlib,collections,statistics,time
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import spec_labels,MAX_TOKENS
ROOT=Path(r"D:\phenikaa")
BASE=ROOT/"results/NLP_V4_R2/unified_v2_v18_clone_20261009_a"
V2_OUT=BASE/"train_v2_fresh_200000_seed2026100925.jsonl"
HARD_OUT=BASE/"train_specialized_fresh_40000_seed2026100926.jsonl"
COMBINED=BASE/"train_unified_240000_interleaved_seed20261009.jsonl"
PROV=BASE/"train_specialized_roles_40000_seed2026100926.jsonl"
REPORT=BASE/"unified_240000_generation_manifest.json"
for file in (V2_OUT,HARD_OUT,COMBINED,PROV,REPORT):
 if file.exists():raise RuntimeError("Refuse overwrite "+str(file))
module_path=BASE/"synth_unified_v2.py"
s=importlib.util.spec_from_file_location("unified_clean_v2_producer",module_path)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
G_v2=m.Generator(seed=2026100925,holdout=False,mode="v2")
G_hard=m.Generator(seed=2026100926,holdout=False,mode="hard")
legacy=ROOT/"src/courier/nlp/synth.py"
clone=BASE/"synth_v2_clone.py"
assert hashlib.sha256(legacy.read_bytes()).digest()==hashlib.sha256(clone.read_bytes()).digest()
reference=collections.Counter();reserved=set()
sourcefiles=[
 ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/incoming_40000.jsonl",
 ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/nlp_v4_curated_200000_quality_v2_reindexed.jsonl",
 ROOT/"results/NLP_V4_R2/semantic_generator_v12_from_zip_20261009_a/specialized_new_40000_v18_role_semantics.jsonl",
]
for file in sourcefiles:
 with file.open(encoding="utf-8") as f:
  for line in f:reserved.add(fold(json.loads(line)["text"]).strip())
print("REFERENCE_UNIQUE_COUNT",len(reserved),flush=True)
SEED_V2=2026100925;SEED_HARD=2026100926
GOAL={"v2":200000,"hard":40000}
count=collections.Counter();rejected=collections.Counter();token_lens={"v2":[],"hard":[]};seen=set();prefix={"v2":set(),"hard":set()}
bad_grammar=re.compile(r"\b(khu vuc khu|dia diem diem|dia diem vi tri|buu kien buu kien|ngoai kien buu kien|ke do moi dia chi|sau do can dia chi)\b")
t0=time.time()
def row_and_meta(mode,index):
 gen=G_v2 if mode=="v2" else G_hard
 while True:
  ex=gen.example()
  n=len(tokenize(ex.text))
  if not 1<=n<=MAX_TOKENS:
   rejected[mode+"_length"]+=1;continue
  txt=fold(ex.text).strip()
  if not txt or txt in seen or txt in reserved:
   rejected[mode+"_duplicate"]+=1;continue
  if bad_grammar.search(txt):
   rejected[mode+"_grammar"]+=1;continue
  labels=spec_labels(ex.goal,ex.via,ex.urgent,ex.fragile)
  if len(labels)!=8:raise ValueError(("invalid heads",mode))
  if mode=="hard" and n<14:raise ValueError(("hard too short",ex.text))
  if mode=="hard":
   p=gen.last_provenance
   if p["former_destination"] and fold(p["former_destination"]) not in txt:raise ValueError("former destination missing")
   if p["former_pickup"] and fold(p["former_pickup"]) not in txt:raise ValueError("former pickup missing")
   for neg in p["negative_roles"]:
    if fold(neg["text"]) not in txt:raise ValueError("neg mention missing")
   if ex.urgent and not any(fold(phrase) in txt for phrase in m.URGENT_POS):raise ValueError("urgent true grounding")
   if ex.fragile and not any(fold(phrase) in txt for phrase in m.FRAGILE_POS):raise ValueError("fragile true grounding")
  seen.add(txt)
  count[mode]+=1
  count[mode+"_via"]+=ex.via is not None
  count[mode+"_accented"]+=any(ord(ch)>127 for ch in ex.text)
  count[mode+"_via_no_before"]+=ex.via is not None and not re.search(r"\btruoc\b",txt)
  count[mode+"_none_with_lay"]+=ex.via is None and bool(re.search(r"\blay\b",txt))
  if mode=="hard":
   count["hard_former_goal"]+=p["former_destination"] is not None
   count["hard_former_pickup"]+=p["former_pickup"] is not None
   count["hard_redirect_via"]+=p["former_destination"] is not None and ex.via is not None
   count["hard_negative_roles"]+=len(p["negative_roles"])
  token_lens[mode].append(n)
  prefix[mode].add(" ".join(txt.split()[:8]))
  obj={"id":f"fresh_{mode}_{index:06d}","text":ex.text,
    "goal":{"type":ex.goal.type,"ref":ex.goal.ref,"anchor":ex.goal.anchor},
    "via":None if ex.via is None else {"type":ex.via.type,"ref":ex.via.ref,"anchor":ex.via.anchor},
    "urgent":ex.urgent,"fragile":ex.fragile}
  return json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n",gen.last_provenance

# Exclusive creation, interleaved 5:1 to keep a balanced stream of both modes.
with V2_OUT.open("x",encoding="utf-8",newline="\n") as v2, \
     HARD_OUT.open("x",encoding="utf-8",newline="\n") as hard, \
     COMBINED.open("x",encoding="utf-8",newline="\n") as combined, \
     PROV.open("x",encoding="utf-8",newline="\n") as prov:
 for batch in range(40000):
  for offset in range(5):
   idx=batch*5+offset+1
   j,p=row_and_meta("v2",idx)
   v2.write(j);combined.write(j)
  j,p=row_and_meta("hard",batch+1)
  hard.write(j);combined.write(j)
  prov.write(json.dumps({"id":f"fresh_hard_{batch+1:06d}",**p},ensure_ascii=False,separators=(",",":"))+"\n")
  if (batch+1)%5000==0:
   print("BATCHES",batch+1,"ROWS",count["v2"],count["hard"],"SECONDS",round(time.time()-t0,1),flush=True)
assert count["v2"]==200000 and count["hard"]==40000
def percentile(mode):
 v=sorted(token_lens[mode])
 return {"p50":v[len(v)//2],"p90":v[int(len(v)*.9)],"p99":v[int(len(v)*.99)],"max":v[-1]}
manifest={
 "status":"CREATED_PENDING_INDEPENDENT_DATA_QA",
 "method":"single integrated cloned generator; v2 unchanged baseline + role-aware hard mode",
 "count":{"v2":count["v2"],"hard":count["hard"],"combined":count["v2"]+count["hard"]},
 "seeds":{"v2":SEED_V2,"hard":SEED_HARD},
 "source_sha256":hashlib.sha256(legacy.read_bytes()).hexdigest(),
 "cloned_v2_sha256":hashlib.sha256(clone.read_bytes()).hexdigest(),
 "quality":dict(count),"rejections":dict(rejected),
 "lengths":{k:percentile(k) for k in GOAL},
 "unique_8word_prefixes":{k:len(v) for k,v in prefix.items()},
 "outputs":{k:str(v) for k,v in {"v2":V2_OUT,"hard":HARD_OUT,"combined":COMBINED,"hard_provenance":PROV}.items()},
 "sha256":{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in {"v2":V2_OUT,"hard":HARD_OUT,"combined":COMBINED}.items()},
 "reference_unique_comparands":len(reserved),
 "train_validation_separation":"Only holdout=False used for corpus generation",
 "training_or_submission_performed":False,
 "runtime_seconds":round(time.time()-t0,1),
}
with REPORT.open("x",encoding="utf-8") as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print("CREATED_240K",json.dumps({k:manifest[k] for k in ("count","lengths","rejections","quality","unique_8word_prefixes","runtime_seconds")},ensure_ascii=True),flush=True)
