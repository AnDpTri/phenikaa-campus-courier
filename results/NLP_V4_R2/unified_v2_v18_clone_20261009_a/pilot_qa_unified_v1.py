"""Deterministic pilot and read-only QA for unified V2/V18-inspired generator.
Writes entirely NEW pilot/report files. Does not train or modify project files.
"""
import os,sys,importlib.util,hashlib,json,collections,statistics,re
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import spec_labels,MAX_TOKENS
ROOT=Path(r"D:\phenikaa")
OUT=ROOT/"results/NLP_V4_R2/unified_v2_v18_clone_20261009_a"
BASE=ROOT/"src/courier/nlp/synth.py"
CLONE=OUT/"synth_v2_clone.py"
SCRIPT=OUT/"synth_unified.py"
PILOT=OUT/"pilot_unified_hard_3000_v1.jsonl"
PROV=OUT/"pilot_unified_provenance_3000_v1.jsonl"
REPORT=OUT/"pilot_unified_audit_v1.json"
for p in [PILOT,PROV,REPORT]:
 if p.exists():raise RuntimeError("Refusing overwrite "+str(p))
assert hashlib.sha256(BASE.read_bytes()).digest()==hashlib.sha256(CLONE.read_bytes()).digest()
sp=importlib.util.spec_from_file_location("unified_v2_v18_pilot",SCRIPT)
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
import courier.nlp.synth as original
for holdout in (False,True):
 for seed in (0,14,2037):
  a=original.generate(75,seed,holdout)
  b=m.generate(75,seed,holdout)
  assert [((x.text,x.goal,x.via,x.urgent,x.fragile)) for x in a]==[tuple((x.text,x.goal,x.via,x.urgent,x.fragile)) for x in b]
print("UNCHANGED_V2_SEEDED_OUTPUT",450,flush=True)
examples=m.generate_roles(3000,2026100923,False,"hard")
exhold=m.generate_roles(200,2026100923,True,"hard")
cnt=collections.Counter();errs=collections.defaultdict(list);seen=set()
old_data=ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/incoming_40000.jsonl"
v18=ROOT/"results/NLP_V4_R2/semantic_generator_v12_from_zip_20261009_a/specialized_new_40000_v18_role_semantics.jsonl"
old_hashes=set()
for source in (old_data,v18):
 with source.open(encoding="utf-8") as f:
  for line in f:old_hashes.add(fold(json.loads(line)["text"]).strip())
lens=[];samples=[]
def fail(k,x,why):
 cnt["error_"+k]+=1
 if len(errs[k])<4:errs[k].append({"text":x.text[:180],"why":why})
for ex,p in examples:
 t=fold(ex.text);norm=t.strip();L=len(tokenize(ex.text));lens.append(L)
 cnt["n"]+=1;cnt["via"]+=ex.via is not None;cnt["accented"]+=any(ord(q)>127 for q in ex.text)
 cnt["former_goal"]+=p["former_destination"] is not None
 cnt["former_pickup"]+=p["former_pickup"] is not None
 cnt["former_goal_via"]+=p["former_destination"] is not None and ex.via is not None
 cnt["via_without_before"]+=ex.via is not None and not re.search(r"\btruoc\b",t)
 cnt["none_with_lay"]+=ex.via is None and bool(re.search(r"\blay\b",t))
 cnt["neg_role_count"]+=len(p["negative_roles"])
 cnt["near_far"]+=ex.goal.ref in ("near","far","anchor_near")
 if not (14<=L<=MAX_TOKENS):fail("max_tokens",ex,L)
 if norm in seen:fail("duplicate_local",ex,ex.text[:40])
 if norm in old_hashes:fail("copy_old_or_v18",ex,ex.text[:40])
 seen.add(norm)
 labels=spec_labels(ex.goal,ex.via,ex.urgent,ex.fragile)
 if len(labels)!=8:fail("bad_labels",ex,labels)
 if p["former_destination"] and fold(p["former_destination"]) not in t:fail("lost_cancel_goal",ex,p["former_destination"])
 if p["former_pickup"] and fold(p["former_pickup"]) not in t:fail("lost_cancel_via",ex,p["former_pickup"])
 for role in p["negative_roles"]:
  if fold(role["text"]) not in t:fail("lost_distractor",ex,role)
 if ex.urgent and not any(fold(s) in t for s in m.URGENT_POS):fail("missing_urgent",ex,ex.text)
 if ex.fragile and not any(fold(s) in t for s in m.FRAGILE_POS):fail("missing_fragile",ex,ex.text)
 if ex.goal.ref=="anchor_near" and not re.search(r"\b(khong tinh|loai tru|tru moc|khong ke|ngoai)\b",t):fail("anchor_exclude",ex,ex.text)
 if re.search(r"\b(khu vuc khu|dia diem diem|buu kien buu kien|ngoai kien buu kien|ke do moi dia chi|sau do can dia chi)\b",t):fail("grammar_known",ex,ex.text)
 if len(samples)<40 and cnt["n"]%70==1:samples.append({"example":ex.text,"goal":repr(ex.goal),"via":repr(ex.via),"urgent":ex.urgent,"fragile":ex.fragile,"roles":p})
with PILOT.open("x",encoding="utf-8",newline="\n") as f:
 for i,(ex,p) in enumerate(examples,1):
  j={"id":f"unified_hard_{i:06d}","text":ex.text,
   "goal":None if ex.goal is None else {"type":ex.goal.type,"ref":ex.goal.ref,"anchor":ex.goal.anchor},
   "via":None if ex.via is None else {"type":ex.via.type,"ref":ex.via.ref,"anchor":ex.via.anchor},
   "urgent":ex.urgent,"fragile":ex.fragile}
  f.write(json.dumps(j,ensure_ascii=False)+"\n")
with PROV.open("x",encoding="utf-8") as f:
 for ex,p in examples:f.write(json.dumps(p,ensure_ascii=False)+"\n")
holdset={fold(x.text).strip() for x,p in exhold}
overlap_holdout=len(holdset&seen)
v2_sample=m.generate(3000,2026100923,False,"v2")
base_n8=len({" ".join(fold(e.text).split()[:8]) for e in v2_sample})
hard_n8=len({" ".join(fold(e.text).split()[:8]) for e,p in examples})
ll=sorted(lens)
report={
 "status":"PILOT_TESTED_UNDER_SEMANTIC_INSPECTION",
 "v2_unchanged_reference":True,
 "original_sha256":hashlib.sha256(BASE.read_bytes()).hexdigest(),
 "clone_sha256":hashlib.sha256(CLONE.read_bytes()).hexdigest(),
 "counts":dict(cnt),"errors":dict(errs),
 "token_p50":ll[1500],"token_p90":ll[2700],"token_p99":ll[2970],"token_max":ll[-1],
 "hard_unique_prefix8":hard_n8,"v2_unique_prefix8":base_n8,
 "train_holdout_exact_overlap_200":overlap_holdout,
 "examples":samples,
}
with REPORT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("QA_SUMMARY",json.dumps({k:report[k] for k in ("counts","errors","token_p50","token_p90","token_p99","token_max","hard_unique_prefix8","v2_unique_prefix8","train_holdout_exact_overlap_200")},ensure_ascii=True),flush=True)
print("OUTPUT",PILOT,flush=True)
for z in samples[:7]:print("SAMPLE",json.dumps(z,ensure_ascii=True),flush=True)
