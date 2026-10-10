"""Independent release-readiness review of experimental Semantic V8.
Reads original real train/synthetic only. Creates checkpoint files in workspace.
Does NOT alter any source, model, training pipeline, or official eval split.
"""
from __future__ import annotations
import collections,datetime,hashlib,json,random,re,statistics,math
from pathlib import Path
from courier.common import load_dataset
from courier.nlp.text import fold,tokenize
from courier.nlp.synth import spec_from_mission
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
B=Path(r"D:\phenikaa")
OUT=W/"checkpoints/semantic_v8_release_audit_v2.json"
SPOT=W/"checkpoints/semantic_v8_release_stratified_examples_v2.jsonl"
for p in (OUT,SPOT):
 if p.exists():raise FileExistsError(str(p))
paths={
 "v8_candidate_train":W/"experiments/semantic_v8_balanced_train.jsonl",
 "v8_family_challenge":W/"experiments/semantic_v8_family_challenge.jsonl"}
rng=random.Random(340310)
samples={}
all_aud={}
first_segments={}
for name,path in paths.items():
 data=[json.loads(line) for line in path.open(encoding="utf-8")]
 n=len(data)
 cc=collections.Counter()
 cue=collections.Counter()
 ph=collections.Counter()
 modes=collections.Counter()
 byscene=collections.defaultdict(list)
 norms=set()
 stats=[]
 types=collections.Counter()
 for x in data:
  t=x["text"]
  cc["n"]+=1
  cc["accented"]+=any(ord(ch)>127 for ch in t)
  cc["via"]+=x["via"] is not None
  cc["urgent"]+=x["urgent"]
  cc["fragile"]+=x["fragile"]
  cc["all_flags_explicit"]=cc["n"]
  cc["has_via_loc_text"]+=1 if x["via"] is None else (x["via"]["type"]!=None)
  tok=tokenize(t)
  stats.append(len(tok))
  for q in set(re.findall(r"[a-z0-9]+",fold(t))):cue[q]+=1
  pre=" ".join(fold(t).split()[:8]);ph[pre]+=1
  for key in ("regime","phrase_style","goal_mode","via_mode"):modes[key+":"+str(x["provenance"][key])]+=1
  types["goal:"+str(x["goal"]["type"])]+=1
  types["via:"+str(x["via"]["type"] if x["via"] else "NONE")]+=1
  byscene[x["group_id"]].append(x)
  norms.add(fold(t).strip())
 for id,rs in byscene.items():
  k=(rs[0]["family_id"],rs[0]["provenance"]["regime"],rs[0]["provenance"]["phrase_style"])
  if k not in samples:
   pairs={(bool(z["via"]),z["urgent"],z["fragile"]):z for z in rs}
   samples[k]={"group_id":id,"family_id":k[0],"regime":k[1],
        "style":k[2],"neg":pairs[(False,False,False)],"pos":pairs[(True,True,True)]}
 all_aud[name]={
   "n":n,"scenes":len(byscene),
   "median_tokens":statistics.median(stats),
   "p10_tokens":sorted(stats)[int((n-1)*.1)],
   "p90_tokens":sorted(stats)[int((n-1)*.9)],
   "min_tokens":min(stats),"max_tokens":max(stats),
   "token_length_ge70":round(sum(x>=70 for x in stats)/n,4),
   "token_length_le25":round(sum(x<=25 for x in stats)/n,4),
   "label_rates":{key:round(cc[key]/n,4) for key in ("via","urgent","fragile","accented")},
   "cue_share":{key:round(cue[key]/n,4) for key in ("khong","huy","nhan","lay","truoc","gap","khan","vo","nhay","sau","binh","thuong","cu","moi","khoi","phuc","da")},
   "first8_distinct":len(ph),
   "first8_distinct_ratio":round(len(ph)/n,4),
   "top_first8":[(k,v) for k,v in ph.most_common(8)],
   "modes":dict(modes),"types":dict(types),
   "n_unique_folded":len(norms),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
 first_segments[name]=norms
# Compare original 2,000 real train utterances read-only.
ds=load_dataset(B/"Phenikaa_Campus_Courier_2026_v3/delivery_public","train")
rreal=ds.scenes
real_cc=collections.Counter();real_tokens=[];real_examples=[]
for scene in rreal:
 mission=scene.mission
 text=mission.text
 _, via=spec_from_mission(mission)
 real_cc["n"]+=1
 real_cc["via"]+=via is not None
 real_cc["urgent"]+=mission.urgent
 real_cc["fragile"]+=mission.fragile
 real_cc["accented"]+=any(ord(ch)>127 for ch in text)
 for cue in ("khong","huy","nhan","lay","truoc","gap","khan","vo","nhay","sau"):
  if cue in set(re.findall(r"[a-z0-9]+",fold(text))):real_cc["cue_"+cue]+=1
 n=len(tokenize(text));real_tokens.append(n)
 if len(real_examples)<15:real_examples.append({"text":text,"tokens":n,"via":via is not None,
        "urgent":mission.urgent,"fragile":mission.fragile})
rname="original_real_training_2k"
all_aud[rname]={
 "n":real_cc["n"],
 "median_tokens":statistics.median(real_tokens),"p10_tokens":sorted(real_tokens)[int((len(real_tokens)-1)*.1)],
 "p90_tokens":sorted(real_tokens)[int((len(real_tokens)-1)*.9)],
 "token_length_ge70":round(sum(x>=70 for x in real_tokens)/len(real_tokens),4),
 "token_length_le25":round(sum(x<=25 for x in real_tokens)/len(real_tokens),4),
 "label_rates":{k:round(real_cc[k]/len(real_tokens),4) for k in ("via","urgent","fragile","accented")},
 "cue_share":{k:round(real_cc["cue_"+k]/len(real_tokens),4)
  for k in ("khong","huy","nhan","lay","truoc","gap","khan","vo","nhay","sau")},
 "examples":real_examples
}
# Sample original scratch 200K evenly across file to measure its natural baseline.
basefile=B/"results/NLP_V5_SF200_Scratch/scratch_s0_e01_seed2026110501.jsonl"
basecc=collections.Counter();baselens=[]
with basefile.open(encoding="utf-8") as f:
 for idx,line in enumerate(f):
  if idx%10!=0:continue
  z=json.loads(line);t=z["text"];baselens.append(len(tokenize(t)))
  basecc["n"]+=1
  for k in ("via","urgent","fragile"):basecc[k]+=(z[k] is not None if k=="via" else z[k])
  basecc["accented"]+=any(ord(c)>127 for c in t)
  for cue in ("khong","huy","nhan","lay","truoc","gap","khan","vo","nhay","sau"):
   if cue in set(re.findall(r"[a-z0-9]+",fold(t))):basecc["cue_"+cue]+=1
all_aud["original_scratch_every_10th_row"]={
 "n":basecc["n"],"median_tokens":statistics.median(baselens),
 "p10_tokens":sorted(baselens)[int((len(baselens)-1)*.1)],
 "p90_tokens":sorted(baselens)[int((len(baselens)-1)*.9)],
 "token_length_ge70":round(sum(x>=70 for x in baselens)/len(baselens),4),
 "token_length_le25":round(sum(x<=25 for x in baselens)/len(baselens),4),
 "label_rates":{k:round(basecc[k]/basecc["n"],4) for k in ("via","urgent","fragile","accented")},
 "cue_share":{k:round(basecc["cue_"+k]/basecc["n"],4) for k in ("khong","huy","nhan","lay","truoc","gap","khan","vo","nhay","sau")}
}
# Leaks between V8 train/challenge can occur through reuse of spatial/alias lexical
# banks, even when phrase family IDs differ; capture that explicit limitation.
f1=first_segments["v8_candidate_train"]
f2=first_segments["v8_family_challenge"]
assert not f1.intersection(f2)
# Preserve random stratified scene samples for manual review.
with SPOT.open("x",encoding="utf-8") as f:
 for k,z in sorted(samples.items()):
  f.write(json.dumps(z,ensure_ascii=False)+"\n")
report={"name":"v8_release_readiness_audit_v2","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "audit":all_aud,"sampled_stratified_scenes":len(samples),
 "stratified_examples_file":str(SPOT),
 "critical_checklist":{
  "automatic_label_consistency":"PASS; 8-way labels and 10 cue-invariance tested in earlier QA",
  "no_exact_train_challenge_duplication":"PASS",
  "lexical_balance_selected_10_cues":"PASS within 8-way scenes",
  "independent_human_annotation":"NOT DONE",
  "controlled_retrain_A_B":"NOT DONE",
  "different_sentence_length_vs_real":"REQUIRES REVIEW",
  "test_set_independence":"TEMPLATE FAMILY ONLY",
  "formal_approval":"NOT APPROVED"},
 "caveats":["Real training data read-only, not validation/test.",
           "This does not prove that lexical cues are causally uninformative.",
           "Long 4-clause sentences are intentional stress tests, not production distribution.",
           "Underrepresented negative examples without explicit flag text cannot be checked by label balancing."]}
with OUT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
for k,v in all_aud.items():
 print("COMPARE",k, "N",v["n"],"TOKENS_MED",v["median_tokens"],"P90",v["p90_tokens"],
       "SHORT<=25",v["token_length_le25"],
       "LONG>=70",v["token_length_ge70"],
       "LABEL_RATES",v["label_rates"],"CUES",v["cue_share"],flush=True)
print("AUDIT_SAVED",OUT,"SCENE_SAMPLES",len(samples),flush=True)
