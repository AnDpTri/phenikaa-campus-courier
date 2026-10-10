"""Read-only independent validation head audit for recent other-thread V5 model edits."""
from __future__ import annotations
import json,hashlib,datetime,collections,time
from pathlib import Path
import torch
from courier.common import load_dataset
from courier.nlp.synth import spec_from_mission
from courier.nlp.neural import NeuralMissionParser,HybridMissionParser,HybridThresholds
from courier.nlp.parser import MissionParser
from courier.nlp import resolve
ROOT=Path(r"D:\phenikaa")
OUT=ROOT/"results/NLP_Data_Research_20261010_a/checkpoints/recent_cross_thread_nlp_valid_audit_20261010.json"
if OUT.exists():raise FileExistsError(str(OUT))
torch.set_num_threads(3)
real=load_dataset(ROOT/"Phenikaa_Campus_Courier_2026_v3/delivery_public","validation").scenes
texts=[s.mission.text for s in real]
gold=[(*spec_from_mission(s.mission),s.mission.urgent,s.mission.fragile) for s in real]
variants={
 "V5_public_70_39":("neural_parser_v5_sf200_scratch_via095.pt",0.8),
 "V5_goal070":("neural_parser_v5_g070_v095.pt",0.7),
 "V5plusV9_goal070":("neural_parser_v5_v9_ensemble_g070_v095.pt",0.7)}
rules=MissionParser()
rparsed=[rules.parse(t) for t in texts]
start=time.monotonic()
def eq(y,t):
 x=(y.goal,y.via,y.urgent,y.fragile)
 ans=[int(a==b) for a,b in zip(x,t)]
 return ans+[int(all(ans))]
summ={}
differ={}
rawpars={}
for name,(file,threshold) in variants.items():
 p=ROOT/"artifacts/nlp"/file
 model=NeuralMissionParser.load(p)
 predictions=[]
 with torch.no_grad():
  for i in range(0,300,64):
   predictions.extend(model.predict_batch(texts[i:i+64]))
 assert len(predictions)==300
 hybrid=HybridMissionParser(model,thresholds=HybridThresholds(goal=threshold,via=.95,flags=2.0),rules=rules)
 hparsed=[hybrid.combine(rule,pred) for rule,pred in zip(rparsed,predictions)]
 neural=[eq(pred.parsed,y) for pred,y in zip(predictions,gold)]
 hybr=[eq(pr,y) for pr,y in zip(hparsed,gold)]
 rec={}
 for term,data in [('neural',neural),('hybrid',hybr)]:
  v=list(zip(*data))
  rec[term]={head:{"correct":sum(row),"n":300,"accuracy":round(sum(row)/300,6)}
       for head,row in zip(('goal','via','urgent','fragile','all'),v)}
 via_conf=collections.Counter()
 for r,p,y in zip(rparsed,predictions,gold):
  if r.via!=p.parsed.via:
   via_conf["disagree"]+=1
   if p.via_confidence >=.95:via_conf["override"]+=1
 rec["via_compared"]=dict(via_conf)
 summ[name]=rec
 rawpars[name]=hparsed
 print("EVAL",name,rec,"ELAPSED",round(time.monotonic()-start,1),flush=True)
for b in ("V5_goal070","V5plusV9_goal070"):
 ds=[i for i,(a,c) in enumerate(zip(rawpars["V5_public_70_39"],rawpars[b]))
    if (a.goal,a.via,a.urgent,a.fragile)!=(c.goal,c.via,c.urgent,c.fragile)]
 cnt=collections.Counter()
 examples=[]
 for i in ds:
  old=rawpars["V5_public_70_39"][i]
  new=rawpars[b][i]
  truth=gold[i]
  for field in ("goal","via","urgent","fragile"):
   if getattr(old,field)!=getattr(new,field):
    cnt["changed_"+field]+=1
    oc=getattr(old,field)==truth[("goal","via","urgent","fragile").index(field)]
    nc=getattr(new,field)==truth[("goal","via","urgent","fragile").index(field)]
    cnt[("fixed_" if nc and not oc else "regressed_" if oc and not nc else "other_")+field]+=1
  if len(examples)<8:
   examples.append({"scene_index":i,"scene_id":real[i].scene_id,
    "text":texts[i],"old":str(old),"new":str(new),"truth":str(truth)})
 differ[b]={"changed_scene_count":len(ds),"change_breakdown":dict(cnt),"examples":examples}
 print("DIFF",b,len(ds),dict(cnt),flush=True)
result={"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models":{k:{"filename":v[0],"goal_threshold":v[1]} for k,v in variants.items()},
        "n":300,"summaries":summ,"differences":differ,
        "elapsed_s":round(time.monotonic()-start,1),
        "scope":"validation text-only raw NLP; no CV and no solver",
        "limitations":["validation reused for selection and may be tuned; not blind test",
        "map-aware NLP differences not scored here, raw semantic parse comparison only",
        "comparison uses same rule parser and same thresholds except goal 0.7 vs 0.8",
        "does not verify new submission used exact model, only compares artifacts"]}
with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print("REPORT",str(OUT),flush=True)
