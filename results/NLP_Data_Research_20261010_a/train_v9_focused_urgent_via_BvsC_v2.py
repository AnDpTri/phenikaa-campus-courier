"""V9-focused repair test: reproduce prior arm B exact training corpus,
then arm C adds 600 independently paired urgent/via counterfactuals.

Same frozen Scratch seed0 epoch5 weights, 3 continuation epochs, same
optimizer, RNG schedule and training size. No official val/test; all writes
create-only in research workspace. B is the prior already trained V9 pilot.
"""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,math,random,sys,time
from pathlib import Path
import torch
from torch import nn
from courier.common import load_dataset
from courier.nlp.synth import spec_from_mission
from courier.nlp.neural import HEADS,NeuralParserNet,NeuralMissionParser,encode_texts,spec_labels
from courier.nlp.parser import TargetSpec
from courier.nlp.text import fold
ROOT=Path(r"D:\phenikaa")
W=ROOT/"results/NLP_Data_Research_20261010_a"
BASECKPT=ROOT/"results/NLP_V5_SF200_Scratch_5EP_20261010_a/scratch_s0_epoch05.pt"
SOURCE=ROOT/"results/NLP_V5_SF200_Scratch/synth_weighted_v4.py"
V9=W/"experiments/semantic_v9_real_like_pilot.jsonl"
FOCUS=W/"experiments/semantic_v9_focused_urgent_via_train_v2.jsonl"
HOLD=W/"experiments/semantic_v9_focused_urgent_via_holdout_v2.jsonl"
FAMILY=W/"experiments/semantic_v8_family_challenge.jsonl"
PRIOR_B=W/"experiments/v9_pilot_B_continuation.pt"
PRIORMAN=W/"experiments/v9_controlled_ab_train_sampling_manifest.json"
REPORT=W/"checkpoints/v9_focused_via_urgent_controlled_BvsC_report_v2.json"
LOG=W/"checkpoints/v9_focused_via_urgent_training_v2.jsonl"
MODEL=W/"experiments/v9_focused_C_seed0_continuation_v2.pt"
MAN=W/"experiments/v9_focused_BvsC_sampling_manifest_v2.json"
for p in (REPORT,LOG,MODEL,MAN):
 if p.exists():raise FileExistsError("Refuse to overwrite "+str(p))
torch.set_num_threads(3)
assert torch.cuda.is_available(),"Only run when free GPU present"
device=torch.device("cuda")
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
SEED=202610101023
EPOCHS=3
N_SYNTH=10000
N_REAL=2000
N_V9=1000
N_FOCUS=600
MICRO=64
ACCUM=2
MAX_LR=.00015
start=time.time()
def loadrows(path):
 return [json.loads(z) for z in path.open(encoding="utf-8")]
def spec(x):
 return None if x is None else TargetSpec(**x)
def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
def event(name,**kwargs):
 obj={"event":name,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs}
 f.write(json.dumps(obj,ensure_ascii=False)+"\n");f.flush()
 print(name,json.dumps(kwargs,ensure_ascii=False),flush=True)
gsp=importlib.util.spec_from_file_location("read_only_v4_gen_for_v9_focus",SOURCE)
G=importlib.util.module_from_spec(gsp);sys.modules[gsp.name]=G;gsp.loader.exec_module(G)
print("GENERATING_TRAIN_ORIGINAL",N_SYNTH,flush=True)
synthetic=G.generate(N_SYNTH,seed=2026101091,holdout=False,mode="weighted")
base=[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in synthetic]
del synthetic
realdata=load_dataset(ROOT/"Phenikaa_Campus_Courier_2026_v3/delivery_public","train")
real=[(s.mission.text,*spec_from_mission(s.mission),
       s.mission.urgent,s.mission.fragile) for s in realdata.scenes]
assert len(real)==N_REAL
prior=json.loads(PRIORMAN.read_text(encoding="utf-8"))
assert prior["seed"]==SEED and prior["synthetic_seed"]==2026101091
assert prior["checkpoint_sha256"]==digest(BASECKPT)
assert prior["candidate_sha256"]==digest(V9)
assert prior["source_generator_sha256"]==digest(SOURCE)
rng=random.Random(SEED+2)
oldpool=loadrows(V9)
originals=rng.sample(oldpool,N_V9)
origpositions=rng.sample(range(N_SYNTH),N_V9)
assert origpositions==prior["replaced_positions"]
assert [z["id"] for z in originals]==prior["replacement_candidate_ids"]
old_training=list(base)
for pos,z in zip(origpositions,originals):
 old_training[pos]=(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
assert len(old_training)==N_SYNTH
print("OLD_B_CORPUS_EXACTLY_RECONSTRUCTED",len(old_training),flush=True)
focused=loadrows(FOCUS)
byid=collections.defaultdict(list)
for z in focused:
 byid[z["group_id"]].append(z)
assert len(byid)==480 and all(len(v)==2 for v in byid.values())
for key,zs in byid.items():
 assert {s["provenance"]["label_switch"] for s in zs}=={False,True},key
order=random.Random(SEED+57)
groups=order.sample(sorted(k for k in byid if "_urgent_" in k),150)
groups+=order.sample(sorted(k for k in byid if "_via_" in k),150)
chosen=[z for k in groups for z in sorted(byid[k],key=lambda z:z["provenance"]["label_switch"])]
assert len(chosen)==N_FOCUS
presentold=set(origpositions)
otherpos=random.Random(SEED+58).sample([k for k in range(N_SYNTH) if k not in presentold],N_FOCUS)
new_training=list(old_training)
for pos,z in zip(otherpos,chosen):
 new_training[pos]=(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
assert sum(a!=b for a,b in zip(old_training,new_training))==N_FOCUS
# The original probe set is same fixed challenge selection; also fresh-seed
# generator probes, plus the NEW heldout authored independently prior to training.
probes={}
for name,n,seed in (("v2_new",1800,20261010711),
                    ("weighted_new",1800,20261010712),
                    ("hard_new",1200,20261010713)):
 samples=G.generate(n,seed=seed,holdout=True,mode=name.removesuffix("_new"))
 probes[name]=[(z.text,z.goal,z.via,z.urgent,z.fragile) for z in samples]
family=loadrows(FAMILY)
held_groups=set(random.Random(SEED+4).sample(sorted({x["group_id"] for x in family}),150))
probes["v8_family_fixed1200"]=[
  (z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
   for z in family if z["group_id"] in held_groups]
assert len(probes["v8_family_fixed1200"])==1200
hold=loadrows(HOLD)
probes["v9_focused_heldout192"]=[
 (z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"]) for z in hold]
probe_texts=set()
for name,rows in probes.items():
 forms={fold(x[0]).strip() for x in rows}
 train_forms={fold(x[0]).strip() for x in new_training}
 assert not forms&train_forms,(name,"TARGET_TRAIN_COLLISION")
 if name!="v9_focused_heldout192":probe_texts.update(forms)
assert not {fold(x[0]).strip() for x in probes["v9_focused_heldout192"]}&probe_texts
model=NeuralParserNet(dim=96,hidden=192)
artifact=torch.load(BASECKPT,map_location="cpu",weights_only=False)
assert artifact["config"]==model.config
def fresh_net():
 net=NeuralParserNet(dim=96,hidden=192).to(device)
 net.load_state_dict(artifact["state_dicts"][0])
 return net
oldcheckpoint=torch.load(PRIOR_B,map_location="cpu",weights_only=False)
assert oldcheckpoint["seed"]==SEED and oldcheckpoint["epochs"]==EPOCHS
assert len(oldcheckpoint["state_dicts"])==1
with MAN.open("x",encoding="utf-8") as sf:
 json.dump({"seed":SEED,"initial_checkpoint":str(BASECKPT),"prior_B_checkpoint":str(PRIOR_B),
  "reproduced_original_B":True,"old_V9_positions":origpositions,
  "focused_positions":otherpos,"focused_groups":groups,
  "focused_train_sha256":digest(FOCUS),"heldout_sha256":digest(HOLD),
  "fresh_seeds":{"v2":20261010711,"weighted":20261010712,"hard":20261010713},
  "probe_sizes":{k:len(v) for k,v in probes.items()},
  "train_row_count":N_SYNTH+N_REAL,
  "synthetic_example_count":N_SYNTH,"real_train_count":N_REAL,
  "synthetic_original_V9_replacements":N_V9,"additional_focused_replacements":N_FOCUS,
  "epochs":EPOCHS,"microbatch":MICRO,"accumulation":ACCUM,"peak_lr":MAX_LR,
  "steps_per_epoch":math.ceil((N_SYNTH+N_REAL)/(MICRO*ACCUM)),
  "no_official_val_test":True,"limitations":["Only seed0, pilot continuation, synthetic probes"],
  },sf,ensure_ascii=False,indent=2)
def eval_one(net,rows):
 parser=NeuralMissionParser(net)
 net.eval()
 n=0;cnt=collections.Counter()
 conf={k:collections.Counter() for k in ("urgent","fragile","via")}
 for offset in range(0,len(rows),64):
  chunk=rows[offset:offset+64]
  out=parser.predict_batch([z[0] for z in chunk])
  for row,p in zip(chunk,out):
   y=p.parsed
   true={"goal":row[1],"via":row[2],"urgent":row[3],"fragile":row[4]}
   hits={key:getattr(y,key)==val for key,val in true.items()}
   for k,ok in hits.items():cnt[k]+=ok
   cnt["all"]+=all(hits.values())
   for head in conf:
    want=bool(true[head]);pred=bool(getattr(y,head))
    conf[head]["tp" if want and pred else "fn" if want else "fp" if pred else "tn"]+=1
   n+=1
 result={"n":n,"exact":{k:round(cnt[k]/n,5) for k in ("goal","via","urgent","fragile","all")}}
 for head,c in conf.items():
  pos=c["tp"]+c["fn"];neg=c["fp"]+c["tn"]
  result[head]={"counts":dict(c),
   "sensitivity":round(c["tp"]/pos,5) if pos else None,
   "specificity":round(c["tn"]/neg,5) if neg else None,
   "false_positive_rate":round(c["fp"]/neg,5) if neg else None,
   "false_negative_rate":round(c["fn"]/pos,5) if pos else None}
 return result
def evalall(net):
 return {name:eval_one(net,rows) for name,rows in probes.items()}
with LOG.open("x",encoding="utf-8") as f:
 event("FOCUSED_AB_START",targeted_rows=N_FOCUS,train_rows=N_SYNTH+N_REAL,
       probes={name:len(rows) for name,rows in probes.items()})
 oldnet=fresh_net()
 oldnet.load_state_dict(oldcheckpoint["state_dicts"][0])
 B=evalall(oldnet)
 event("OLD_B_REEVALUATED",metrics=B,elapsed_s=round(time.time()-start,1))
 del oldnet
 torch.cuda.empty_cache()
 random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 net=fresh_net()
 opt=torch.optim.AdamW(net.parameters(),lr=MAX_LR,weight_decay=.0001)
 steps=math.ceil((N_SYNTH+N_REAL)/(MICRO*ACCUM))
 sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=MAX_LR,
     total_steps=EPOCHS*steps,pct_start=.1,anneal_strategy="cos")
 data=new_training+real
 history=[]
 begun=time.time()
 for epoch in range(1,EPOCHS+1):
  shuffled=list(data)
  random.Random(SEED+epoch).shuffle(shuffled)
  net.train();opt.zero_grad(set_to_none=True)
  loss_sum=0.;seen=0;updates=0
  for ix in range(0,len(shuffled),MICRO):
   part=shuffled[ix:ix+MICRO]
   x=encode_texts([row[0] for row in part]).to(device)
   labs=[spec_labels(z[1],z[2],z[3],z[4]) for z in part]
   labels={h:torch.tensor([z[h] for z in labs],device=device) for h in HEADS}
   logits=net(x)
   loss=sum(nn.functional.cross_entropy(logits[h],labels[h]) for h in HEADS)
   if not torch.isfinite(loss):raise RuntimeError(("NONFINITE_LOSS",epoch,ix))
   (loss/ACCUM).backward()
   seen+=len(part);loss_sum+=float(loss.detach().item())*len(part)
   if (ix//MICRO+1)%ACCUM==0 or seen==len(data):
    nn.utils.clip_grad_norm_(net.parameters(),1.)
    opt.step();sched.step();opt.zero_grad(set_to_none=True)
    updates+=1
    if updates%50==0:event("TRAIN_PROGRESS",epoch=epoch,step=updates,
       loss=round(loss_sum/seen,5),elapsed=round(time.time()-begun,1))
  assert seen==len(data) and updates==steps
  hist={"epoch":epoch,"steps":updates,"loss":round(loss_sum/seen,6)}
  history.append(hist)
  event("EPOCH_DONE",**hist)
 C=evalall(net)
 with MODEL.open("xb") as out:
  torch.save({"kind":"courier.nlp.neural_parser","config":net.config,
       "state_dicts":[{k:v.detach().cpu() for k,v in net.state_dict().items()}],
       "experiment":"V9 focused seed0 BversusC controlled continuation",
       "seed":SEED,"init_checkpoint":str(BASECKPT),
       "epochs":EPOCHS,"steps":EPOCHS*steps},out)
 event("NEW_C_EVALUATED",metrics=C,model=str(MODEL))
 comparisons={}
 for probe in probes:
  aa=B[probe];bb=C[probe]
  comparisons[probe]={"n":bb["n"],
   "exact_all_delta_pp":round(100*(bb["exact"]["all"]-aa["exact"]["all"]),3),
   "head_accuracy_delta_pp":{k:round(100*(bb["exact"][k]-aa["exact"][k]),3)
     for k in ("goal","via","urgent","fragile")},
   "false_positive_rate_delta_pp":{k:round(100*(bb[k]["false_positive_rate"]-aa[k]["false_positive_rate"]),3)
      for k in ("via","urgent","fragile")},
   "false_negative_rate_delta_pp":{k:round(100*(bb[k]["false_negative_rate"]-aa[k]["false_negative_rate"]),3)
      for k in ("via","urgent","fragile")}}
 rep={"experiment":"V9-focused B existing baseline versus C focused semantic repair",
   "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
   "design":{"same_initial_weights":True,"same_V9_previous_1000":True,
       "same_synthetic_generator":True,"same_real_train":True,
       "same_optimizer_hparams":True,"same_training_rng":True,
       "new_semantic_replacements":N_FOCUS,"optimizer_updates":EPOCHS*steps},
   "baseline_B_checkpoint":str(PRIOR_B),"experimental_C_checkpoint":str(MODEL),
   "manifest":str(MAN),"old_B":B,"new_C":C,"comparison":comparisons,
   "train_history_C":history,"runtime_seconds":round(time.time()-start,1),
   "limitations":["One training seed, synthetic tests, no independent human holdout",
      "Focused holdout uses unseen templates from same generator shell",
      "No official validation/test read or trained on",
      "Pilot only, never promote without independent validation"]} 
 with REPORT.open("x",encoding="utf-8") as out:json.dump(rep,out,ensure_ascii=False,indent=2)
 event("RESEARCH_COMPLETE",result=str(REPORT),comparison=comparisons,
    total_seconds=round(time.time()-start,1))
print("COMPLETE",str(REPORT),flush=True)
