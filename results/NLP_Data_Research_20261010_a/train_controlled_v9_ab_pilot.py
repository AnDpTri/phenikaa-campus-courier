"""Controlled paired continuation A/B on Scratch 5-epoch seed0 checkpoint.

Arm A: 10K new weighted V4 synthetic + 2K real train.
Arm B: same except replace 1K synthetic examples with cleaned V9.
Both: same seed, init weights, 3 epochs, batch64 accum2, AdamW, OneCycleLR.
Never accesses official val/test. Writes create-only in this workspace.
"""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,math,random,sys,time
from pathlib import Path
import torch
from torch import nn
from courier.common import load_dataset
from courier.nlp.synth import spec_from_mission
from courier.nlp.neural import (HEADS,NeuralParserNet,NeuralMissionParser,
                                 encode_texts,spec_labels)
from courier.nlp.parser import TargetSpec
ROOT=Path(r"D:\phenikaa")
WORK=ROOT/"results/NLP_Data_Research_20261010_a"
SOURCE=ROOT/"results/NLP_V5_SF200_Scratch"
CKPT=ROOT/"results/NLP_V5_SF200_Scratch_5EP_20261010_a/scratch_s0_epoch05.pt"
REPORT=WORK/"checkpoints/v9_controlled_ab_pilot_results.json"
LOG=WORK/"checkpoints/v9_controlled_ab_pilot_metrics.jsonl"
SAMP=WORK/"experiments/v9_controlled_ab_train_sampling_manifest.json"
MODELS={arm:WORK/("experiments/v9_pilot_"+arm+"_continuation.pt") for arm in ("A","B")}
for p in (REPORT,LOG,SAMP,*MODELS.values()):
 if p.exists():raise FileExistsError("Do not overwrite "+str(p))
torch.set_num_threads(3)
SEED=202610101023
EPOCHS=3
N_SYNTH=10000
N_AUG=1000
N_REAL=2000
MICRO=64
ACCUM=2
MAX_LR=.00015
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert DEVICE.type=="cuda","Require free GPU for controlled training"
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
def event(kind,**kw):
 item={"event":kind,"timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
 line=json.dumps(item,ensure_ascii=False)
 fh.write(line+"\n");fh.flush()
 print(kind,line,flush=True)
# Generate fresh matched baseline synthetic training samples once (read-only generator).
mod=importlib.util.spec_from_file_location("frozen_weighted_for_v9_ab",SOURCE/"synth_weighted_v4.py")
G=importlib.util.module_from_spec(mod);sys.modules[mod.name]=G;mod.loader.exec_module(G)
start=time.time()
print("GENERATING_MATCHED_BASELINE",N_SYNTH,flush=True)
synthetic=G.generate(N_SYNTH,seed=2026101091,holdout=False,mode="weighted")
base=[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in synthetic]
del synthetic
print("BASELINE_SYNTH_READY",len(base),round(time.time()-start,1),flush=True)
realdata=load_dataset(ROOT/"Phenikaa_Campus_Courier_2026_v3/delivery_public","train")
real=[(s.mission.text,*spec_from_mission(s.mission),
       s.mission.urgent,s.mission.fragile) for s in realdata.scenes]
assert len(real)==N_REAL
# Only use experimental candidate TRAIN split, not challenge.
v9file=WORK/"experiments/semantic_v9_real_like_pilot.jsonl"
with v9file.open(encoding="utf-8") as f:v9=[json.loads(x) for x in f]
selection=random.Random(SEED+2)
chosen=selection.sample(v9,N_AUG)
positions=selection.sample(range(N_SYNTH),N_AUG)
def spec(x):
 return None if x is None else TargetSpec(**x)
augmented=list(base)
for pos,z in zip(positions,chosen):
 augmented[pos]=(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
assert len(augmented)==len(base)
# Independent fixed probes: same seed and identical examples for A, B, before.
probes={}
for name,n,sd in (("v2",1000,2026111901),
                  ("weighted",1000,2026111902),("hard",600,2026111903)):
 k=G.generate(n,seed=sd,holdout=True,mode=name)
 probes[name]=[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in k]
# Challenge is disjoint structural families and READ-ONLY throughout.
cp=WORK/"experiments/semantic_v8_family_challenge.jsonl"
with cp.open(encoding="utf-8") as f:chall=[json.loads(s) for s in f]
selected_groups=set(random.Random(SEED+4).sample(sorted({x["group_id"] for x in chall}),150))
probes["family_challenge"]=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
 for z in chall if z["group_id"] in selected_groups]
assert len(probes["family_challenge"])==1200
artifact=torch.load(CKPT,map_location="cpu",weights_only=False)
assert artifact["config"]=={"dim":96,"hidden":192}
with SAMP.open("x",encoding="utf-8") as f:
 json.dump({"seed":SEED,"initial_checkpoint":str(CKPT),"checkpoint_sha256":sha(CKPT),
   "candidate_sha256":sha(v9file),"baseline_synthetic_examples":N_SYNTH,
   "real_train_n":len(real),"synthetic_replaced":N_AUG,
   "synthetic_seed":2026101091,"source_generator_sha256":sha(SOURCE/"synth_weighted_v4.py"),
   "replaced_positions":positions,
   "replacement_candidate_ids":[z["id"] for z in chosen],
   "probe_size":{k:len(v) for k,v in probes.items()},
   "epochs_per_arm":EPOCHS,"microbatch":MICRO,"accumulate":ACCUM,
   "peak_lr":MAX_LR,"initialization":"same frozen pretrained 5epoch scratch",
   "train_steps_expected":EPOCHS*math.ceil((N_SYNTH+N_REAL)/(MICRO*ACCUM)),
   "no_val_test":True,
   "limitations":["one seed pilot; not full from-scratch retraining",
     "short 3-epoch finetune; performance cannot establish full-data V5 improvement",
     "heldout synthetic probes; family challenge remains generator-created"]},
   f,ensure_ascii=False,indent=2)
@torch.no_grad()
def eval_rows(net,rows):
 net.eval()
 parser=NeuralMissionParser(net)
 n=0;correct=collections.Counter()
 absent=present=fps=missed=0
 for off in range(0,len(rows),96):
  chunk=rows[off:off+96]
  pp=parser.predict_batch([z[0] for z in chunk])
  for row,p in zip(chunk,pp):
   y=p.parsed
   q={"goal":y.goal==row[1],"via":y.via==row[2],
      "urgent":y.urgent==row[3],"fragile":y.fragile==row[4]}
   for name,v in q.items():correct[name]+=int(v)
   correct["all"]+=all(q.values())
   if row[2] is None:absent+=1;fps+=y.via is not None
   else:present+=1;missed+=y.via is None
   n+=1
 return {"n":n,"exact":{k:round(correct[k]/n,5)
       for k in ("goal","via","urgent","fragile","all")},
       "via_fpr":round(fps/absent,5) if absent else None,
       "via_miss_rate":round(missed/present,5) if present else None}
def evaluate(net):
 return {name:eval_rows(net,rows) for name,rows in probes.items()}
def fresh_net():
 net=NeuralParserNet(dim=96,hidden=192).to(DEVICE)
 net.load_state_dict(artifact["state_dicts"][0])
 return net
totrows=N_SYNTH+N_REAL
steps_per_epoch=math.ceil(totrows/(MICRO*ACCUM))
metrics={}
with LOG.open("x",encoding="utf-8") as fh:
 event("AB_START",n_synth=N_SYNTH,n_real=N_REAL,replace=N_AUG,
       training_epochs=EPOCHS,optimizer_updates_per_arm=EPOCHS*steps_per_epoch)
 net=fresh_net()
 initial=evaluate(net)
 event("INITIAL_EVAL",probes=initial)
 del net
 for arm,corpus in (("A",base),("B",augmented)):
  random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
  net=fresh_net()
  opt=torch.optim.AdamW(net.parameters(),lr=MAX_LR,weight_decay=.0001)
  sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=MAX_LR,
     total_steps=EPOCHS*steps_per_epoch,pct_start=.1,anneal_strategy="cos")
  rows=corpus+real
  history=[]
  begun=time.time()
  event("ARM_BEGIN",arm=arm,total_train_examples=len(rows),
         replacement_pct_of_synthetic=0 if arm=="A" else N_AUG/N_SYNTH)
  for epoch in range(1,EPOCHS+1):
   shuffled=list(rows)
   random.Random(SEED+epoch).shuffle(shuffled)
   net.train();opt.zero_grad(set_to_none=True)
   running=0.;count=0;updates=0
   for start_i in range(0,len(shuffled),MICRO):
    part=shuffled[start_i:start_i+MICRO]
    x=encode_texts([z[0] for z in part]).to(DEVICE)
    labs=[spec_labels(z[1],z[2],z[3],z[4]) for z in part]
    labels={head:torch.tensor([z[head] for z in labs],device=DEVICE)
       for head in HEADS}
    logits=net(x)
    loss=sum(nn.functional.cross_entropy(logits[h],labels[h]) for h in HEADS)
    if not torch.isfinite(loss):raise RuntimeError(("LOSS_NONFINITE",arm,epoch))
    (loss/ACCUM).backward()
    count+=len(part);running+=float(loss.detach().item())*len(part)
    minibatch=start_i//MICRO+1
    if minibatch%ACCUM==0 or count==len(shuffled):
     nn.utils.clip_grad_norm_(net.parameters(),1.)
     opt.step();sched.step();opt.zero_grad(set_to_none=True)
     updates+=1
     if updates%50==0: event("ARM_PROGRESS",arm=arm,epoch=epoch,
        optimizer_steps=updates,steps_per_epoch=steps_per_epoch,
        avg_train_loss=round(running/count,6),
        elapsed_s=round(time.time()-begun,1))
   assert count==totrows and updates==steps_per_epoch
   hist={"epoch":epoch,"train_loss":round(running/count,6),"steps":updates}
   history.append(hist)
   event("ARM_EPOCH_COMPLETE",arm=arm,**hist)
  results=evaluate(net)
  with MODELS[arm].open("xb") as f:
   torch.save({"kind":"courier.nlp.neural_parser","config":net.config,
     "state_dicts":[{k:v.detach().cpu() for k,v in net.state_dict().items()}],
     "experiment":"V9 A/B continuing 5epoch scratch",
     "arm":arm,"init_checkpoint":str(CKPT),
     "seed":SEED,"epochs":EPOCHS,"steps":EPOCHS*steps_per_epoch},f)
  metrics[arm]={"train_epochs":history,"probes":results,"model_path":str(MODELS[arm]),
                 "seconds":round(time.time()-begun,1)}
  event("ARM_FINISH",arm=arm,metrics=metrics[arm])
  del net,opt,sched
  torch.cuda.empty_cache()
 # Same sample pairs, so compare exact rate differences in percentage points.
 comparisons={}
 for name in probes:
  a=metrics["A"]["probes"][name];b=metrics["B"]["probes"][name]
  comparisons[name]={"n":a["n"],
     "delta_all_accuracy_pp":round(100*(b["exact"]["all"]-a["exact"]["all"]),3),
     "delta_via_accuracy_pp":round(100*(b["exact"]["via"]-a["exact"]["via"]),3),
     "delta_fragile_accuracy_pp":round(100*(b["exact"]["fragile"]-a["exact"]["fragile"]),3),
     "delta_urgent_accuracy_pp":round(100*(b["exact"]["urgent"]-a["exact"]["urgent"]),3),
     "delta_via_miss_rate_pp":round(100*(b["via_miss_rate"]-a["via_miss_rate"]),3),
     "delta_via_fpr_pp":round(100*(b["via_fpr"]-a["via_fpr"]),3)}
 report={"experiment":"controlled_v9_10percent_synthetic_replacement",
   "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
   "data_manifest":str(SAMP),"initial":initial,
   "arm_A_original_generator":metrics["A"],"arm_B_augmented_V9":metrics["B"],
   "armB_minus_armA":comparisons,
   "limitations":["Single-seed, short continuation pilot, not proof original V5 scratch fixed",
       "Synthetic challenge families sampled by fixed precommitted seed; no real heldout",
       "No official validation/test accessed","V9 source is not independently human-annotated"]} 
 with REPORT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
 event("AB_COMPLETE",report=str(REPORT),comparisons=comparisons,
       total_seconds=round(time.time()-start,1))
print("COMPLETE",REPORT,flush=True)
