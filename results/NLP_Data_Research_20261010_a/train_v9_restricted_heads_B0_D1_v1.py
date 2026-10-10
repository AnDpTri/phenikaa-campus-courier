"""V9 restricted-head B0/D1 experiment; no retrain old V5.
Freeze encoder, goal, fragile completely; train only urgent and via heads.
Matched B0 baseline weighted replacement vs D1 focused counterfactual.
Uses real TRAIN and synthetic only, never official val/test.
All outputs create-only within research workspace.
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
R=Path(r"D:\phenikaa");W=R/"results/NLP_Data_Research_20261010_a"
B_PATH=W/"experiments/v9_pilot_B_continuation.pt"
OLDMAN=W/"experiments/v9_controlled_ab_train_sampling_manifest.json"
V9=W/"experiments/semantic_v9_real_like_pilot.jsonl"
FOCUSED=W/"experiments/semantic_v9_focused_urgent_via_train_v2.jsonl"
FAMILY=W/"experiments/semantic_v8_family_challenge.jsonl"
HOLD=W/"experiments/semantic_v9_focused_urgent_via_holdout_v2.jsonl"
SOURCE=R/"results/NLP_V5_SF200_Scratch/synth_weighted_v4.py"
REPORT=W/"checkpoints/v9_restricted_heads_control_B0_vs_D1_report_v1.json"
LOG=W/"checkpoints/v9_restricted_heads_control_B0_vs_D1_metrics_v1.jsonl"
MAN=W/"experiments/v9_restricted_heads_sampling_manifest_v1.json"
MODELS={"B0":W/"experiments/v9_restricted_B0_control_v1.pt",
       "D1":W/"experiments/v9_restricted_D1_focused_v1.pt"}
for p in (REPORT,LOG,MAN,*MODELS.values()):
 if p.exists():raise FileExistsError(str(p))
torch.set_num_threads(3)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=="cuda"
start=time.time()
SEED=202610101183
INITIAL_B=torch.load(B_PATH,map_location="cpu",weights_only=False)
assert INITIAL_B["kind"]=="courier.nlp.neural_parser"
assert INITIAL_B["config"]=={"dim":96,"hidden":192}
KEEP_HEADS=("via_mode","via_type","via_anchor","urgent")
EPOCHS=2
MICRO=64
ACCUM=2
LR=.00015
def getj(p):
 return [json.loads(s) for s in p.open(encoding="utf-8")]
def spec(x):
 return None if x is None else TargetSpec(**x)
gsp=importlib.util.spec_from_file_location("readonly_v4_gen_focused_restricted",SOURCE)
G=importlib.util.module_from_spec(gsp);sys.modules[gsp.name]=G;gsp.loader.exec_module(G)
# Same 6,400 samples and same positions across control and intervention.
base_gen=G.generate(3400,seed=202610101809,holdout=False,mode="weighted")
base=[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in base_gen]
v9=getj(V9)
rand=random.Random(SEED+1)
v9chosen=rand.sample(v9,1000)
old=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"]) for z in v9chosen]
real_data=load_dataset(R/"Phenikaa_Campus_Courier_2026_v3/delivery_public","train")
real=[(s.mission.text,*spec_from_mission(s.mission),
       s.mission.urgent,s.mission.fragile) for s in real_data.scenes]
assert len(real)==2000
foc=getj(FOCUSED)
groups=collections.defaultdict(list)
for z in foc:groups[z["group_id"]].append(z)
rr=random.Random(SEED+2)
chosen_groups=(rr.sample(sorted(k for k in groups if "_urgent_" in k),150)
              +rr.sample(sorted(k for k in groups if "_via_" in k),150))
selected=[z for group in chosen_groups for z in sorted(groups[group],key=lambda z:z["provenance"]["label_switch"])]
assert len(selected)==600
positions=random.Random(SEED+3).sample(range(len(base)),600)
intervention=list(base)
for pos,z in zip(positions,selected):
 intervention[pos]=(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
corpora={"B0":base+old+real,"D1":intervention+old+real}
assert len(corpora["B0"])==len(corpora["D1"])==6400
assert sum(a!=b for a,b in zip(corpora["B0"],corpora["D1"]))==600
probes={}
for name,n,seed in (("v2",1600,20261010921),
                   ("weighted",1600,20261010922),
                   ("hard",1000,20261010923)):
 generated=G.generate(n,seed=seed,holdout=True,mode=name)
 probes[name]=[(x.text,x.goal,x.via,x.urgent,x.fragile) for x in generated]
family=getj(FAMILY)
fams=set(random.Random(202610101023+4).sample(sorted({z["group_id"] for z in family}),150))
probes["family1200"]=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"]) for z in family if z["group_id"] in fams]
hold=getj(HOLD)
probes["focus192"]=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"]) for z in hold]
# Ensure no direct train/probe collisions.
corpus_norm={fold(x[0]).strip() for x in corpora["D1"]}
for name,rows in probes.items():
 assert not corpus_norm&{fold(x[0]).strip() for x in rows},("DATA_COLLISION",name)
with MAN.open("x",encoding="utf-8") as f:
 json.dump({"initial_B":str(B_PATH),"seed":SEED,"train_rows_per_arm":6400,
   "real_rows_per_arm":len(real),"original_v9_rows":len(old),
   "baseline_weighted_rows":len(base),"targeted_new_examples":len(selected),
   "focus_groups":chosen_groups,"replaced_positions":positions,
   "freeze_encoder_goal_fragile":True,"trainable_heads":KEEP_HEADS,
   "epochs":EPOCHS,"max_lr":LR,"micro":MICRO,"accum":ACCUM,
   "holdout_sizes":{k:len(v) for k,v in probes.items()},
   "official_validation_test_touched":False},f,ensure_ascii=False,indent=2)
def make_net():
 net=NeuralParserNet(dim=96,hidden=192)
 net.load_state_dict(INITIAL_B["state_dicts"][0])
 for k,p in net.named_parameters():
  p.requires_grad=any(k.startswith(prefix+".") for head in KEEP_HEADS for prefix in
     ("attention."+head,"output."+head))
 # Frozen backbone and non-target heads exact invariants.
 frozen={k:v.detach().clone() for k,v in net.state_dict().items() if
  not any(k.startswith(prefix+".") for head in KEEP_HEADS
          for prefix in ("attention."+head,"output."+head))}
 return net.to(device),frozen
@torch.inference_mode()
def evaluate(net):
 net.eval()
 parser=NeuralMissionParser(net)
 report={}
 for name,rows in probes.items():
  c=collections.Counter()
  conf={k:collections.Counter() for k in ("via","urgent","fragile")}
  for i in range(0,len(rows),64):
   batch=rows[i:i+64]
   out=parser.predict_batch([x[0] for x in batch])
   for rr,pr in zip(batch,out):
    y=pr.parsed
    truth={"goal":rr[1],"via":rr[2],"urgent":rr[3],"fragile":rr[4]}
    hits={k:getattr(y,k)==v for k,v in truth.items()}
    for k,h in hits.items():c[k]+=h
    c["all"]+=all(hits.values())
    for k in conf:
     expected=bool(truth[k]);pred=bool(getattr(y,k))
     conf[k]["tp" if expected and pred else "fn" if expected else "fp" if pred else "tn"]+=1
  n=len(rows)
  entry={"n":n,"exact":{k:round(c[k]/n,6) for k in ("goal","via","urgent","fragile","all")}}
  for h,values in conf.items():
   pos=values["tp"]+values["fn"];neg=values["tn"]+values["fp"]
   entry[h]={"false_positive_rate":round(values["fp"]/neg,6),
             "false_negative_rate":round(values["fn"]/pos,6),
             "counts":dict(values)}
  report[name]=entry
 return report
def emit(f,name,**kw):
 data={"name":name,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
 f.write(json.dumps(data,ensure_ascii=False)+"\n");f.flush()
 print(name,json.dumps(kw,ensure_ascii=False)[:7000],flush=True)
metrics={}
with LOG.open("x",encoding="utf-8") as f:
 emit(f,"RESTRICTED_AB_START",train_rows=6400,targeted_replacements=600)
 for arm,corpus in corpora.items():
  random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
  net,frozen=make_net()
  trainable=[p for p in net.parameters() if p.requires_grad]
  assert len(trainable)==len(KEEP_HEADS)*6
  opt=torch.optim.AdamW(trainable,lr=LR,weight_decay=.0001)
  steps=math.ceil(len(corpus)/(MICRO*ACCUM))
  sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=EPOCHS*steps,pct_start=.1,anneal_strategy="cos")
  history=[]
  for ep in range(1,EPOCHS+1):
   rows=list(corpus)
   random.Random(SEED+ep).shuffle(rows)
   net.eval() # freeze dropout and backbone, head attention still differentiable
   opt.zero_grad(set_to_none=True)
   loss_sum=0.;count=0;steps_done=0
   for i in range(0,len(rows),MICRO):
    chunk=rows[i:i+MICRO]
    x=encode_texts([z[0] for z in chunk]).to(device)
    labs=[spec_labels(z[1],z[2],z[3],z[4]) for z in chunk]
    label={h:torch.tensor([z[h] for z in labs],device=device) for h in KEEP_HEADS}
    logits=net(x)
    loss=sum(nn.functional.cross_entropy(logits[h],label[h]) for h in KEEP_HEADS)
    if not torch.isfinite(loss):raise RuntimeError(("BAD_LOSS",arm,ep))
    (loss/ACCUM).backward()
    loss_sum+=float(loss.detach().item())*len(chunk);count+=len(chunk)
    if (i//MICRO+1)%ACCUM==0 or count==len(rows):
     nn.utils.clip_grad_norm_(trainable,1.)
     opt.step();sched.step();opt.zero_grad(set_to_none=True);steps_done+=1
   assert steps_done==steps
   item={"epoch":ep,"loss":round(loss_sum/count,6),"steps":steps_done}
   history.append(item)
   emit(f,"RESTRICTED_EPOCH",arm=arm,**item,elapsed_s=round(time.time()-start,1))
  current=net.state_dict()
  assert all(torch.equal(current[k].cpu(),v) for k,v in frozen.items()),"FROZEN_HEAD_CHANGED"
  result=evaluate(net)
  with MODELS[arm].open("xb") as out:
   torch.save({"kind":"courier.nlp.neural_parser","config":net.config,
      "state_dicts":[{k:v.detach().cpu() for k,v in current.items()}],
      "experiment":"V9 restricted head matched A/B",
      "seed":SEED,"trainable_heads":KEEP_HEADS,"epochs":EPOCHS,
      "initial_B":str(B_PATH),"arm":arm},out)
  metrics[arm]={"head_history":history,"probes":result,"path":str(MODELS[arm])}
  emit(f,"RESTRICTED_ARM_COMPLETE",arm=arm,probes=result,elapsed_s=round(time.time()-start,1))
  del net,opt,sched,trainable
  torch.cuda.empty_cache()
 comparisons={}
 for name in probes:
  a=metrics["B0"]["probes"][name];b=metrics["D1"]["probes"][name]
  comparisons[name]={"n":a["n"],
    "all_delta_pp":round(100*(b["exact"]["all"]-a["exact"]["all"]),3),
    "exact_pp":{k:round(100*(b["exact"][k]-a["exact"][k]),3)
                for k in ("goal","via","urgent","fragile")},
    "FPR_delta_pp":{k:round(100*(b[k]["false_positive_rate"]-a[k]["false_positive_rate"]),3)
      for k in ("via","urgent","fragile")},
    "FNR_delta_pp":{k:round(100*(b[k]["false_negative_rate"]-a[k]["false_negative_rate"]),3)
      for k in ("via","urgent","fragile")}}
  assert comparisons[name]["exact_pp"]["goal"]==0.0 and comparisons[name]["exact_pp"]["fragile"]==0.0
 report={"kind":"V9 restricted-head frozen backbone controlled A/B",
    "date":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "baseline":metrics["B0"],"targeted":metrics["D1"],"comparison":comparisons,
    "manifest":str(MAN),"elapsed_seconds":round(time.time()-start,1),
    "limitations":["Only one training seed", "synthetic and matched holdout",
     "no official validation/test", "frozen goal/fragile head means no learning there",
     "research checkpoints not approved for deployment"]}
 with REPORT.open("x",encoding="utf-8") as out:json.dump(report,out,ensure_ascii=False,indent=2)
 emit(f,"RESTRICTED_COMPLETE",report=str(REPORT),comparisons=comparisons,total_s=report["elapsed_seconds"])
print("COMPLETE",str(REPORT),flush=True)
