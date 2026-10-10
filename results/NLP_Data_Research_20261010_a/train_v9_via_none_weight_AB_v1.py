"""V9 pilot: controlled via-mode negative-class weighting with frozen trunk, heads.
All artifacts confined to research workspace. No official val/test.
Start H composition: B0 all except urgent D1, keep urgent frozen.
Both arms see exactly the same training set/schedule, differ only in
via_mode cross-entropy weight of label NONE: 1.0 control vs 2.5 experimental.
"""
from __future__ import annotations
import collections,datetime,importlib.util,json,math,random,sys,time
from pathlib import Path
import torch
from torch import nn
from courier.common import load_dataset
from courier.nlp.synth import spec_from_mission
from courier.nlp.neural import NeuralParserNet,NeuralMissionParser,encode_texts,spec_labels,VIA_MODES
from courier.nlp.parser import TargetSpec
from courier.nlp.text import fold
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
R=Path(r"D:\phenikaa")
P1=W/"experiments/v9_restricted_B0_control_v1.pt"
P2=W/"experiments/v9_restricted_D1_focused_v1.pt"
DATA=W/"experiments/semantic_v9_focused_urgent_via_train_v2.jsonl"
V9=W/"experiments/semantic_v9_real_like_pilot.jsonl"
HOLD=W/"experiments/semantic_v9_focused_urgent_via_holdout_v2.jsonl"
FAMILY=W/"experiments/semantic_v8_family_challenge.jsonl"
GEN=R/"results/NLP_V5_SF200_Scratch/synth_weighted_v4.py"
PRIORMAN=W/"experiments/v9_restricted_heads_sampling_manifest_v1.json"
REPORT=W/"checkpoints/v9_via_none_weighted_AB_report_v1.json"
LOG=W/"checkpoints/v9_via_none_weighted_AB_metrics_v1.jsonl"
MODELS={"E0_normal":W/"experiments/v9_via_E0_normal_control_v1.pt",
        "E25_none":W/"experiments/v9_via_E25_none_weight_v1.pt"}
for p in (REPORT,LOG,*MODELS.values()):
 if p.exists():raise FileExistsError(str(p))
torch.set_num_threads(3)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=="cuda"
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
T0=time.time()
SEED=202610101183
MICRO=64;ACCUM=2;EPOCHS=2;LR=.00015
getj=lambda p:[json.loads(s) for s in p.open(encoding="utf8")]
sp=lambda x:None if x is None else TargetSpec(**x)
a=torch.load(P1,map_location="cpu",weights_only=False)["state_dicts"][0]
b=torch.load(P2,map_location="cpu",weights_only=False)["state_dicts"][0]
init_state={k:(b[k] if k.startswith(("attention.urgent.","output.urgent.")) else a[k]) for k in a}
MAN=json.loads(PRIORMAN.read_text(encoding="utf8"))
assert MAN["seed"]==SEED and MAN["train_rows_per_arm"]==6400
gg=importlib.util.spec_from_file_location("read_only_G_v9_none_loss",GEN)
G=importlib.util.module_from_spec(gg);sys.modules[gg.name]=G;gg.loader.exec_module(G)
bg=G.generate(3400,seed=202610101809,holdout=False,mode="weighted")
original=[(z.text,z.goal,z.via,z.urgent,z.fragile) for z in bg]
vp=random.Random(SEED+1).sample(getj(V9),1000)
old=[(z["text"],sp(z["goal"]),sp(z["via"]),z["urgent"],z["fragile"]) for z in vp]
realdata=load_dataset(R/"Phenikaa_Campus_Courier_2026_v3/delivery_public","train")
real=[(z.mission.text,*spec_from_mission(z.mission),z.mission.urgent,z.mission.fragile) for z in realdata.scenes]
bygroup=collections.defaultdict(list)
for z in getj(DATA):bygroup[z["group_id"]].append(z)
groups=MAN["focus_groups"]
positions=MAN["replaced_positions"]
assert len(groups)==300 and len(positions)==600
chosen=[z for group in groups for z in sorted(bygroup[group],key=lambda z:z["provenance"]["label_switch"])]
for pos,z in zip(positions,chosen):
 original[pos]=(z["text"],sp(z["goal"]),sp(z["via"]),z["urgent"],z["fragile"])
corpus=original+old+real
assert len(corpus)==6400
probes={}
for name,n,seed in (("v2_fresh",1500,20261010990),
                    ("weighted_fresh",1500,20261010991),
                    ("hard_fresh",1000,20261010992)):
 gen=G.generate(n,seed=seed,holdout=True,mode=name.removesuffix("_fresh"))
 probes[name]=[(z.text,z.goal,z.via,z.urgent,z.fragile) for z in gen]
fam=getj(FAMILY)
gids=set(random.Random(202610101023+4).sample(sorted({z["group_id"] for z in fam}),150))
probes["family1200"]=[(z["text"],sp(z["goal"]),sp(z["via"]),z["urgent"],z["fragile"])
 for z in fam if z["group_id"] in gids]
hold=getj(HOLD)
probes["focused192"]=[(z["text"],sp(z["goal"]),sp(z["via"]),z["urgent"],z["fragile"]) for z in hold]
existing_train={fold(z[0]).strip() for z in corpus}
for name,rows in probes.items():
 assert not existing_train&{fold(z[0]).strip() for z in rows},(name,"collision")
@torch.inference_mode()
def evaluate(net):
 net.eval()
 parser=NeuralMissionParser(net)
 out={}
 for name,rows in probes.items():
  counts=collections.Counter()
  conf={z:collections.Counter() for z in ("via","urgent","fragile")}
  for i in range(0,len(rows),64):
   sub=rows[i:i+64]
   preds=parser.predict_batch([z[0] for z in sub])
   for truth,z in zip(sub,preds):
    true={"goal":truth[1],"via":truth[2],"urgent":truth[3],"fragile":truth[4]}
    hit={k:getattr(z.parsed,k)==v for k,v in true.items()}
    for k,v in hit.items():counts[k]+=int(v)
    counts["all"]+=all(hit.values())
    for k in conf:
     expected=bool(true[k]);pred=bool(getattr(z.parsed,k))
     conf[k]["tp" if expected and pred else "fn" if expected else "fp" if pred else "tn"]+=1
  n=len(rows)
  row={"n":n,"exact":{k:round(counts[k]/n,6) for k in ("goal","via","urgent","fragile","all")}}
  for h,v in conf.items():
   row[h]={"counts":dict(v),
    "FPR":round(v["fp"]/(v["fp"]+v["tn"]),6),
    "FNR":round(v["fn"]/(v["fn"]+v["tp"]),6)}
  out[name]=row
 return out
def net_new():
 net=NeuralParserNet(dim=96,hidden=192)
 net.load_state_dict(init_state)
 for key,param in net.named_parameters():
  param.requires_grad=key.startswith(("attention.via_","output.via_"))
 frozen={k:v.clone() for k,v in net.state_dict().items() if
   not k.startswith(("attention.via_","output.via_"))}
 return net.to(device),frozen
def emit(f,name,**kw):
 obj={"event":name,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw}
 f.write(json.dumps(obj,ensure_ascii=False)+"\n");f.flush()
 print(name,json.dumps(kw,ensure_ascii=False)[:4000],flush=True)
metrics={}
with LOG.open("x",encoding="utf8") as f:
 baseline,baseline_frozen=net_new()
 metrics["H_untrained"]={"metrics":evaluate(baseline)}
 emit(f,"START",train_n=len(corpus),probe_n={k:len(v) for k,v in probes.items()},
       baseline=metrics["H_untrained"],secs=round(time.time()-T0,1))
 del baseline
 torch.cuda.empty_cache()
 for arm,none_weight in (("E0_normal",1.0),("E25_none",2.5)):
  random.seed(SEED+97);torch.manual_seed(SEED+97);torch.cuda.manual_seed_all(SEED+97)
  net,frozen=net_new()
  pars=[p for p in net.parameters() if p.requires_grad]
  assert len(pars)==18
  opt=torch.optim.AdamW(pars,lr=LR,weight_decay=.0001)
  per=math.ceil(len(corpus)/(MICRO*ACCUM))
  sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,
   total_steps=EPOCHS*per,pct_start=.1,anneal_strategy="cos")
  w=torch.tensor([none_weight]+[1.]*(len(VIA_MODES)-1),device=device)
  hist=[]
  for epoch in range(1,EPOCHS+1):
   shuffled=list(corpus)
   random.Random(SEED+epoch).shuffle(shuffled)
   net.eval()
   opt.zero_grad(set_to_none=True)
   running=0.;seen=0;steps=0
   for index in range(0,len(shuffled),MICRO):
    batch=shuffled[index:index+MICRO]
    x=encode_texts([z[0] for z in batch]).to(device)
    lab=[spec_labels(z[1],z[2],z[3],z[4]) for z in batch]
    labels={h:torch.tensor([z[h] for z in lab],device=device) for h in
            ("via_mode","via_type","via_anchor")}
    logits=net(x)
    loss=nn.functional.cross_entropy(logits["via_mode"],labels["via_mode"],weight=w)
    loss+=nn.functional.cross_entropy(logits["via_type"],labels["via_type"])
    loss+=nn.functional.cross_entropy(logits["via_anchor"],labels["via_anchor"])
    if not torch.isfinite(loss):raise RuntimeError("nonfinite")
    (loss/ACCUM).backward()
    running+=float(loss.detach().item())*len(batch)
    seen+=len(batch)
    if (index//MICRO+1)%ACCUM==0 or seen==len(shuffled):
     nn.utils.clip_grad_norm_(pars,1.)
     opt.step();sched.step();opt.zero_grad(set_to_none=True)
     steps+=1
   assert steps==per
   rec={"epoch":epoch,"loss":round(running/seen,6)}
   hist.append(rec)
   emit(f,"EPOCH",arm=arm,**rec,time_s=round(time.time()-T0,1))
  assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in frozen.items())
  output=evaluate(net)
  with MODELS[arm].open("xb") as o:
   torch.save({"kind":"courier.nlp.neural_parser","config":net.config,
      "state_dicts":[{k:v.detach().cpu() for k,v in net.state_dict().items()}],
      "experiment":"V9 frozen via head none-weight A/B",
      "arm":arm,"none_weight":none_weight,"epochs":EPOCHS,
      "shared_urgent":"D1","shared_others":"B0"},o)
  metrics[arm]={"loss_weight_none":none_weight,"epochs":hist,
                "metrics":output,"path":str(MODELS[arm])}
  emit(f,"ARM_DONE",arm=arm,metrics=output,time_s=round(time.time()-T0,1))
  del net,opt,sched,pars
  torch.cuda.empty_cache()
 comp={}
 for name in probes:
  x=metrics["E0_normal"]["metrics"][name];y=metrics["E25_none"]["metrics"][name]
  z=metrics["H_untrained"]["metrics"][name]
  comp[name]={"n":x["n"],
   "E25_minus_E0_all_pp":round(100*(y["exact"]["all"]-x["exact"]["all"]),3),
   "E25_minus_H_all_pp":round(100*(y["exact"]["all"]-z["exact"]["all"]),3),
   "E25_minus_E0_via_FPR_pp":round(100*(y["via"]["FPR"]-x["via"]["FPR"]),3),
   "E25_minus_H_via_FPR_pp":round(100*(y["via"]["FPR"]-z["via"]["FPR"]),3),
   "E25_minus_E0_via_FNR_pp":round(100*(y["via"]["FNR"]-x["via"]["FNR"]),3),
   "E25_minus_H_via_FNR_pp":round(100*(y["via"]["FNR"]-z["via"]["FNR"]),3)}
 report={"experiment":"V9 none class via head-only loss controlled",
  "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "initial":"B0 all except urgent from D1",
  "only_via_heads_trainable":True,
  "same_data_rng_budget":True,
  "n_train":6400,"focused_train_rows":600,
  "loss_weights":{"E0_normal":1.0,"E25_none":2.5},
  "time_s":round(time.time()-T0,1),
  "results":metrics,"comparison":comp,
  "limitations":["One seed and synthetic probes","Evaluation holds reused family and focus suites",
    "No official val/test accessed", "Only experimental research files"]}
 with REPORT.open("x",encoding="utf8") as out:json.dump(report,out,ensure_ascii=False,indent=2)
 emit(f,"COMPLETE",report=str(REPORT),comparison=comp,time_s=report["time_s"])
print("V9_VIA_TRAIN_COMPLETE",str(REPORT),flush=True)
