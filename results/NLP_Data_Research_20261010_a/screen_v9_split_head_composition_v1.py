"""V9 split-head recombination screen: retain known reliable B0 via, D1 urgent.
No training or external artifact changes. Controlled paired evaluation. Only create
new research results. Validation/test official never read.
"""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,random,sys,time
from pathlib import Path
import torch
from courier.nlp.neural import NeuralParserNet,NeuralMissionParser, HEADS
from courier.nlp.parser import TargetSpec
from courier.nlp.text import fold
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
R=Path(r"D:\phenikaa")
B=W/"experiments/v9_restricted_B0_control_v1.pt"
D=W/"experiments/v9_restricted_D1_focused_v1.pt"
S=R/"results/NLP_V5_SF200_Scratch/synth_weighted_v4.py"
HOLD=W/"experiments/semantic_v9_focused_urgent_via_holdout_v2.jsonl"
FAMILY=W/"experiments/semantic_v8_family_challenge.jsonl"
REPORT=W/"checkpoints/v9_split_head_B0_D1_H_evaluation_v1.json"
if REPORT.exists():raise FileExistsError(str(REPORT))
torch.set_num_threads(3)
t0=time.time()
bb=torch.load(B,map_location="cpu",weights_only=False)
dd=torch.load(D,map_location="cpu",weights_only=False)
assert bb["seed"]==dd["seed"] and bb["config"]==dd["config"]
a,b=bb["state_dicts"][0],dd["state_dicts"][0]
assert set(a)==set(b)
changes=[k for k in a if not torch.equal(a[k],b[k])]
assert changes and all(k.startswith(("attention.urgent.","output.urgent.",
       "attention.via_","output.via_")) for k in changes)
# Comp H=B0 unchanged via heads + D1 only urgent head.
compose={}
for k in a:
 compose[k]=(b[k] if k.startswith(("attention.urgent.","output.urgent.")) else a[k])
assert any(not torch.equal(compose[k],a[k]) for k in compose)
def new_net(weights):
 n=NeuralParserNet(**bb["config"]).to("cuda" if torch.cuda.is_available() else "cpu")
 n.load_state_dict(weights)
 return n.eval()
nets={"B0":new_net(a),"D1":new_net(b),"H_urgent_only":new_net(compose)}
gs=importlib.util.spec_from_file_location("readonly_g_v9_split",S)
G=importlib.util.module_from_spec(gs);sys.modules[gs.name]=G;gs.loader.exec_module(G)
probes={}
for name,n,seed in (("v2_fresh",1600,20261010977),
                    ("weighted_fresh",1600,20261010978),
                    ("hard_fresh",1000,20261010979)):
 z=G.generate(n,seed=seed,holdout=True,mode=name.removesuffix("_fresh"))
 probes[name]=[(x.text,x.goal,x.via,x.urgent,x.fragile) for x in z]
def spec(x):return None if x is None else TargetSpec(**x)
zs=[json.loads(s) for s in FAMILY.open(encoding="utf8")]
fams=set(random.Random(202610101023+4).sample(sorted({z["group_id"] for z in zs}),150))
probes["family1200"]=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
 for z in zs if z["group_id"] in fams]
zs=[json.loads(s) for s in HOLD.open(encoding="utf8")]
probes["focus192"]=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"]) for z in zs]
@torch.no_grad()
def eval_model(net,rows):
 p=NeuralMissionParser(net)
 counts=collections.Counter()
 conf={k:collections.Counter() for k in ("via","urgent","fragile")}
 for off in range(0,len(rows),64):
  chunk=rows[off:off+64]
  outs=p.predict_batch([x[0] for x in chunk])
  for row,z in zip(chunk,outs):
   got=z.parsed;truth=dict(zip(("goal","via","urgent","fragile"),row[1:]))
   hits={k:getattr(got,k)==val for k,val in truth.items()}
   for k,h in hits.items():counts[k]+=int(h)
   counts["all"]+=all(hits.values())
   for k in conf:
    y=bool(truth[k]);pred=bool(getattr(got,k))
    conf[k]["tp" if y and pred else "fn" if y else "fp" if pred else "tn"]+=1
 n=len(rows)
 result={"n":n,"exact":{k:round(counts[k]/n,6) for k in ("goal","via","urgent","fragile","all")}}
 for k,x in conf.items():
  result[k]={"counts":dict(x),
    "fpr":round(x["fp"]/(x["fp"]+x["tn"]),6),
    "fnr":round(x["fn"]/(x["fn"]+x["tp"]),6)}
 return result
scores={k:{} for k in nets}
for name,net in nets.items():
 for s,rows in probes.items():
  scores[name][s]=eval_model(net,rows)
  print("MEASURE",name,s,scores[name][s]["exact"],
    {k:scores[name][s][k]["fpr"] for k in ("via","urgent")},flush=True)
compare={}
for s in probes:
 a=scores["B0"][s];h=scores["H_urgent_only"][s];d=scores["D1"][s]
 compare[s]={"n":a["n"],"HminusB0_all_pp":round((h["exact"]["all"]-a["exact"]["all"])*100,3),
 "HminusD1_all_pp":round((h["exact"]["all"]-d["exact"]["all"])*100,3),
 "urgent_acc_gain_vs_B0_pp":round((h["exact"]["urgent"]-a["exact"]["urgent"])*100,3),
 "HminusB0_via_FPR_pp":round((h["via"]["fpr"]-a["via"]["fpr"])*100,3),
 "HminusD1_via_FPR_pp":round((h["via"]["fpr"]-d["via"]["fpr"])*100,3)}
report={"kind":"V9 independent split-head recombination research screen",
 "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "models":{"B0":str(B),"D1":str(D),"H":"in-memory: B0 state except urgent heads from D1"},
 "unchanged_parameter_tensors":sum(torch.equal(bb["state_dicts"][0][k],dd["state_dicts"][0][k]) for k in bb["state_dicts"][0]),
 "total_parameter_tensors":len(bb["state_dicts"][0]),"changed_tensors":changes,
 "fresh_generator_seeds":[20261010977,20261010978,20261010979],
 "scores":scores,"comparisons":compare,"time_s":round(time.time()-t0,1),
 "limitations":["No human-blind data","synthetic probes including reused family challenge",
    "single seed training","No official val/test used","No deployable checkpoint saved"]}
with REPORT.open("x",encoding="utf8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("COMPLETE",str(REPORT),flush=True)
