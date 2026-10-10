"""Exploratory weight interpolation: can a small fraction of focused C
repair urgent/via without forgetting fragile? Frozen B and C checkpoints only.
NO official validation or test. Never change or promote user artifacts.
"""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,random,sys,time
from pathlib import Path
import torch
from courier.nlp.neural import NeuralParserNet,NeuralMissionParser
from courier.nlp.parser import TargetSpec
ROOT=Path(r"D:\phenikaa")
W=ROOT/"results/NLP_Data_Research_20261010_a"
SOURCE=ROOT/"results/NLP_V5_SF200_Scratch/synth_weighted_v4.py"
FAMILY=W/"experiments/semantic_v8_family_challenge.jsonl"
HOLD=W/"experiments/semantic_v9_focused_urgent_via_holdout_v2.jsonl"
B_PATH=W/"experiments/v9_pilot_B_continuation.pt"
C_PATH=W/"experiments/v9_focused_C_seed0_continuation_v2.pt"
PREV=W/"checkpoints/v9_focused_via_urgent_controlled_BvsC_report_v2.json"
OUT=W/"checkpoints/v9_focused_weight_interpolation_study_v1.json"
if OUT.exists():raise FileExistsError(str(OUT))
torch.set_num_threads(3)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
start=time.time()
prior=json.loads(PREV.read_text(encoding="utf-8"))
B=torch.load(B_PATH,map_location="cpu",weights_only=False)["state_dicts"][0]
C=torch.load(C_PATH,map_location="cpu",weights_only=False)["state_dicts"][0]
assert set(B)==set(C)
spec=lambda x:None if x is None else TargetSpec(**x)
GSP=importlib.util.spec_from_file_location("readonly_v4_interp_v9",SOURCE)
G=importlib.util.module_from_spec(GSP);sys.modules[GSP.name]=G;GSP.loader.exec_module(G)
probes={}
for name,n,seed in (("weighted_new",1800,20261010712),("hard_new",1200,20261010713)):
 z=G.generate(n,seed=seed,holdout=True,mode=name.removesuffix("_new"))
 probes[name]=[(p.text,p.goal,p.via,p.urgent,p.fragile) for p in z]
family=[json.loads(s) for s in FAMILY.open(encoding="utf-8")]
groups=set(random.Random(202610101023+4).sample(sorted({z["group_id"] for z in family}),150))
probes["v8_family_fixed1200"]=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"])
 for z in family if z["group_id"] in groups]
held=[json.loads(s) for s in HOLD.open(encoding="utf-8")]
probes["v9_focused_heldout192"]=[(z["text"],spec(z["goal"]),spec(z["via"]),z["urgent"],z["fragile"]) for z in held]
assert all(prior["old_B"][name]["n"]==len(rows) for name,rows in probes.items())
@torch.inference_mode()
def evaluate(net,rows):
 parser=NeuralMissionParser(net)
 net.eval()
 cnt=collections.Counter()
 conf={key:collections.Counter() for key in ("via","urgent","fragile")}
 for i in range(0,len(rows),64):
  batch=rows[i:i+64]
  results=parser.predict_batch([x[0] for x in batch])
  for pred,truth in zip(results,batch):
   y=pred.parsed
   vals={"goal":truth[1],"via":truth[2],"urgent":truth[3],"fragile":truth[4]}
   hits={k:getattr(y,k)==v for k,v in vals.items()}
   for k,v in hits.items():cnt[k]+=v
   cnt["all"]+=all(hits.values())
   for key,st in conf.items():
    t=bool(vals[key]);p=bool(getattr(y,key))
    st["tp" if t and p else "fn" if t else "fp" if p else "tn"]+=1
 n=len(rows)
 out={"n":n,"exact":{k:round(cnt[k]/n,5) for k in ("goal","via","urgent","fragile","all")}}
 for head,z in conf.items():
  neg=z["tn"]+z["fp"];pos=z["tp"]+z["fn"]
  out[head]={"false_positive_rate":round(z["fp"]/neg,5),
    "false_negative_rate":round(z["fn"]/pos,5),"counts":dict(z)}
 return out
def new_net(alpha):
 net=NeuralParserNet(dim=96,hidden=192).to(device)
 state={k: B[k]+alpha*(C[k]-B[k]) for k in B}
 net.load_state_dict(state)
 return net
def score(name,trial):
 old=prior["old_B"][name]
 return {"delta_exact_all_pp":round(100*(trial["exact"]["all"]-old["exact"]["all"]),3),
 "delta_urgent_accuracy_pp":round(100*(trial["exact"]["urgent"]-old["exact"]["urgent"]),3),
 "delta_fragile_accuracy_pp":round(100*(trial["exact"]["fragile"]-old["exact"]["fragile"]),3),
 "delta_via_accuracy_pp":round(100*(trial["exact"]["via"]-old["exact"]["via"]),3),
 "delta_via_FPR_pp":round(100*(trial["via"]["false_positive_rate"]-old["via"]["false_positive_rate"]),3)}
out={}
for alpha in (0.10,0.25,0.40,0.60):
 model=new_net(alpha)
 results={name:evaluate(model,rows) for name,rows in probes.items()}
 gains={name:score(name,results[name]) for name in probes}
 # Conservative, predeclared safety gate: not a test-set optimization.
 gate=(gains["weighted_new"]["delta_exact_all_pp"]>=-2
     and gains["hard_new"]["delta_exact_all_pp"]>=-2
     and gains["weighted_new"]["delta_fragile_accuracy_pp"]>=-2
     and gains["hard_new"]["delta_fragile_accuracy_pp"]>=-2
     and gains["v8_family_fixed1200"]["delta_urgent_accuracy_pp"]>=1.5
     and gains["v8_family_fixed1200"]["delta_via_FPR_pp"]<=2
     and gains["v9_focused_heldout192"]["delta_exact_all_pp"]>=2)
 out[str(alpha)]={"metrics":results,"versus_B":gains,"passes_exploratory_gate":bool(gate)}
 print("ALPHA",alpha,"GATE",gate,"GAINS",gains,"ELAPSED_S",round(time.time()-start,1),flush=True)
 del model
 if device.type=="cuda":torch.cuda.empty_cache()
report={"kind":"V9 focused interpolation screening research only",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "baseline_B":str(B_PATH),"target_C":str(C_PATH),
 "alphas":[.10,.25,.40,.60],"probes":{name:len(rows) for name,rows in probes.items()},
 "results":out,"elapsed_seconds":round(time.time()-start,1),
 "decision":"DO NOT PROMOTE: interpolation is exploratory and uses synthetic sets with limited independence",
 "limitations":["Same evaluation sets inspected during study, thus not blind",
  "No official validation/test","Single training seed",
  "No combined checkpoint saved; deployment never authorized",
  "Do not select hyperparameters for official submission based solely on synthetic suites"]}
with OUT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("RESEARCH_INTERPOLATION_COMPLETE",str(OUT),flush=True)
