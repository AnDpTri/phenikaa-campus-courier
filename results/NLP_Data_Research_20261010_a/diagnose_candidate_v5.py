"""Read-only frozen-model diagnosis on experimental V5 counterfactual data.

Uses CPU only to avoid contention with live 5-epoch GPU training.
Creates new JSON + MD checkpoints in research workspace, never trains.
"""
from __future__ import annotations
import collections,datetime,hashlib,json,random,statistics,time
from pathlib import Path
import torch
from courier.nlp.neural import NeuralMissionParser
from courier.nlp.parser import TargetSpec
ROOT=Path(r"D:\phenikaa")
WORK=ROOT/"results/NLP_Data_Research_20261010_a"
DATA=WORK/"experiments"
OUT=WORK/"checkpoints/semantic_v5_frozen_model_diagnosis.json"
OUTMD=WORK/"checkpoints/semantic_v5_frozen_model_diagnosis.md"
for p in (OUT,OUTMD):
    if p.exists():raise FileExistsError("Refuse existing: "+str(p))
torch.set_num_threads(2)
cpu=torch.device("cpu")
st=time.time()
def data(path):
    with path.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
train=data(DATA/"semantic_v5_balanced_train.jsonl")
challenge=data(DATA/"semantic_v5_family_challenge.jsonl")
gids=sorted({z["group_id"] for z in train})
selected=set(random.Random(88001).sample(gids,250))
probe_train=[x for x in train if x["group_id"] in selected]
assert len(probe_train)==2000 and len(challenge)==2000
sets={"candidate_train_families":probe_train,"heldout_template_families":challenge}
paths={
  "Old_Scratch_epoch02":ROOT/"results/NLP_V5_SF200_Scratch/scratch_s0_epoch02.pt",
  "V5_SF200_finetuned":ROOT/"results/NLP_V5_SemanticFusion_200K/sf200_best.pt",
  "New_Scratch_5ep_seed1":ROOT/"results/NLP_V5_SF200_Scratch_5EP_20261010_a/scratch_s0_epoch05.pt",
}
for model,p in paths.items():
    if not p.is_file():raise FileNotFoundError(model+" "+str(p))
def obj(v):
    return TargetSpec(v["type"],v["ref"],v["anchor"]) if v is not None else None
def cases_metric(rows,predictions):
    ctr=collections.Counter();sub=collections.defaultdict(collections.Counter)
    pair=collections.defaultdict(dict)
    discord=[]
    for r,p in zip(rows,predictions):
        y=p.parsed
        truth={"goal":obj(r["goal"]),"via":obj(r["via"]),"urgent":r["urgent"],"fragile":r["fragile"]}
        acc={k:(getattr(y,k)==v) for k,v in truth.items()}
        acc["all"]=all(acc.values())
        ctr["n"]+=1
        for k,v in acc.items():ctr[k]+=v
        for subgroup in (("family_"+str(r["family_id"])),
                         ("via_true" if truth["via"] is not None else "via_false"),
                         ("urgent_true" if r["urgent"] else "urgent_false"),
                         ("fragile_true" if r["fragile"] else "fragile_false")):
            q=sub[subgroup];q["n"]+=1
            for k,v in acc.items():q[k]+=v
        if truth["via"] is not None and y.via is None:ctr["via_miss"]+=1
        if truth["via"] is None and y.via is not None:ctr["via_fp"]+=1
        pair[r["group_id"]][(r["via"] is not None,r["urgent"],r["fragile"])]=acc["all"]
        if not acc["all"] and len(discord)<12:
            discord.append({"id":r["id"],"family":r["family_id"],"truth":{
                "goal":r["goal"],"via":r["via"],"urgent":r["urgent"],"fragile":r["fragile"]},
                "pred":{"goal":None if y.goal is None else {"type":y.goal.type,"ref":y.goal.ref,"anchor":y.goal.anchor},
                        "via":None if y.via is None else {"type":y.via.type,"ref":y.via.ref,"anchor":y.via.anchor},
                        "urgent":y.urgent,"fragile":y.fragile},
                "text":r["text"]})
    all_pairs=sum(len(v)==8 and all(v.values()) for v in pair.values())
    return {"n":ctr["n"],"all_8way_scene_correct_fraction":round(all_pairs/len(pair),4),
            "acc":{k:round(ctr[k]/ctr["n"],4) for k in ("goal","via","urgent","fragile","all")},
            "via_miss_rate":round(ctr["via_miss"]/sum(r["via"] is not None for r in rows),4),
            "via_false_positive_rate":round(ctr["via_fp"]/sum(r["via"] is None for r in rows),4),
            "subgroups":{k:{"n":v["n"],
                **{z:round(v[z]/v["n"],4) for z in ("goal","via","urgent","fragile","all")}}
                for k,v in sub.items()},"error_examples":discord}
scores={}
for name,path in paths.items():
    m=NeuralMissionParser.load(path)
    for n in m.nets:n.to(cpu).eval()
    scores[name]={}
    print("EVAL_LOADING",name,flush=True)
    with torch.no_grad():
        for split,rows in sets.items():
            y=[]
            for i in range(0,len(rows),64):
                y.extend(m.predict_batch([r["text"] for r in rows[i:i+64]]))
            scores[name][split]=cases_metric(rows,y)
            s=scores[name][split]
            print("EVAL_SCORE",name,split,"n",s["n"],
                  "all",s["acc"]["all"],"via",s["acc"]["via"],
                  "fragile",s["acc"]["fragile"],"urgent",s["acc"]["urgent"],
                  "8way",s["all_8way_scene_correct_fraction"],flush=True)
    del m
r={"kind":"experimental_semantic_v5_frozen_diagnosis","created":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sources":{name:{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in paths.items()},
    "data":{"train_like_n":len(probe_train),"challenge_family_n":len(challenge),
            "train_like_families":list(range(6)),"challenge_families":[6,7]},
    "results":scores,"seconds":round(time.time()-st,1),
    "caveats":["All samples are synthetic. Challenge is family-disjoint only within experimental V5.",
              "Neither model has been trained on experimental V5 candidate; these are *pre-intervention* baseline scores.",
              "Do not select checkpoints or tune on the family-challenge results; protect for final comparison.",
              "Named and spatial goals require human label audit on a representative sample.",
              "Models evaluated entirely on CPU; GPU training not interrupted."]}
with OUT.open("x",encoding="utf-8") as f:json.dump(r,f,ensure_ascii=False,indent=2)
lines=["# Semantic V5 — frozen checkpoint diagnostics","","No model was retrained. CPU-only predictions.","",
       "| Model | Family split | all exact | goal | via | urgent | fragile | all 8 labels/case | via FP | via miss |",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for name,items in scores.items():
    for split,s in items.items():
        a=s["acc"]
        lines.append(f'| {name} | {split} | {a["all"]:.2%} | {a["goal"]:.2%} | {a["via"]:.2%} | {a["urgent"]:.2%} | {a["fragile"]:.2%} | {s["all_8way_scene_correct_fraction"]:.2%} | {s["via_false_positive_rate"]:.2%} | {s["via_miss_rate"]:.2%} |')
lines+=["","These are experimental generated challenges, NOT an official benchmark.",
        "For true evaluation, train with a frozen candidate train set, then test once on held-out families.",
        "No files outside research workspace were altered."]
with OUTMD.open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
print("EVAL_DONE",str(OUT),str(OUTMD),"seconds",r["seconds"],flush=True)
