"""Controlled lexical ablation: replace heldout fragile evidence with seen synonyms."""
from __future__ import annotations
import datetime, importlib.util, json, sys
from pathlib import Path
import torch
from courier.nlp.neural import NeuralMissionParser
from courier.nlp.text import fold
ROOT=Path(r"D:\phenikaa")
RUN=ROOT/"results/NLP_V5_SF200_Scratch"
WORK=ROOT/"results/NLP_Data_Research_20261010_a"
gsp=importlib.util.spec_from_file_location("research_flag_readonly",RUN/"synth_weighted_v4.py")
G=importlib.util.module_from_spec(gsp);sys.modules[gsp.name]=G;gsp.loader.exec_module(G)
olds=("Lô hàng nhạy va chạm, phải tránh rung lắc.","Hàng này không nhạy va đập.")
news=("Kiện này chứa vật dễ vỡ, cần chống va đập.","Đơn này không chứa vật dễ vỡ.")
rows=G.generate(750,seed=2026111903,holdout=True,mode="hard")
def alter(t):
    out=t
    matched=False
    for a,b in zip(olds,news):
        if a in out:
            out=out.replace(a,b);matched=True
        if fold(a) in out:
            out=out.replace(fold(a),fold(b));matched=True
    return out,matched
mut=[alter(x.text) for x in rows]
assert len(rows)==750
def score(model,texts):
    answers=[]
    for off in range(0,len(texts),96):
        answers.extend(p.parsed for p in model.predict_batch(texts[off:off+96]))
    r={"n":len(rows),"accuracy_all":0,"fragile_accuracy":0,
       "fragile_true_n":0,"fragile_false_n":0,"fragile_true_correct":0,
       "fragile_false_correct":0,"fragile_true_predicted":0,
       "via_accuracy":0,"goal_accuracy":0}
    for e,p in zip(rows,answers):
        r["accuracy_all"]+=p.goal==e.goal and p.via==e.via and p.urgent==e.urgent and p.fragile==e.fragile
        r["fragile_accuracy"]+=p.fragile==e.fragile
        r["via_accuracy"]+=p.via==e.via
        r["goal_accuracy"]+=p.goal==e.goal
        key="fragile_true" if e.fragile else "fragile_false"
        r[key+"_n"]+=1
        r[key+"_correct"]+=p.fragile==e.fragile
        if e.fragile:r["fragile_true_predicted"]+=p.fragile is True
    r["accuracy_all"]/=len(rows);r["fragile_accuracy"]/=len(rows)
    r["via_accuracy"]/=len(rows);r["goal_accuracy"]/=len(rows)
    r["fragile_recall"]=r["fragile_true_correct"]/r["fragile_true_n"]
    r["fragile_specificity"]=r["fragile_false_correct"]/r["fragile_false_n"]
    return r
torch.set_num_threads(2)
output={"experiment":"fragile_holdout_lexical_swap_v1","timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n":len(rows),"matched_swaps":sum(b for _,b in mut),
        "old_phrase_banks":olds,"substituted_seen_phrase_banks":news,
        "models":{},"limitations":"Replaces only lexical evidence with semantically equivalent seen-bank phrases; a causal stress test, not proof of natural-language generalization."}
for ep in (1,2):
    path=RUN/f"scratch_s0_epoch{ep:02d}.pt"
    model=NeuralMissionParser.load(path)
    for net in model.nets:net.cpu().eval()
    with torch.inference_mode():
        orig=score(model,[x.text for x in rows])
        mod=score(model,[t for t,_ in mut])
    output["models"][f"epoch_{ep}"]={"original":orig,"swapped_synonym":mod,
     "delta_all":round(mod["accuracy_all"]-orig["accuracy_all"],4),
     "delta_fragile_accuracy":round(mod["fragile_accuracy"]-orig["fragile_accuracy"],4)}
    print(f"EPOCH_{ep}",json.dumps(output["models"][f"epoch_{ep}"],ensure_ascii=False),flush=True)
t=WORK/"checkpoints/fragile_lexical_ablation_v1.json"
with t.open("x",encoding="utf-8") as f:json.dump(output,f,ensure_ascii=False,indent=2)
print("FRAGILE_LEXICAL_CHECKPOINT",t)
