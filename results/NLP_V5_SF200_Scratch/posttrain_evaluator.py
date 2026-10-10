"""Post-training-only V5 Scratch evaluation. Runs once, create-only outputs.

Never load the official validation/test splits. Evaluate frozen epoch15
checkpoints (three individual seeds and ensemble) vs V2 seed0, V2 ensemble,
V4-R1 and V5-SF200 fine-tune on *identical* final synthetic holdouts and
the existing handwritten challenge set. All synthetic heldouts may still
share template families with the training generator.

No checkpoint selection, model retraining, threshold fitting or file edits.
"""
from __future__ import annotations
import collections, csv, hashlib, importlib.util, json, math, sys, time
from pathlib import Path

import torch
from courier.nlp.neural import NeuralMissionParser
from courier.nlp.text import fold

ROOT=Path(r"D:\phenikaa")
RUN=ROOT/"results/NLP_V5_SF200_Scratch"
REPORT=RUN/"scratch_final_report.json"
ASSEMBLY=RUN/"scratch_3seed_epoch15_ensemble.pt"
OUT_JSON=RUN/"posttrain_analysis_results.json"
OUT_MD=RUN/"posttrain_analysis_report.md"
OUT_ERRORS=RUN/"posttrain_discordant_examples.jsonl"
OUT_CURVES=RUN/"posttrain_learning_curves.csv"
for p in (OUT_JSON,OUT_MD,OUT_ERRORS,OUT_CURVES):
    if p.exists():raise FileExistsError("Refusing to replace existing posttrain report: "+str(p))
if not REPORT.is_file() or not ASSEMBLY.is_file():
    raise RuntimeError("Training is not fully complete: final report or fixed ensemble absent")
training=json.loads(REPORT.read_text(encoding="utf-8"))
assert training["status"]=="DONE" and training["epochs_per_seed"]==15
assert len(training["model_seed_final_checkpoints"])==3
for p in training["model_seed_final_checkpoints"]:
    assert Path(p).exists(),p
assert Path(training["ensemble_checkpoint"])==ASSEMBLY
SCRIPTS=ROOT/"scripts/nlp/train_neural_parser.py"
spec=importlib.util.spec_from_file_location("scratch_posttrain_helpers",SCRIPTS)
trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer
spec.loader.exec_module(trainer)
gen_source=RUN/"synth_weighted_v4.py"
generator_hash=hashlib.sha256(gen_source.read_bytes()).hexdigest()
assert generator_hash==json.loads((RUN/"scratch_config.json").read_text(encoding="utf-8"))["generator_sha256"]
g_spec=importlib.util.spec_from_file_location("scratch_posttrain_frozen_generator",gen_source)
G=importlib.util.module_from_spec(g_spec);sys.modules[g_spec.name]=G
g_spec.loader.exec_module(G)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
print("POSTTRAIN_EVALUATION_START",str(device),flush=True)

def convert(items):
    return [(e.text,e.goal,e.via,e.urgent,e.fragile) for e in items]

# Old final holdouts allow direct replication of the preceding A/B study.
# Fresh final holdouts use new, fixed, predeclared seeds, never present in
# scratch's train or synthetic-probe set.
SETS={
  "reference_v2_final_3000":trainer.labelled_synthetic(3000,seed=2026101171,holdout=True),
  "reference_weighted_final_2000":convert(G.generate(2000,seed=2026101172,holdout=True,mode="weighted")),
  "reference_hard_final_1000":convert(G.generate(1000,seed=2026101173,holdout=True,mode="hard")),
  "fresh_v2_final_3000":trainer.labelled_synthetic(3000,seed=2026123101,holdout=True),
  "fresh_weighted_final_2000":convert(G.generate(2000,seed=2026123102,holdout=True,mode="weighted")),
  "fresh_hard_final_1000":convert(G.generate(1000,seed=2026123103,holdout=True,mode="hard")),
  "handwritten_50":trainer.labelled_challenge(ROOT/"tests/nlp/challenge_missions.json"),
}
assert {k:len(v) for k,v in SETS.items()}=={
    "reference_v2_final_3000":3000,"reference_weighted_final_2000":2000,
    "reference_hard_final_1000":1000,"fresh_v2_final_3000":3000,
    "fresh_weighted_final_2000":2000,"fresh_hard_final_1000":1000,"handwritten_50":50}
print("FINAL_DATA_READY",{k:len(v) for k,v in SETS.items()},flush=True)
models={
  "V2_seed0":ROOT/"results/nlp_neural_v2/checkpoints/seed_20261008_best.pt",
  "V2_ensemble3":ROOT/"artifacts/nlp/neural_parser_v2.pt",
  "V4_R1":ROOT/"results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/model.pt",
  "V5_SF200_finetuned":ROOT/"results/NLP_V5_SemanticFusion_200K/sf200_best.pt",
  "Scratch_seed1":Path(training["model_seed_final_checkpoints"][0]),
  "Scratch_seed2":Path(training["model_seed_final_checkpoints"][1]),
  "Scratch_seed3":Path(training["model_seed_final_checkpoints"][2]),
  "Scratch_ensemble3":ASSEMBLY,
}
for name,path in models.items():
    assert path.is_file(),(name,path)
print("MODEL_PATHS_VERIFIED",list(models),flush=True)

def exact_ok(row,p):
    return (row[1]==p.goal and row[2]==p.via
            and (row[3] is None or row[3]==p.urgent)
            and (row[4] is None or row[4]==p.fragile))
def metric(rows,preds):
    z=trainer.spec_score(rows,preds)
    absent=[i for i,r in enumerate(rows) if r[2] is None]
    present=[i for i,r in enumerate(rows) if r[2] is not None]
    z["via_absent_n"]=len(absent)
    z["via_present_n"]=len(present)
    z["via_false_positive_rate"]=(sum(preds[i].via is not None for i in absent)/len(absent)
                                   if absent else None)
    z["via_missed_rate"]=(sum(preds[i].via is None for i in present)/len(present)
                          if present else None)
    return z
def compare_paired(rows,a,b):
    c1=[exact_ok(r,x) for r,x in zip(rows,a)]
    c2=[exact_ok(r,y) for r,y in zip(rows,b)]
    win=sum(x and not y for x,y in zip(c1,c2))
    loss=sum(y and not x for x,y in zip(c1,c2))
    n=len(rows)
    delta=(win-loss)/n
    var=max(0.0,((win+loss)/n-delta*delta)/n)
    return {"n":n,"scratch_only_correct":win,"baseline_only_correct":loss,
            "both_equal_outcome":n-win-loss,"delta_all":delta,
            "delta_95pct_normal_ci":[delta-1.96*math.sqrt(var),delta+1.96*math.sqrt(var)]}
def grouping(rows):
    d=collections.defaultdict(list)
    for i,(txt,goal,via,urgent,fragile) in enumerate(rows):
        ft=fold(txt)
        tok=set(ft.split())
        d["all"].append(i)
        d["via_present" if via is not None else "via_absent"].append(i)
        if via is None and "lay" in tok:d["pickup_keyword_no_via"].append(i)
        if via is not None and "truoc" not in tok:d["via_without_truoc"].append(i)
        if goal is not None and goal.ref is not None:d["spatial_goal"].append(i)
        if via is not None and via.ref is not None:d["spatial_via"].append(i)
        if any(ord(c)>127 for c in txt):d["accented"].append(i)
        else:d["unaccented"].append(i)
    return {k:v for k,v in d.items() if len(v)>=25}

start=time.monotonic()
predictions={}
scores={}
hashes={}
for name,path in models.items():
    print("POSTTRAIN_MODEL_LOADING",name,flush=True)
    art=torch.load(path,map_location="cpu",weights_only=False)
    assert art["config"]=={"dim":96,"hidden":192}
    assert len(art["state_dicts"])==(3 if name.endswith("ensemble3") else 1)
    model=NeuralMissionParser.load(path)
    for net in model.nets:net.to(device).eval()
    hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    predictions[name]={}
    scores[name]={}
    with torch.inference_mode():
        for ds,rows in SETS.items():
            p=[z.parsed for z in trainer.predict_all(model,[r[0] for r in rows],batch=112)]
            predictions[name][ds]=p
            scores[name][ds]=metric(rows,p)
            print("POSTTRAIN_SCORE",name,ds,"all",round(scores[name][ds]["all"],5),
                  "goal",round(scores[name][ds]["goal"],5),
                  "via",round(scores[name][ds]["via"],5),flush=True)
    for net in model.nets:net.to("cpu")
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
paired={}
baselines=["V2_seed0","V2_ensemble3","V4_R1","V5_SF200_finetuned"]
for base in baselines:
    paired[base]={ds:compare_paired(rows,predictions["Scratch_ensemble3"][ds],
                        predictions[base][ds]) for ds,rows in SETS.items()}
subgroups={}
for ds,rows in SETS.items():
    subgroups[ds]={}
    for kind,idx in grouping(rows).items():
        srows=[rows[i] for i in idx]
        subgroups[ds][kind]={"n":len(idx),"models":{
            name:metric(srows,[predictions[name][ds][i] for i in idx])
            for name in models}}
discordance=[]
for base in ("V2_ensemble3","V4_R1","V5_SF200_finetuned"):
    for ds,rows in SETS.items():
        a=predictions["Scratch_ensemble3"][ds]
        b=predictions[base][ds]
        for outcome in ("scratch_gain","scratch_regression"):
            count=0
            for row,x,y in zip(rows,a,b):
                if outcome=="scratch_gain" and not(exact_ok(row,x) and not exact_ok(row,y)):continue
                if outcome=="scratch_regression" and not(exact_ok(row,y) and not exact_ok(row,x)):continue
                def compact(p):
                    return None if p is None else {"type":p.type,"ref":p.ref,"anchor":p.anchor}
                def outputs(p):
                    return {"goal":compact(p.goal),"via":compact(p.via),
                            "urgent":p.urgent,"fragile":p.fragile}
                discordance.append({"compared_with":base,"dataset":ds,"outcome":outcome,
                    "text":row[0],"truth":{"goal":compact(row[1]),"via":compact(row[2]),
                            "urgent":row[3],"fragile":row[4]},
                    "scratch":outputs(x),"baseline":outputs(y)})
                count+=1
                if count>=12:break
# Audit overfit trajectories from the actual, saved per-epoch diagnostics.
curves=[]
overfit_summary={}
for seed,epoch_events in training["epoch_diagnostics"].items():
    serial=sorted(epoch_events,key=lambda x:x["epoch"])
    assert len(serial)==15
    for e in serial:
        curves.append({
            "seed":seed,"epoch":e["epoch"],
            "train_loss":e["train_loss"],
            "seen_probe_loss":e["overfit"]["seen_train_probe_loss"],
            "holdout_loss":e["overfit"]["synthetic_holdout_mean_loss"],
            "generalization_gap":e["overfit"]["holdout_minus_seen_gap"],
            "v2_all":e["synthetic_heldout"]["v2"]["exact"]["all"],
            "weighted_all":e["synthetic_heldout"]["weighted"]["exact"]["all"],
            "hard_all":e["synthetic_heldout"]["hard"]["exact"]["all"],
            "weighted_via":e["synthetic_heldout"]["weighted"]["exact"]["via"],
            "hard_via":e["synthetic_heldout"]["hard"]["exact"]["via"],
            "overfit_warning":e["overfit"]["overfit_warning"],
        })
    overfit_summary[seed]={
        "train_loss_first":serial[0]["train_loss"],
        "train_loss_last":serial[-1]["train_loss"],
        "holdout_loss_first":serial[0]["overfit"]["synthetic_holdout_mean_loss"],
        "holdout_loss_last":serial[-1]["overfit"]["synthetic_holdout_mean_loss"],
        "train_holdout_gap_last":serial[-1]["overfit"]["holdout_minus_seen_gap"],
        "synthetic_warning_epochs":[e["epoch"] for e in serial if e["overfit"]["overfit_warning"]],
        "epochs":len(serial),
        "highest_diagnostic_all_by_set":{
            name:{"epoch":max(serial,key=lambda e:e["synthetic_heldout"][name]["exact"]["all"])["epoch"],
                  "all":max(e["synthetic_heldout"][name]["exact"]["all"] for e in serial)}
            for name in ("v2","weighted","hard")},
        "fixed_checkpoint_epoch":15,
        "note":"Best diagnostic epoch reported descriptively, NOT used for checkpoint selection."
    }
result={
    "status":"DONE",
    "experiment":"V5-SF200-Scratch",
    "scope":"neural-only fixed epoch15 and exact same inputs; no official validation/test",
    "generated_final_holdout_seeds":{
        "fresh_v2_final_3000":2026123101,
        "fresh_weighted_final_2000":2026123102,
        "fresh_hard_final_1000":2026123103},
    "model_sha256":hashes,
    "model_checkpoints":{k:str(v) for k,v in models.items()},
    "set_sizes":{k:len(v) for k,v in SETS.items()},
    "scores":scores,"paired_scratch_ensemble_vs_baseline":paired,
    "subgroups":subgroups,"overfit_summary":overfit_summary,
    "training_trajectory":curves,
    "evaluation_seconds":round(time.monotonic()-start,1),
    "limitations":[
       "Scratch used 3 random-initialized nets and no official validation during training.",
       "Model selection precommitted epoch 15; no posthoc selection or tuning from this evaluation.",
       "Both final synthetic phrase-bank splits and the previous final splits may share template families with training.",
       "The 50 handwritten challenge samples alone cannot establish robust real-world generalization.",
       "Official validation and official test were NOT loaded here.",
       "Paired CIs are illustrative and do not adjust for template-family correlation or multiple comparisons."]
}
with OUT_JSON.open("x",encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
with OUT_ERRORS.open("x",encoding="utf-8",newline="\n") as f:
    for item in discordance:f.write(json.dumps(item,ensure_ascii=False)+"\n")
with OUT_CURVES.open("x",encoding="utf-8",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader()
    writer.writerows(curves)
lines=[
 "# KIỂM ĐỊNH HẬU HUẤN LUYỆN — V5-SF200-Scratch",
 "",
 "Đánh giá *neural-only*. Checkpoint Scratch là **epoch 15 cố định** của mỗi seed,",
 "không có official validation/test trong train hoặc đánh giá này. Các synthetic",
 "final holdout mới dùng seed chưa từng dùng cho train Scratch.",
 "",
 "## 1. Exact all accuracy (%)", "",
]
order=list(models)
for section,datasets in (
 ("Reference final 20261011",("reference_v2_final_3000","reference_weighted_final_2000","reference_hard_final_1000")),
 ("Fresh final 20261231",("fresh_v2_final_3000","fresh_weighted_final_2000","fresh_hard_final_1000")),
 ("Handwritten challenge",("handwritten_50",)),
):
    lines.extend(["### "+section,""])
    lines+=["| Model | "+" | ".join(datasets)+" |",
            "|---|"+ "|".join(["---:"]*len(datasets))+"|"]
    for name in order:
        lines.append("| "+name+" | "+" | ".join(
           f'{100*scores[name][ds]["all"]:.2f}' for ds in datasets)+" |")
    lines.append("")
for head in ("goal","via","urgent","fragile","via_false_positive_rate","via_missed_rate"):
    lines.extend(["## "+head+" (%) — final fresh holdouts","",
     "| Model | Fresh V2 | Fresh weighted | Fresh hard |",
     "|---|---:|---:|---:|"])
    for name in order:
        vals=[]
        for ds in ("fresh_v2_final_3000","fresh_weighted_final_2000","fresh_hard_final_1000"):
            v=scores[name][ds][head]
            vals.append("—" if v is None else f"{v*100:.2f}")
        lines.append("| "+name+" | "+" | ".join(vals)+" |")
    lines.append("")
lines+=["## Paired: Scratch ensemble vs prior models (fresh final only)","",
 "| Prior model | Test set | Scratch-only correct | Prior-only correct | Δ all (pp) |",
 "|---|---|---:|---:|---:|"]
for base in baselines:
    for ds in ("fresh_v2_final_3000","fresh_weighted_final_2000","fresh_hard_final_1000","handwritten_50"):
        q=paired[base][ds]
        lines.append(f'| {base} | {ds} | {q["scratch_only_correct"]} | {q["baseline_only_correct"]} | {100*q["delta_all"]:+.2f} |')
lines+=["","## Overfitting diagnostics", "",
 "| Seed | Training loss E1 → E15 | Synthetic probe loss E1 → E15 | Generalization gap E15 | Warning epochs |",
 "|---|---:|---:|---:|---|"]
for seed,k in overfit_summary.items():
    lines.append("| "+seed+" | "+
       f'{k["train_loss_first"]:.4f} → {k["train_loss_last"]:.4f} | '+
       f'{k["holdout_loss_first"]:.4f} → {k["holdout_loss_last"]:.4f} | '+
       f'{k["train_holdout_gap_last"]:.4f} | '+
       (", ".join(map(str,k["synthetic_warning_epochs"])) or "None")+" |")
lines+=["",
 "## Limitations and methodology",
 "- Không dùng official validation/test; không điều chỉnh checkpoint theo holdout.",
 "- Đây vẫn là synthetic phrase-bank holdout; không tương đương blind human test.",
 "- Bộ challenge viết tay chỉ có 50 câu và có thể không đại diện cho mọi ngữ cảnh thực tế.",
 "- Paired win/loss thể hiện bất đồng từng câu; confidence interval chưa điều chỉnh tương quan template.",
 "- Không thay thế so sánh hybrid/rule threshold hoặc đánh giá end-to-end trên bản đồ.",
 "",
 "## Artifacts",
 "- posttrain_analysis_results.json: toàn bộ số liệu đầy đủ, subgroups, paired comparisons.",
 "- posttrain_learning_curves.csv: tiến trình train/holdout 45 epoch.",
 "- posttrain_discordant_examples.jsonl: ví dụ Scratch cải thiện/suy giảm so với baseline.",
 "- Chỉ tạo file mới dưới thư mục Scratch; checkpoint gốc và mã train không bị sửa.",
]
with OUT_MD.open("x",encoding="utf-8",newline="\n") as f:f.write("\n".join(lines)+"\n")
print("POSTTRAIN_EVALUATION_FINISHED",json.dumps({
   "json":str(OUT_JSON),"report":str(OUT_MD),
   "summary":{
       ds:{name:round(scores[name][ds]["all"],4) for name in order}
       for ds in ("fresh_v2_final_3000","fresh_weighted_final_2000",
                  "fresh_hard_final_1000","handwritten_50")
   },
   "time_s":result["evaluation_seconds"]},ensure_ascii=False),flush=True)
