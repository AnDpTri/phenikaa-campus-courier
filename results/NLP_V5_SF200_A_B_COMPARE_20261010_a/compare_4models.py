"""Read-only checkpoint A/B: V2 seed0 vs V2 3-net ensemble vs V4-R1 vs V5-SF200.

No model training, no test split, no alteration of any source or checkpoints.
New summary/error artifacts are created ONLY with exclusive opens in results.
"""
from __future__ import annotations

import collections, hashlib, importlib.util, json, math, re, sys, time
from pathlib import Path
import torch
from courier.common import load_dataset
from courier.nlp.neural import NeuralMissionParser
from courier.nlp.text import fold, tokenize

ROOT=Path(r"D:\phenikaa")
OUT=ROOT/"results/NLP_V5_SF200_A_B_COMPARE_20261010_a"
GEN_DIR=ROOT/"results/NLP_V4_R2/unified_weighted_200k_20261010_a"
paths={
    "V2_seed0":ROOT/"results/nlp_neural_v2/checkpoints/seed_20261008_best.pt",
    "V2_ensemble3":ROOT/"artifacts/nlp/neural_parser_v2.pt",
    "V4_R1":ROOT/"results/NLP_V4_Curated200K_Finetune_R1/resume_E1_B64_LR5e5/model.pt",
    "V5_SF200":ROOT/"results/NLP_V5_SemanticFusion_200K/sf200_best.pt",
}
outputs={
    "json":OUT/"ab_4models_results.json",
    "markdown":OUT/"ab_4models_report.md",
    "examples":OUT/"ab_4models_discordant_examples.jsonl",
}
for name,path in outputs.items():
    if path.exists():raise FileExistsError("Refusing overwrite: "+str(path))
for path in paths.values():
    if not path.is_file():raise FileNotFoundError(path)
start=time.time()
sp=importlib.util.spec_from_file_location("v5_ab_trainer_helpers",ROOT/"scripts/nlp/train_neural_parser.py")
trainer=importlib.util.module_from_spec(sp);sys.modules[sp.name]=trainer;sp.loader.exec_module(trainer)
spg=importlib.util.spec_from_file_location("v5_ab_weighted_generator",GEN_DIR/"synth_weighted_v4.py")
weighted=importlib.util.module_from_spec(spg);sys.modules[spg.name]=weighted;spg.loader.exec_module(weighted)
torch.set_num_threads(4)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("START_A_B_GPU",str(device),flush=True)
data_path=ROOT/"Phenikaa_Campus_Courier_2026_v3/delivery_public"
def tuples(items):return [(e.text,e.goal,e.via,e.urgent,e.fragile) for e in items]
# Freeze identical test inputs for all four models. Final seeds were previously
# predetermined and used to evaluate V2 seed0 and V5, NOT for V5 selection.
sets={
    "real_validation":trainer.labelled_real(load_dataset(data_path,"validation")),
    "v2_final_holdout":trainer.labelled_synthetic(3000,seed=2026101171,holdout=True),
    "weighted_final_holdout":tuples(weighted.generate(2000,seed=2026101172,holdout=True,mode="weighted")),
    "hard_final_holdout":tuples(weighted.generate(1000,seed=2026101173,holdout=True,mode="hard")),
    "handwritten_challenge":trainer.labelled_challenge(ROOT/"tests/nlp/challenge_missions.json"),
}
for name,rows in sets.items():
    assert rows,("Empty set",name)
    print("DATASET",name,len(rows),flush=True)
ranges={}
for name,rows in sets.items():
    ranges[name]=(sum(len(x) for x in list(sets.values())[:list(sets).index(name)]),
                  sum(len(x) for x in list(sets.values())[:list(sets).index(name)+1]))
all_rows=[r for rs in sets.values() for r in rs]
# Immutable ground truth and text set signature, includes exact sequence.
truth_signature=hashlib.sha256()
for text,goal,via,urgent,fragile in all_rows:
    truth_signature.update(repr((text,goal,via,urgent,fragile)).encode("utf-8"))
    truth_signature.update(b"\n")
print("GROUND_TRUTH_SHA256",truth_signature.hexdigest(),flush=True)

def metrics(rows,preds):
    basic=trainer.spec_score(rows,preds)
    absent=[i for i,r in enumerate(rows) if r[2] is None]
    present=[i for i,r in enumerate(rows) if r[2] is not None]
    basic["via_absent_n"]=len(absent)
    basic["via_false_positive_rate"]=sum(preds[i].via is not None for i in absent)/len(absent) if absent else None
    basic["via_present_n"]=len(present)
    basic["via_missed_rate"]=sum(preds[i].via is None for i in present)/len(present) if present else None
    return basic

def subgroup_index(rows):
    buckets=collections.defaultdict(list)
    for i,(text,goal,via,urgent,fragile) in enumerate(rows):
        ft=fold(text)
        toks=tokenize(text)
        if via is None and re.search(r"\blay\b",ft):buckets["no_via_but_lay"].append(i)
        if via is not None and not re.search(r"\btruoc\b",ft):buckets["via_without_truoc"].append(i)
        if via is None:buckets["via_none"].append(i)
        if via is not None:buckets["via_present"].append(i)
        if goal is not None and goal.ref not in ("named",None):buckets["goal_spatial"].append(i)
        if via is not None and via.ref not in ("named",None):buckets["via_spatial"].append(i)
        if len(toks)<=40:buckets["short_upto40tokens"].append(i)
        if any(ord(c)>127 for c in text):buckets["accented"].append(i)
        else:buckets["unaccented"].append(i)
        if urgent:buckets["urgent_true"].append(i)
        if fragile:buckets["fragile_true"].append(i)
    return buckets

def paired_stats(rows,a,b):
    # a and b are predictions; positive delta means a has more exact matches.
    a_ok=[p.goal==r[1] and p.via==r[2] and p.urgent==r[3] and p.fragile==r[4] for r,p in zip(rows,a)]
    b_ok=[p.goal==r[1] and p.via==r[2] and p.urgent==r[3] and p.fragile==r[4] for r,p in zip(rows,b)]
    wins=sum(x and not y for x,y in zip(a_ok,b_ok))
    losses=sum(y and not x for x,y in zip(a_ok,b_ok))
    n=len(rows)
    delta=(wins-losses)/n
    # Normal approximation to paired difference in proportions; 95% CI,
    # descriptive only (heldout phrasing has shared generation templates).
    second=(wins+losses)/n
    se=math.sqrt(max(0.0,(second-delta*delta)/n))
    return {"n":n,"v5_wins":wins,"v5_losses":losses,
            "ties":n-wins-losses,"delta_all":delta,
            "delta_95pct_normal_ci":[delta-1.96*se,delta+1.96*se]}

all_predictions={}
summary={
  "status":"EVALUATING","device":str(device),"input_model_paths":{k:str(p) for k,p in paths.items()},
  "scope":"NEURAL_ONLY model comparison; rule/hybrid threshold tuning intentionally excluded",
  "data_sources":{"real":"validation split only, NOT test",
        "v2_final":"holdout=True seed 2026101171",
        "weighted_final":"holdout=True seed 2026101172",
        "hard_final":"holdout=True seed 2026101173",
        "handwritten":"tests/nlp/challenge_missions.json"},
  "ground_truth_sha256":truth_signature.hexdigest(),
  "sizes":{k:len(v) for k,v in sets.items()},
  "models":{},"paired":{},"subgroups":{},
  "notes":["These heldout generators still share possible template families with training; not a blind real-world test.",
           "real_validation is a small 300-example split and may have been used in previous model selection.",
           "V2 ensemble uses 3 nets while other models use 1, so model size/inference cost differ.",
           "No comparison with Kaggle hidden test and no new training."]}
for model_name,path in paths.items():
    print("MODEL_LOADING",model_name,path,flush=True)
    artifact=torch.load(path,map_location="cpu",weights_only=False)
    config=artifact["config"]
    assert config=={"dim":96,"hidden":192},(model_name,config)
    state_list=artifact.get("state_dicts") or [artifact["state_dict"]]
    assert len(state_list)==(3 if model_name=="V2_ensemble3" else 1)
    model=NeuralMissionParser.load(path)
    for net in model.nets:net.to(device).eval()
    print("MODEL_STARTED",model_name,"nets",len(model.nets),flush=True)
    results_by_set={}
    preds_by_set={}
    with torch.inference_mode():
        for testname,rows in sets.items():
            t0=time.time()
            pred=[p.parsed for p in trainer.predict_all(model,[r[0] for r in rows],batch=128)]
            preds_by_set[testname]=pred
            results_by_set[testname]=metrics(rows,pred)
            print("AB_SCORE",model_name,testname,"all",f'{results_by_set[testname]["all"]:.4f}',
                  "goal",f'{results_by_set[testname]["goal"]:.4f}',
                  "via",f'{results_by_set[testname]["via"]:.4f}',
                  "sec",round(time.time()-t0,1),flush=True)
    summary["models"][model_name]={
        "nets":len(model.nets),"configuration":config,
        "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "scores":results_by_set}
    all_predictions[model_name]=preds_by_set
    for net in model.nets:net.to("cpu")
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    print("MODEL_DONE",model_name,"elapsed_total_s",round(time.time()-start,1),flush=True)
# Compare V5 to each earlier model, on exact same text and truth.
for base_name in ("V2_seed0","V2_ensemble3","V4_R1"):
    summary["paired"][base_name]={}
    for setname,rows in sets.items():
        a=all_predictions["V5_SF200"][setname]
        b=all_predictions[base_name][setname]
        summary["paired"][base_name][setname]=paired_stats(rows,a,b)
        print("PAIRED",base_name,setname,summary["paired"][base_name][setname],flush=True)
# Focused, *matched* type-conditioned strata on final holdouts and real validation.
for setname in ("real_validation","v2_final_holdout","weighted_final_holdout","hard_final_holdout"):
    rows=sets[setname]
    groups=subgroup_index(rows)
    summary["subgroups"][setname]={}
    for bucket,ids in sorted(groups.items()):
        if len(ids)<15:continue
        selected=[rows[i] for i in ids]
        summary["subgroups"][setname][bucket]={"n":len(ids)}
        for name in paths:
            selected_pred=[all_predictions[name][setname][i] for i in ids]
            met=metrics(selected,selected_pred)
            summary["subgroups"][setname][bucket][name]={
                "all":met["all"],"goal":met["goal"],"via":met["via"],
                "via_false_positive_rate":met["via_false_positive_rate"]}
# Representative discordant examples for V5 vs V2 ensemble and V4 R1.
examples=[]
for base_name in ("V2_ensemble3","V4_R1"):
    for testname in ("v2_final_holdout","weighted_final_holdout","hard_final_holdout","real_validation","handwritten_challenge"):
        rows=sets[testname]
        v5=all_predictions["V5_SF200"][testname]
        base=all_predictions[base_name][testname]
        for outcome in ("v5_newly_correct","v5_regressed"):
            n=0
            for row,p,b in zip(rows,v5,base):
                ok_p=bool(p.goal==row[1] and p.via==row[2] and p.urgent==row[3] and p.fragile==row[4])
                ok_b=bool(b.goal==row[1] and b.via==row[2] and b.urgent==row[3] and b.fragile==row[4])
                if (outcome=="v5_newly_correct" and not (ok_p and not ok_b)) or (outcome=="v5_regressed" and not(ok_b and not ok_p)):
                    continue
                def spec(x):return None if x is None else {"type":x.type,"ref":x.ref,"anchor":x.anchor}
                examples.append({"base":base_name,"dataset":testname,"outcome":outcome,
                     "text":row[0],"expected":{"goal":spec(row[1]),"via":spec(row[2]),"urgent":row[3],"fragile":row[4]},
                     "v5":{"goal":spec(p.goal),"via":spec(p.via),"urgent":p.urgent,"fragile":p.fragile},
                     "base_prediction":{"goal":spec(b.goal),"via":spec(b.via),"urgent":b.urgent,"fragile":b.fragile}})
                n+=1
                if n>=20:break
summary["status"]="DONE"
summary["elapsed_seconds"]=round(time.time()-start,1)
with outputs["json"].open("x",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2)
with outputs["examples"].open("x",encoding="utf-8",newline="\n") as f:
    for ex in examples:f.write(json.dumps(ex,ensure_ascii=False)+"\n")
lines=[
 "# Đối chiếu A/B: V2 seed 0 · V2 ensemble 3 seed · V4 R1 · V5-SF200",
 "",
 "Tất cả checkpoint **chỉ đọc**, không train, không đụng tập test. Chỉ đánh giá neural parser, không áp dụng rule/hybrid.",
 "",
 "## All exact accuracy",
 "",
 "| Bộ kiểm thử (N) | V2 seed 0 | V2 ensemble 3 | V4 R1 | V5-SF200 |",
 "|---|---:|---:|---:|---:|",
]
for setname in sets:
    z=[setname+f" ({len(sets[setname]):,})"]
    for modelname in paths:
        z.append(f'{100*summary["models"][modelname]["scores"][setname]["all"]:.2f}%')
    lines.append("| "+" | ".join(z)+" |")
lines+=["","## Via exact accuracy","",
 "| Bộ kiểm thử | V2 seed 0 | V2 ensemble 3 | V4 R1 | V5-SF200 |",
 "|---|---:|---:|---:|---:|"]
for setname in sets:
    z=[setname]
    for modelname in paths:
        z.append(f'{100*summary["models"][modelname]["scores"][setname]["via"]:.2f}%')
    lines.append("| "+" | ".join(z)+" |")
lines+=["","## Goal exact accuracy","",
 "| Bộ kiểm thử | V2 seed 0 | V2 ensemble 3 | V4 R1 | V5-SF200 |",
 "|---|---:|---:|---:|---:|"]
for setname in sets:
    z=[setname]
    for modelname in paths:
        z.append(f'{100*summary["models"][modelname]["scores"][setname]["goal"]:.2f}%')
    lines.append("| "+" | ".join(z)+" |")
lines+=["","## Paired V5 gain/loss: all exact","",
 "| Baseline | Tập kiểm thử | V5 đúng/base sai | V5 sai/base đúng | Δ all (điểm %) |",
 "|---|---|---:|---:|---:|"]
for base_name in summary["paired"]:
    for ds,pair in summary["paired"][base_name].items():
        lines.append(f'| {base_name} | {ds} | {pair["v5_wins"]} | {pair["v5_losses"]} | {pair["delta_all"]*100:+.2f} |')
lines+=["","## Hạn chế diễn giải",
 "- V5 chỉ được chốt theo tập dev; final synthetic holdout dùng seed khác nhưng không tách hoàn toàn các họ template.",
 "- Validation thực tế nhỏ và có thể đã tham gia lựa chọn mô hình trong lịch sử; không đại diện cho hidden test.",
 "- Ensemble 3 mô hình có chi phí suy luận lớn hơn mô hình đơn.",
 "- Bộ 49 câu khó cũ chỉ được đánh giá nếu có tập gán nhãn chính thức; phiên bản này không tự tạo/giả mạo 49 câu.",
 "- Các mẫu V5 đúng/sai tương đối so với baseline nằm trong ab_4models_discordant_examples.jsonl.",
 "",
 "## Đầu ra",
 "- ab_4models_results.json: toàn bộ chỉ số, nhóm tình huống, CI và checkpoint SHA.",
 "- ab_4models_discordant_examples.jsonl: ví dụ sai/đúng bất đồng đã gắn nhãn.",
 f'- Tổng thời gian đánh giá: {summary["elapsed_seconds"]} giây.',
]
with outputs["markdown"].open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
print("AB_FINISHED",json.dumps({"elapsed_seconds":summary["elapsed_seconds"],"report":str(outputs["json"]),
    "final_all":{k:{m:round(summary["models"][m]["scores"][k]["all"],4) for m in paths}
                 for k in sets}},ensure_ascii=False),flush=True)
