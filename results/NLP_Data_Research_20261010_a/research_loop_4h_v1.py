"""Four-hour, 25-cycle CPU research loop (10 minute cadence).

Exclusive outputs in one research workspace. Never modifies source, datasets,
training jobs, existing checkpoints, or real validation/test.
"""
from __future__ import annotations
import datetime, hashlib, importlib.util, json, math, random, statistics, sys, time, traceback
from collections import Counter
from pathlib import Path
import torch
from courier.nlp.neural import NeuralMissionParser
from courier.nlp.text import fold,tokenize
WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
SCRATCH=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch")
LATEST=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch_5EP_20261010_a")
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location("cvia_cycle_module",WORK/"counterfactual_via_v1.py")
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
spec2=importlib.util.spec_from_file_location("research_gen_cycle_module",SCRATCH/"synth_weighted_v4.py")
gen=importlib.util.module_from_spec(spec2);sys.modules[spec2.name]=gen;spec2.loader.exec_module(gen)
MODEL_PATHS=[SCRATCH/"scratch_s0_epoch01.pt",SCRATCH/"scratch_s0_epoch02.pt"]
assert all(p.is_file() for p in MODEL_PATHS)
assert WORK.joinpath("checkpoints/corpus_audit_v1.json").exists()
ORIGINAL_AUDIT=json.loads(WORK.joinpath("checkpoints/corpus_audit_v1.json").read_text(encoding="utf-8"))
SCHEDULE_INTERVAL_S=600
MINIMUM_ROUNDS=25                 # cycle 0 at t0, cycle 24 at t+4 hours
START_TS=time.time()
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()
def save_new(path,obj):
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write("\n")
def probe_stats(rows):
    c=Counter();n=len(rows)
    for a in rows:
        w=set(fold(a.text).split())
        yes=a.via is not None
        c["via"]+=yes;c["fragile"]+=a.fragile;c["urgent"]+=a.urgent
        c["accents"]+=any(ord(z)>127 for z in a.text)
        for lex in ("lay","truoc","nhan","khong","huy"):
            if lex in w:
                c["has_"+lex]+=1
                c["via_has_"+lex]+=yes
    return {"count":n,"via_pct":round(100*c["via"]/n,2),
            "fragile_pct":round(100*c["fragile"]/n,2),
            "urgent_pct":round(100*c["urgent"]/n,2),
            "accent_pct":round(100*c["accents"]/n,2),
            "p_via_given_word":{k:round(c["via_has_"+k]/c["has_"+k],4)
                if c["has_"+k] else None for k in ("lay","truoc","nhan","khong","huy")}}
def model_test(rows,path):
    model=NeuralMissionParser.load(path)
    for m in model.nets:m.cpu().eval()
    with torch.inference_mode():
        answer=module.inspect(rows,path.name,model)
    del model
    # Avoid writing thousands of repeated mistake excerpts in per-round files.
    answer["mistakes_total_in_top30"]=len(answer.pop("mistakes"))
    return answer
def get_training_observation():
    metrics=LATEST/"training_metrics.jsonl"
    if not metrics.exists():return {"status":"no_training_metrics"}
    last=None;ends=[]
    with metrics.open(encoding="utf-8") as f:
        for x in f:
            try:r=json.loads(x)
            except (ValueError,UnicodeError):continue
            last=r
            if r.get("event")=="EPOCH_COMPLETE":ends.append(r)
    if not last:return {"status":"empty_metrics"}
    z={"last_event":last.get("event"),"seed_index":last.get("seed_index"),
       "epoch":last.get("epoch"),"updates":last.get("updates"),
       "complete_epoch_count":len(ends),"time_unix":last.get("time_unix")}
    if ends:
        l=ends[-1]
        z["latest_complete"]={"seed_index":l["seed_index"],"epoch":l["epoch"],
           "train_loss":round(l["train_loss"],5),
           "synthetic_holdout_loss":round(l["overfit"]["synthetic_holdout_mean_loss"],5),
           "synthetic_holdout_all":{k:round(v["exact"]["all"],4)
                                    for k,v in l["synthetic_heldout"].items()}}
    return z
def do_cycle(i):
    rseed=2026101100+i*10007
    pairs=module.examples(num_pairs=240,seed=rseed)
    out=WORK/f"experiments/counterfactual_round_{i:03d}.jsonl"
    with out.open("x",encoding="utf-8",newline="\n") as f:
        for row in pairs:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    a=model_test(pairs,MODEL_PATHS[0])
    b=model_test(pairs,MODEL_PATHS[1])
    mode=("v2","weighted","hard")[i%3]
    holdout=bool(i%2)
    synth=gen.generate(900,seed=rseed+4321,holdout=holdout,mode=mode)
    stats=probe_stats(synth)
    improvement=b["via_acc"]-a["via_acc"]
    pair_improvement=b["both_counterfactual_correct_fraction"]-a["both_counterfactual_correct_fraction"]
    # Preserve exact evidence and distinction between measured vs hypothesized.
    report={
       "round":i,"started_utc":utc(),"hours_since_start":round((time.time()-START_TS)/3600,3),
       "test":"independent seeded paired counterfactual via + synthetic distribution",
       "seed":rseed,"synthetic_mode":mode,"holdout_phrase_bank":holdout,
       "paired_samples":len(pairs)//2,
       "sample_file":str(out),"sample_sha256":digest(out),
       "baseline_checkpoints":[str(p) for p in MODEL_PATHS],
       "baseline_metrics":{a["model"]:a,b["model"]:b},
       "via_accuracy_change_e2_minus_e1":round(improvement,5),
       "both_pair_accuracy_change_e2_minus_e1":round(pair_improvement,5),
       "synthetic_probe_distribution":stats,
       "training_observation_read_only":get_training_observation(),
       "hypothesis":"If via-positive cases lag while cancelled cases recover, data needs contrastive pickup/cancellation with balanced labels.",
       "next_action":"Continue varied seeded counterfactual groups and re-estimate template-shift, without altering running training.",
       "limitations":["Counterfactual synthetic diagnostic only, not natural blind test",
                      "Old epoch1/2 checkpoints, not ongoing 5-epoch checkpoint",
                      "Only target role named; not all spatial instructions tested",
                      "Same six fixed pattern families varied across samples"]}
    dest=WORK/f"checkpoints/round_{i:03d}.json"
    save_new(dest,report)
    print("RESEARCH_CHECKPOINT",str(dest),"e1",a["via_acc"],"e2",b["via_acc"],
          "pair_e1",a["both_counterfactual_correct_fraction"],
          "pair_e2",b["both_counterfactual_correct_fraction"],
          "fresh_probe",mode,stats["via_pct"],flush=True)
print("RESEARCH_LOOP_BEGIN",utc(),"CYCLES",MINIMUM_ROUNDS,
      "min_wall_clock_hours",SCHEDULE_INTERVAL_S*(MINIMUM_ROUNDS-1)/3600,flush=True)
for i in range(MINIMUM_ROUNDS):
    # Pause between investigations, with diagnostic work/checkpoint every 10 mins.
    due=START_TS+i*SCHEDULE_INTERVAL_S
    sleep=max(0.0,due-time.time())
    if sleep:time.sleep(sleep)
    if (WORK/"STOP_RESEARCH").exists():
        print("RESEARCH_LOOP_EXPLICIT_STOP_FLAG",utc(),flush=True)
        break
    try:do_cycle(i)
    except Exception as ex:
        fail={"round":i,"timestamp":utc(),"error_type":type(ex).__name__,
              "error":str(ex),"traceback":traceback.format_exc()}
        save_new(WORK/f"checkpoints/round_{i:03d}_ERROR.json",fail)
        print("RESEARCH_ROUND_ERROR",i,str(ex),flush=True)
else:
    summary={
       "status":"MINIMUM_FOUR_HOURS_COMPLETED",
       "started_utc":datetime.datetime.fromtimestamp(START_TS,datetime.timezone.utc).isoformat(),
       "ended_utc":utc(),"elapsed_seconds":round(time.time()-START_TS,1),
       "checkpoints":len(list((WORK/"checkpoints").glob("round_???.json"))),
       "rounds_expected":MINIMUM_ROUNDS,"source_run_unmodified":True,
       "further_autonomous_research":"Separate hourly ChatGPT task can continue subsequent investigations"}
    save_new(WORK/"checkpoints/four_hour_cycle_completion.json",summary)
    print("RESEARCH_LOOP_FOUR_HOURS_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)
