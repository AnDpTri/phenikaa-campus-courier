"""Read-only early-stopping ADVISOR for V5-SF200-Scratch.

Only analyzes epoch-complete JSONL metrics and checkpoint existence.
NEVER sends stop signals, writes training files, or accesses official validation/test.
Thresholds were defined before first epoch completes; synthetic probes become
a tuning/development set if used to make stop decisions. Reserve untouched
blind test for final performance claims.
"""
from collections import defaultdict
import json
from pathlib import Path

RUN=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch")
SRC=RUN/"training_metrics.jsonl"
MIN_EPOCH=8
PATIENCE=3
MIN_SCORE_GAIN=0.005   # absolute 0.5 percentage point
MIN_LOSS_GAIN=0.02     # summed eight-head CE loss
WEIGHTS={"v2":0.20,"weighted":0.35,"hard":0.45}

def get_score(e):
    return sum(WEIGHTS[k]*e["synthetic_heldout"][k]["exact"]["all"] for k in WEIGHTS)
def get_loss(e):
    return sum(WEIGHTS[k]*e["synthetic_heldout"][k]["sum_loss"] for k in WEIGHTS)

rows=defaultdict(dict)
if SRC.exists():
    with SRC.open("r",encoding="utf-8") as f:
        for line in f:
            try:r=json.loads(line)
            except (ValueError,TypeError):continue
            if r.get("event")=="EPOCH_COMPLETE":
                rows[int(r["seed_index"])][int(r["epoch"])]=r

results=[]
for seed_index in range(3):
    ordered=sorted(rows[seed_index].items())
    if not ordered:
        results.append({"seed_index":seed_index,"status":"AWAIT_FIRST_EPOCH",
             "action":"CONTINUE","reason":"No completed epoch or diagnostic data"})
        continue
    best_score=-float("inf")
    best_loss=float("inf")
    best_score_epoch=None
    best_loss_epoch=None
    stale=0
    timeline=[]
    for epoch,r in ordered:
        s=get_score(r)
        loss=get_loss(r)
        score_changed=s>=best_score+MIN_SCORE_GAIN
        loss_changed=loss<=best_loss-MIN_LOSS_GAIN
        if score_changed or loss_changed:stale=0
        else:stale+=1
        if s>best_score:best_score=s;best_score_epoch=epoch
        if loss<best_loss:best_loss=loss;best_loss_epoch=epoch
        timeline.append({"epoch":epoch,"score":round(s,5),"loss":round(loss,5),
                        "stale_epochs":stale,"score_improved":score_changed,
                        "loss_improved":loss_changed,
                        "overfit_flag":bool(r.get("overfit",{}).get("overfit_warning"))})
    last=timeline[-1]
    # Requires 3 COMPLETED consecutive stagnant epochs, and cannot recommend
    # before epoch 8. The active implementation cannot gracefully skip only
    # one seed: NEVER automatically kill the ongoing run.
    flag=last["epoch"]>=MIN_EPOCH and stale>=PATIENCE
    status="EARLY_STOP_REVIEW" if flag else ("WARMUP_MONITOR" if last["epoch"]<MIN_EPOCH else "LEARNING_OR_PATIENCE")
    ckpt=RUN/f"scratch_s{seed_index}_epoch{last['epoch']:02d}.pt"
    results.append({"seed_index":seed_index,"status":status,
        "action":"RECOMMEND_MANUAL_REVIEW" if flag else "CONTINUE",
        "completed_epochs":len(ordered),"latest_epoch":last["epoch"],
        "latest_composite_score":last["score"],"latest_weighted_holdout_loss":last["loss"],
        "best_composite_score":round(best_score,5),"best_score_epoch":best_score_epoch,
        "best_weighted_holdout_loss":round(best_loss,5),"best_loss_epoch":best_loss_epoch,
        "patience_used":stale,"patience_limit":PATIENCE,
        "checkpoint_exists":ckpt.exists(),
        "recent_epoch_history":timeline[-5:],
        "explanation":("Plateau on both synthetic accuracy and loss; investigate manual safe stop, "
                        "do not treat probe-guided checkpoint selection as blind testing.") if flag
                      else "Continue; insufficient consecutive lack of improvement or under minimum epochs."})
print(json.dumps({"advisory_only":True,"will_modify_or_stop_process":False,
  "no_official_validation":True,
  "config":{"min_epoch":MIN_EPOCH,"patience":PATIENCE,
            "min_composite_score_gain":MIN_SCORE_GAIN,
            "min_loss_reduction":MIN_LOSS_GAIN,"weights":WEIGHTS},
  "seeds":results},ensure_ascii=False,indent=2))
