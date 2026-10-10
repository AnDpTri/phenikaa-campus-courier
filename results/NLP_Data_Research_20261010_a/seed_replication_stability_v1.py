"""Independent-seed replication audit. Read-only source files."""
import json,statistics,datetime
from pathlib import Path
w=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
files=sorted((w/"checkpoints").glob("round_???.json"))
assert len(files)>=6
rows=[json.loads(p.read_text(encoding="utf-8")) for p in files]
a=[];b=[];pairs=[]
for x in rows:
    z=x["baseline_metrics"]
    e1=z["scratch_s0_epoch01.pt"]
    e2=z["scratch_s0_epoch02.pt"]
    a.append(e2["cases"]["active_via"]["correct_via"]-e1["cases"]["active_via"]["correct_via"])
    b.append(e2["cases"]["cancelled_via"]["correct_via"]-e1["cases"]["cancelled_via"]["correct_via"])
    pairs.append(e2["both_counterfactual_correct_fraction"]-e1["both_counterfactual_correct_fraction"])
def summarize(xs):
    n=len(xs);mu=statistics.mean(xs);sd=statistics.stdev(xs)
    return {"n_rounds":n,"mean":round(mu,5),"min":round(min(xs),5),"max":round(max(xs),5),"sd":round(sd,5),
            "positive_rounds":sum(z>0 for z in xs),"negative_rounds":sum(z<0 for z in xs),
            "all_round_values":[round(z,5) for z in xs]}
out={"study":"seed_replication_stability_v1",
     "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
     "sources":[str(p) for p in files],"n_pairs_per_round":[x["paired_samples"] for x in rows],
     "delta_active":summarize(a),"delta_cancelled":summarize(b),
     "delta_both_pair":summarize(pairs),
     "hypothesis":"Improvement is cancellation-specific rather than balanced across positive/negative via.",
     "limitations":["Repeated six fixed templates across rounds","Old epoch1/2 models, synthetic diagnostics only",
                    "Rounds have independent generator seeds but not independent template families",
                    "No official validation/test used"],
     "next_action":"Design disjoint template-family counterfactuals and quantify cancellation versus active-via tradeoff."}
dest=w/"checkpoints/seed_replication_stability_v1.json"
with dest.open("x",encoding="utf-8") as f:json.dump(out,f,ensure_ascii=False,indent=2)
print("SAVED",dest)
for k in ("delta_active","delta_cancelled","delta_both_pair"):print(k,out[k])
