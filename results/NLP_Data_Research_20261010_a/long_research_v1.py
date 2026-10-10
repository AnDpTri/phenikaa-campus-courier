"""CPU-only four-hour lexical shortcut audit. External data is read-only."""
import collections,datetime,hashlib,json,math,random,re,time
from pathlib import Path
from courier.nlp.text import fold
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
C=W/"checkpoints"
SRC=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch")
DATA=SRC/"scratch_s0_e01_seed2026110501.jsonl"
assert W.is_dir() and C.is_dir()
def save(name,payload):
    payload["timestamp_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload["source_data"]=str(DATA)
    payload["scope"]="Read-only inputs; CPU only; synthetic diagnostics only"
    p=C/name
    with p.open("x",encoding="utf-8") as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    print("CHECKPOINT",p.name,flush=True)
def tokens(s):return set(re.findall(r"[a-z0-9]+",fold(s).lower()))
def yval(r,target):return int(r["via"] is not None) if target=="via" else int(r[target])
def audit(rows,target,cues):
    result={}
    for cue in cues:
        a=b=c=d=0
        for r in rows:
            x=cue in r["tokens"];y=yval(r,target)
            if x and y:a+=1
            elif x:b+=1
            elif y:c+=1
            else:d+=1
        result[cue]={"counts":[a,b,c,d],"p_pos_with":round(a/(a+b),4) if a+b else None,
                     "p_pos_without":round(c/(c+d),4) if c+d else None,
                     "risk_diff":round(a/(a+b)-c/(c+d),4) if a+b and c+d else None}
    return result
def load():
    rows=[]
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            x=json.loads(line)
            rows.append({"tokens":tokens(x["text"]),"via":x["via"],"urgent":x["urgent"],
                         "fragile":x["fragile"],"accent":any(ord(z)>127 for z in x["text"])})
    return rows

def probes():
    import importlib.util,sys
    sp=importlib.util.spec_from_file_location("research_probe_long_v1",SRC/"synth_weighted_v4.py")
    g=importlib.util.module_from_spec(sp);sys.modules[sp.name]=g;sp.loader.exec_module(g)
    out={}
    for mode,n,seed in (("v2",1500,2026111901),("weighted",1500,2026111902),("hard",750,2026111903)):
        xs=g.generate(n,seed=seed,holdout=True,mode=mode)
        out[mode]=[{"tokens":tokens(x.text),"via":x.via,"urgent":x.urgent,
                    "fragile":x.fragile,"accent":any(ord(z)>127 for z in x.text)} for x in xs]
    return out
def metric(data):
    return {"n":len(data),"via":round(sum(yval(r,"via") for r in data)/len(data),4),
            "fragile":round(sum(yval(r,"fragile") for r in data)/len(data),4),
            "urgent":round(sum(yval(r,"urgent") for r in data)/len(data),4),
            "accent":round(sum(r["accent"] for r in data)/len(data),4)}
def study():
    start=time.monotonic()
    save("long_research_start_v1.json",{"phase":"start","planned_minimum_hours":4.1,
        "input_sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "hypothesis":"Train recipe mixture induces lexical shortcuts and label-prior shift"})
    train=load()
    probe=probes()
    save("long_research_loaded_v1.json",{"phase":"loaded","train":metric(train),
         "probe":{k:metric(v) for k,v in probe.items()},"next":"Conditional cue/label shift"})
    cues=("lay","truoc","nhan","ghe","khong","huy","bo","nham","de","vo","be",
          "gap","khan","hoa","toc","mong","nhe","ngay","sau","di")
    i=0
    until=start+4*3600+10*60
    while time.monotonic()<until:
        t=time.monotonic()
        rng=random.Random(120000+i*3719)
        tr=rng.sample(train,min(18000,len(train)))
        hold={k:[rng.choice(v) for _ in range(min(600,len(v)))] for k,v in probe.items()}
        findings={}
        for target in ("via","fragile","urgent"):
            a=audit(tr,target,cues)
            h={k:audit(v,target,cues) for k,v in hold.items()}
            shifts=[]
            for mode,stat in h.items():
                for cue in cues:
                    x=a[cue]["p_pos_with"];y=stat[cue]["p_pos_with"]
                    if x is not None and y is not None and sum(stat[cue]["counts"][:2])>=20:
                        shifts.append({"probe":mode,"cue":cue,"train":x,"holdout":y,
                                       "delta":round(y-x,4),"probe_count":sum(stat[cue]["counts"][:2])})
            findings[target]={"train":a,"holdout":h,
                              "largest_conditional_shifts":sorted(shifts,key=lambda z:abs(z["delta"]),reverse=True)[:15]}
        save(f"research_cycle_{i:03d}_bootstrap.json",{"phase":"conditional_cue_bootstrap",
            "iteration":i,"random_seed":120000+i*3719,
            "seconds_elapsed":round(time.monotonic()-start,1),
            "train_sample":metric(tr),"probe_samples":{k:metric(v) for k,v in hold.items()},
            "metrics":findings,
            "hypothesis":"P(label|cue) differs by generator and phrase-bank split",
            "limitations":"Observational lexical statistics; cannot alone prove neural overfit",
            "next":"Compare repeated bootstrap distributions; design scoped negation and pickup minimal pairs"})
        i+=1
        if time.monotonic()>=until:break
        time.sleep(max(0,600-(time.monotonic()-t)))
    save("long_research_complete_v1.json",{"phase":"complete",
         "seconds_elapsed":round(time.monotonic()-start,1),"cycles":i,
         "next":"Summarize stable shortcut findings; prototype counterfactual generator inside workspace only"})
if __name__=="__main__":study()
