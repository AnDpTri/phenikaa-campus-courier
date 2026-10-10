"""Audit distribution and conditional shortcuts. READ-ONLY inputs, create-only output."""
from __future__ import annotations
import collections, datetime, hashlib, importlib.util, json, math, re, statistics, sys, time
from pathlib import Path
from courier.nlp.text import fold, tokenize
ROOT=Path(r"D:\phenikaa")
WORK=ROOT/"results/NLP_Data_Research_20261010_a"
SRC=ROOT/"results/NLP_V5_SF200_Scratch"
OLD=SRC/"scratch_s0_e01_seed2026110501.jsonl"
ROLES=SRC/"scratch_s0_e01_roles.jsonl"
SPEC=importlib.util.spec_from_file_location("research_readonly_generator",SRC/"synth_weighted_v4.py")
gen=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=gen;SPEC.loader.exec_module(gen)
def bin_key(rec):
    g=rec.get("goal") or {}
    v=rec.get("via") or {}
    return (str(bool(v)),str(rec.get("urgent")),str(rec.get("fragile")),g.get("ref","none") or "none")
def flags(txt):
    t=fold(txt).lower()
    w=set(t.split())
    return {"lay_word":int("lay" in w),"truoc_word":int("truoc" in w),
            "nhan_word":int("nhan" in w),"no_word":int("khong" in w),
            "huy_word":int("huy" in w),"fragile_lex":int(("de vo" in t or "de be" in t or "va dap" in t)),
            "urgent_lex":int(("gap" in w or "khan" in w or "hoa toc" in t)),
            "accents":int(any(ord(c)>127 for c in txt)),
            "n_tokens":len(tokenize(txt))}
def evaluate_records(records,label):
    n=0
    num=collections.Counter()
    lex=collections.Counter()
    via_lex=collections.Counter()
    negative_lex=collections.Counter()
    joint=collections.Counter()
    oov_token=collections.Counter()
    prefix=collections.Counter()
    lens=[]
    for r in records:
        n+=1;t=r["text"];x=flags(t);yes=r.get("via") is not None
        num["via"]+=yes;num["urgent"]+=r.get("urgent") is True
        num["fragile"]+=r.get("fragile") is True
        num["accented"]+=x["accents"]
        num["spatial_goal"]+=(r.get("goal") or {}).get("ref") is not None
        num["spatial_via"]+=(r.get("via") or {}).get("ref") is not None
        lens.append(x["n_tokens"])
        z=tuple(fold(t).split()[:8]);prefix[z]+=1
        for key,value in x.items():
            if key!="n_tokens":
                lex[key]+=value
                if value:via_lex[key]+=yes;negative_lex[key]+=not yes
        joint[(yes,r.get("fragile") is True,r.get("urgent") is True)]+=1
        for w in set(fold(t).split()):oov_token[w]+=1
    out={"dataset":label,"n":n,"label_positive_fraction":{k:round(v/n,4) for k,v in num.items()},
         "lexical_fraction":{k:round(v/n,4) for k,v in lex.items()},
         "p_via_given_lex":{k:round(v/lex[k],4) if lex[k] else None for k,v in via_lex.items()},
         "p_via_without_lex":{k:round((num["via"]-via_lex[k])/(n-lex[k]),4) if n>lex[k] else None for k in lex},
         "token_p50":statistics.median(lens),"token_p90":sorted(lens)[int(.9*(len(lens)-1))],
         "prefix_distinct_rate":round(len(prefix)/n,4),
         "max_prefix_repeat":max(prefix.values()),
         "joint_flags_and_via":{str(k):v for k,v in joint.items()},
         "frequent_tokens":[(k,v) for k,v in oov_token.most_common(40)]}
    return out,set(oov_token),oov_token
def old_records():
    with OLD.open(encoding="utf-8") as f:
        for line in f:yield json.loads(line)
start=time.time()
train,train_vocab,train_freq=evaluate_records(old_records(),"train_s0_e1_200k")
provenance=collections.defaultdict(list)
with OLD.open(encoding="utf-8") as ff,ROLES.open(encoding="utf-8") as fr:
    for a,b in zip(ff,fr):
        x=json.loads(a);y=json.loads(b)
        q=y["recipe"]
        z=flags(x["text"])
        provenance[q].append((x["via"] is not None,x["fragile"],x["urgent"],z["n_tokens"],z["lay_word"],z["truoc_word"],z["accents"]))
recs={}
for k,v in provenance.items():
    n=len(v)
    recs[k]={"n":n,"via_rate":round(sum(x[0] for x in v)/n,4),
             "fragile_rate":round(sum(x[1] for x in v)/n,4),
             "urgent_rate":round(sum(x[2] for x in v)/n,4),
             "p50_tokens":statistics.median(x[3] for x in v),
             "with_lay":round(sum(x[4] for x in v)/n,4),
             "with_truoc":round(sum(x[5] for x in v)/n,4),
             "accented":round(sum(x[6] for x in v)/n,4)}
# Locked probes use the same seeds as the prior run; no official validation.
probes={}
for name,count,seed in (("v2",1500,2026111901),("weighted",1500,2026111902),("hard",750,2026111903)):
    samples=gen.generate(count,seed=seed,holdout=True,mode=name)
    xx=[{"text":x.text,"goal":{"ref":x.goal.ref} if x.goal else None,
        "via":{"ref":x.via.ref} if x.via else None,"urgent":x.urgent,"fragile":x.fragile} for x in samples]
    diag,vocab,freq=evaluate_records(xx,name+"_probe")
    unseen=vocab-train_vocab
    diag["vocab_oov_ratio"]=round(len(unseen)/len(vocab),4)
    diag["document_token_oov_fraction"]=round(sum(c for t,c in freq.items() if t in unseen)/sum(freq.values()),4)
    diag["oov_examples"]=sorted(unseen)[:60]
    probes[name]=diag
summary={"research":"corpus_distribution_audit_v1","generated_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "seconds":round(time.time()-start,1),
         "source_sha256":hashlib.sha256((SRC/"synth_weighted_v4.py").read_bytes()).hexdigest(),
         "data":train,"per_recipe":recs,"probes":probes,
         "interpretation_notes":[
            "Synthetic probe holds out 20% phrase-bank positions, not families/semantics",
            "A model can memorize generator structure despite unique sentences",
            "Large train-vs-probe loss gap may mix overconfidence with real distribution shift",
            "Probe diagnostic cannot establish real-life generalization or replace blind test"]}
out=WORK/"checkpoints/corpus_audit_v1.json"
with out.open("x",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print("AUDIT_COMPLETE",out,"SECONDS",summary["seconds"])
print("TRAIN",json.dumps({k:train[k] for k in ("n","label_positive_fraction","token_p50","token_p90","prefix_distinct_rate")},ensure_ascii=False))
print("RECIPES",json.dumps(recs,ensure_ascii=False))
for n,a in probes.items():
    print("PROBE",n,"labels",a["label_positive_fraction"],"oov",a["document_token_oov_fraction"],"p_via_lex",a["p_via_given_lex"])
