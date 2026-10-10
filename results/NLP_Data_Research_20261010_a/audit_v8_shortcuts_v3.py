"""Measure remaining V8 lexical shortcuts (unigrams, 2grams) after 10-cue invariant QA.
Read only data; create new release audit JSON inside workspace.
"""
import collections,datetime,json,math,re
from pathlib import Path
from courier.nlp.text import fold
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
OUTPUT=W/"checkpoints/semantic_v8_unmonitored_shortcuts_v3.json"
if OUTPUT.exists():raise FileExistsError(str(OUTPUT))
def tokens(s):
 return re.findall(r"[a-z0-9]+",fold(s))
def audit(path):
 rows=[]
 with path.open(encoding="utf-8") as f:
  rows=[json.loads(x) for x in f]
 N=len(rows)
 counts={"via":collections.defaultdict(lambda:[0,0]),
         "urgent":collections.defaultdict(lambda:[0,0]),
         "fragile":collections.defaultdict(lambda:[0,0])}
 for r in rows:
  tok=tokens(r["text"])
  un=set(tok)
  bi={tok[i]+" "+tok[i+1] for i in range(len(tok)-1)}
  grams={"unigram":un,"bigram":bi}
  for head in ("via","urgent","fragile"):
   pos=bool(r[head])
   for typ,vals in grams.items():
    for t in vals:
     n=counts[head][(typ,t)];n[0]+=1;n[1]+=pos
 report={}
 for head,count in counts.items():
  res=[]
  for (typ,t),(support,positive) in count.items():
   if support<max(64,int(.03*N)) or support>.97*N:continue
   p=positive/support
   risk=abs(p-.5)
   if risk<.001:continue
   res.append({"gram":typ,"token":t,"support":support,"fraction":round(support/N,4),
      "p_positive":round(p,4),"deviation_pp":round(risk*100,2)})
  ranked=sorted(res,key=lambda x:(-x["deviation_pp"],-x["support"],x["token"]))
  reliable=sorted((z for z in ranked if z["support"]>=.1*N),key=lambda x:(-x["deviation_pp"],-x["support"]))
  report[head]={"large_support_shortcuts_top20":reliable[:20],
               "medium_support_shortcuts_top20":ranked[:20],
               "num_unigram_or_bigram_shortcuts_gt_20pp":sum(x["deviation_pp"]>=20 for x in ranked),
               "num_shortcuts_gt_10pp":sum(x["deviation_pp"]>=10 for x in ranked)}
 return {"n":N,"labels":report}
datasets={
 "candidate_train":W/"experiments/semantic_v8_balanced_train.jsonl",
 "family_challenge":W/"experiments/semantic_v8_family_challenge.jsonl"}
result={"name":"v8_post_10cue_lexical_leakage_v3",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "logic":"Document-frequency cue at 3%+ support, max97%; P(label=True|unigram/bigram) vs balanced 50% overall",
 "sets":{k:audit(v) for k,v in datasets.items()},
 "warnings":["Association is not causality; sometimes label-bearing phrases must correlate with correct meaning",
 "Families differ in wording; challenge had only two families",
 "10 invariant cues do not prevent shortcuts involving other ngrams"]}
with OUTPUT.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
for split,z in result["sets"].items():
 for head,info in z["labels"].items():
  print("RANK",split,head, ">=20pp",info["num_unigram_or_bigram_shortcuts_gt_20pp"],
        "TOP",json.dumps(info["large_support_shortcuts_top20"][:7],ensure_ascii=False),flush=True)
print("SAVED",OUTPUT,flush=True)
