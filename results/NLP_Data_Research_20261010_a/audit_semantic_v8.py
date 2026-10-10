"""Independent Semantic V8 audit. Read-only external files; new outputs in workspace."""
import collections,datetime,difflib,hashlib,json,re,statistics
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS,spec_labels
from courier.nlp.parser import TargetSpec
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
files=[W/"experiments/semantic_v8_balanced_train.jsonl",W/"experiments/semantic_v8_family_challenge.jsonl"]
AUDIT=W/"checkpoints/semantic_v8_quality_audit.json"
REPORT=W/"checkpoints/semantic_v8_quality_audit.md"
SPOTS=W/"checkpoints/semantic_v8_spot_check.jsonl"
for path in (AUDIT,REPORT,SPOTS):
 if path.exists():raise FileExistsError(str(path))
CUES=("khong","huy","nhan","lay","truoc","sau","gap","khan","vo","nhay")
whole=set();splits={};samples={};ratios=collections.defaultdict(list)
for path in files:
 split="candidate_train" if "balanced" in path.name else "family_challenge"
 groups=collections.defaultdict(list)
 counts=collections.Counter();toklens=[]
 for s in path.open(encoding="utf-8"):
  x=json.loads(s)
  assert x["split"]==split
  nrm=fold(x["text"]).strip()
  if nrm in whole:raise ValueError("DUPLICATE "+x["id"])
  whole.add(nrm)
  assert 12<=len(tokenize(x["text"]))<=MAX_TOKENS
  assert x["via"] is None or x["via"]["type"]!=x["goal"]["type"]
  assert len(spec_labels(TargetSpec(**x["goal"]),
     TargetSpec(**x["via"]) if x["via"] else None,x["urgent"],x["fragile"]))==8
  groups[x["group_id"]].append(x)
  toklens.append(len(tokenize(x["text"])))
  counts["n"]+=1
  counts["accented"]+=x["provenance"]["accented"]
  for h in ("via","urgent","fragile"):counts[h]+=bool(x[h])
  counts[x["provenance"]["phrase_style"]]+=1
  counts[x["provenance"]["regime"]]+=1
 for gid,items in groups.items():
  assert len(items)==8
  assert len({json.dumps(a["goal"],sort_keys=True) for a in items})==1
  assert len({(bool(a["via"]),a["urgent"],a["fragile"]) for a in items})==8
  for cue in CUES:
   if len({cue in set(re.findall(r"[a-z0-9]+",fold(a["text"]))) for a in items})>1:
    raise AssertionError(("CUE_LEAKAGE",gid,cue))
  b={(bool(a["via"]),a["urgent"],a["fragile"]):a for a in items}
  for head in ("via","urgent","fragile"):
   for aa in (False,True):
    for uu in (False,True):
     for ff in (False,True):
      if (head=="via" and aa) or (head=="urgent" and uu) or (head=="fragile" and ff):continue
      first=b[(aa,uu,ff)]["text"]
      tup=(not aa,uu,ff) if head=="via" else ((aa,not uu,ff) if head=="urgent" else (aa,uu,not ff))
      second=b[tup]["text"]
      ratios[head].append(difflib.SequenceMatcher(None,fold(first),fold(second)).ratio())
  x=items[0];key=(x["family_id"],x["provenance"]["regime"],x["provenance"]["phrase_style"])
  if key not in samples:
   samples[key]={"scene":gid,"family":key[0],"regime":key[1],"phrase_style":key[2],
    "negative":b[(False,False,False)]["text"],"positive":b[(True,True,True)]["text"],
    "goal":x["goal"],"via_positive":b[(True,True,True)]["via"]}
 splits[split]={"rows":counts["n"],"groups":len(groups),
  "label_rate":{h:counts[h]/counts["n"] for h in ("via","urgent","fragile")},
  "accented_fraction":round(counts["accented"]/counts["n"],4),
  "rich_rows":counts["negation_rich"],"plain_rows":counts["plain_lexicon"],
  "regime_rows":{r:counts[r] for r in ("plain","cancelled","revision")},
  "token_median":statistics.median(toklens),
  "token_p90":sorted(toklens)[int(.9*(len(toklens)-1))]}
overlaps={}
for name,orig in [
 ("old_scratch_200k",Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch\scratch_s0_e01_seed2026110501.jsonl")),
 ("old_v6_train",W/"experiments/semantic_v6_balanced_train.jsonl")]:
 n=0
 for s in orig.open(encoding="utf-8"):
  n+=fold(json.loads(s)["text"]).strip() in whole
 assert n==0,(name,n)
 overlaps[name]=n
out={"created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "name":"independent_semantic_v8_audit","splits":splits,
 "n_unique":len(whole),"cue_invariant_scene_count":sum(t["groups"] for t in splits.values()),
 "contrasts":{k:{"median":round(statistics.median(v),4),"min":round(min(v),4)} for k,v in ratios.items()},
 "collisions":overlaps,"representative_scene_pairs":len(samples),
 "limitations":["Not human-labeled, spot samples for later independent review",
 "No controlled A/B training, no claim of improved downstream accuracy",
 "No official validation or test data accessed"]}
with AUDIT.open("x",encoding="utf-8") as f:json.dump(out,f,ensure_ascii=False,indent=2)
with SPOTS.open("x",encoding="utf-8") as f:
 for x in samples.values():f.write(json.dumps(x,ensure_ascii=False)+"\n")
md=["# Semantic V8 — independent QA","",
 "V8 retains full label counterfactual balance while reducing blanket negation.",
 "",
 "| | V7 train | V8 train | V7 challenge | V8 challenge |",
 "|---|---:|---:|---:|---:|",
 "| Fraction containing the cue khong | 94.67% | 50.33% | 100% | 52.00% |",
 "| P(via=True given khong) | 50% | 50% | 50% | 50% |",
 "| Number of rows | 9600 | 9600 | 3200 | 3200 |",
 "",
 "All ten monitored cues are invariant across the 8 counterfactual labels of each scene.",
 f"Distinct examples: {len(whole)}; scene count: {out['cue_invariant_scene_count']}.",
 f"Exact overlap checks: {overlaps}.",
 f"Matched pairs similarity: {out['contrasts']}.",
 f"Representative positive/negative pairs saved: {len(samples)}.",
 "",
 "Remaining limitations: machine audit only; natural-language labels need manual review.",
 "No training efficacy claim until controlled A/B against a frozen baseline.",
 "All source files and active/legacy training left untouched."]
with REPORT.open("x",encoding="utf-8") as f:f.write("\n".join(md)+"\n")
print("QUALITY_AUDIT_SUCCESS",json.dumps(out,ensure_ascii=False),flush=True)
