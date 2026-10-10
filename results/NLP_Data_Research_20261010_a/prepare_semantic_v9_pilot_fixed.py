"""Semantic V9 pilot: repair fragile property assertions, calibrate real-like
label / accent / length distribution. ALL writes within research workspace.
Source V8 and real train read-only. No official val/test.
"""
from __future__ import annotations
import collections,hashlib,json,random,re,datetime,math,statistics,copy
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS,spec_labels
from courier.nlp.parser import TargetSpec
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
ORIG=W/"experiments/semantic_v8_balanced_train.jsonl"
OUTFULL=W/"experiments/semantic_v9_label_repaired_full.jsonl"
OUTPILOT=W/"experiments/semantic_v9_real_like_pilot.jsonl"
MAN=W/"checkpoints/semantic_v9_repair_and_sampling_manifest.json"
for p in (OUTFULL,OUTPILOT,MAN):
 if p.exists():raise FileExistsError(str(p))
# Property assertions, not just transport-policy states.
PATCHES=[
 ("Yêu cầu bảo vệ đồ dễ bể đã hết hiệu lực.",
  "Yêu cầu bảo vệ đồ dễ bể đã hết hiệu lực vì thực tế kiện là hàng bền, chịu va đập tốt."),
 ("Yêu cầu bảo vệ đồ dễ bể vẫn còn hiệu lực.",
  "Yêu cầu bảo vệ đồ dễ bể vẫn còn hiệu lực vì kiện thực sự chứa vật dễ bể."),
 ("Quy định vận chuyển đồ dễ bể đã hết hiệu lực.",
  "Quy định vận chuyển đồ dễ bể đã hết hiệu lực vì kiện được xác nhận là hàng bền, chịu rung tốt."),
 ("Quy định vận chuyển đồ dễ bể vẫn còn hiệu lực.",
  "Quy định vận chuyển đồ dễ bể vẫn còn hiệu lực vì kiện thực sự chứa vật dễ bể."),
 ("Yêu cầu chống va đập dành cho hàng dễ vỡ đã hết hiệu lực.",
  "Yêu cầu chống va đập dành cho hàng dễ vỡ đã hết hiệu lực vì kiện là hàng bền, chịu va đập tốt."),
 ("Yêu cầu chống va đập dành cho hàng dễ vỡ vẫn hiệu lực.",
  "Yêu cầu chống va đập dành cho hàng dễ vỡ vẫn hiệu lực vì kiện thực sự chứa đồ dễ vỡ."),
 ("Nhãn dễ vỡ đã được gỡ, robot vận chuyển bình thường.",
  "Nhãn dễ vỡ đã được gỡ vì kiện là hàng bền, robot vận chuyển bình thường."),
 ("Trạng thái nhạy va chạm đã gỡ, vận chuyển bình thường.",
  "Trạng thái nhạy va chạm đã gỡ vì kiện chịu rung tốt, vận chuyển bình thường."),
 ("Trạng thái nhạy va chạm đã xác nhận, phải nâng nhẹ.",
  "Trạng thái nhạy va chạm đã xác nhận vì kiện có vật dễ hỏng khi rung, phải nâng nhẹ."),
 ("Nhãn dễ vỡ đã được xác nhận, robot cần kê đệm.",
  "Nhãn dễ vỡ đã được xác nhận vì kiện chứa vật dễ vỡ, robot cần kê đệm."),
]
rng=random.Random(2026101099)
bybin=collections.defaultdict(list);allrows=[];patched=collections.Counter()
for line in ORIG.open(encoding="utf-8"):
 z=json.loads(line)
 txt=z["text"]
 accented=z["provenance"]["accented"]
 hit=0
 for old,new in PATCHES:
  o=old if accented else fold(old)
  nn=new if accented else fold(new)
  if o in txt:
   txt=txt.replace(o,nn)
   patched[(old,bool(z["fragile"]))]+=1
   hit+=1
 if hit>1:raise RuntimeError(("MULTIPLE_PATCH",z["id"],hit))
 z["text"]=txt
 z["id"]=z["id"].replace("sf8_","sf9_")
 z["provenance"]["source"]="v8_label_grounding_repair"
 z["provenance"]["v9_patch_applied"]=bool(hit)
 if len(tokenize(txt))>MAX_TOKENS:raise ValueError(("TOKENS_AFTER_PATCH",z["id"]))
 bybin[(bool(z["via"]),z["urgent"],z["fragile"],accented)].append(z)
 allrows.append(z)
assert len(allrows)==9600
# Exact target probabilities estimated from real 2k train. 16 strata:
# via x urgent x fragile x accented; quotas sum exactly to 3200.
probs={"via":.327,"urgent":.382,"fragile":.392,"accented":.706}
N=3200
weighted=[]
for key,lst in bybin.items():
 p=1.
 for val,kind in zip(key,("via","urgent","fragile","accented")):
  p*=probs[kind] if val else 1-probs[kind]
 weighted.append((key,p*N,lst))
assert len(weighted)==16
initial={k:int(math.floor(w)) for k,w,_ in weighted}
remaining=N-sum(initial.values())
for k,ww,_ in sorted(weighted,key=lambda z:-(z[1]-math.floor(z[1])))[:remaining]:initial[k]+=1
chosen=[]
for k,ww,lst in weighted:
 quota=initial[k]
 if quota>len(lst):raise RuntimeError(("INSUFFICIENT_LABEL_CLASS",k,quota,len(lst)))
 chosen.extend(copy.deepcopy(z) for z in rng.sample(lst,quota))
rng.shuffle(chosen)
# Clauses are exactly four sentences in V8. The clause order is encoded
# and can be unpacked deterministically without guessing semantic labels.
def split_clauses(z):
 arr=re.split(r'(?<=\.)\s+',z["text"].strip())
 assert len(arr)==4,(z["id"],len(arr))
 order=z["provenance"]["order"]
 original={0:arr,1:[arr[1],arr[0],arr[2],arr[3]],
           2:[arr[1],arr[2],arr[0],arr[3]],
           3:[arr[0],arr[2],arr[3],arr[1]]}[order]
 return original
full_text={z["id"]:z["text"] for z in chosen}
notes=collections.Counter()
for z in chosen:
 original=split_clauses(z)
 present=[True,True,True,True]
 # Explicit negative statements are not always present in real requests.
 if z["via"] is None and rng.random()<.55:present[1]=False;notes["via_absent"]+=1
 if not z["urgent"] and rng.random()<.72:present[2]=False;notes["urgent_absent"]+=1
 if not z["fragile"] and rng.random()<.72:present[3]=False;notes["fragile_absent"]+=1
 # Also omit plain redundant wording about no-via when the other flags are explicit.
 assert present[0]
 kept=[part for part,yes in zip(original,present) if yes]
 style=z["provenance"]["order"]
 if style==1 and len(kept)>1:kept=[kept[1],kept[0],*kept[2:]]
 elif style==2 and len(kept)>2:kept=[kept[2],kept[0],kept[1],*kept[3:]]
 elif style==3 and len(kept)>3:kept=[kept[0],kept[3],kept[1],kept[2]]
 z["text"]=" ".join(kept)
 z["provenance"]["v9_pilot_omitted_fields"]=[k for k,yes in zip(("goal","via","urgent","fragile"),present) if not yes]
 assert 5<=len(tokenize(z["text"]))<=MAX_TOKENS,(z["id"],len(tokenize(z["text"])))
 assert len(spec_labels(TargetSpec(**z["goal"]),
   TargetSpec(**z["via"]) if z["via"] is not None else None,z["urgent"],z["fragile"]))==8
# Avoid duplicate texts (possible when different examples share same sentence content).
unique=set()
for z in chosen:
 t=fold(z["text"]).strip()
 if t in unique:
  z["text"]=full_text[z["id"]]
  z["provenance"]["v9_pilot_omitted_fields"]=[]
  t=fold(z["text"]).strip()
  if t in unique:raise RuntimeError(("NORMALIZED_DUPLICATE_AFTER_REPAIR",z["id"]))
 unique.add(t)
# Full label-repaired long-form pool and distribution-weighted pilot.
with OUTFULL.open("x",encoding="utf-8",newline="\n") as f:
 for z in allrows:f.write(json.dumps(z,ensure_ascii=False)+"\n")
with OUTPILOT.open("x",encoding="utf-8",newline="\n") as f:
 for z in chosen:f.write(json.dumps(z,ensure_ascii=False)+"\n")
stats=collections.Counter();lens=[]
for z in chosen:
 stats["n"]+=1
 for k in ("via","urgent","fragile"):
  stats[k]+=bool(z[k])
 stats["accented"]+=z["provenance"]["accented"]
 lens.append(len(tokenize(z["text"])))
pstats={"n":stats["n"],"label_rates":{k:round(stats[k]/stats["n"],4)
 for k in ("via","urgent","fragile","accented")},
 "median_tokens":statistics.median(lens),"p90_tokens":sorted(lens)[int(.9*(len(lens)-1))],
 "ge70_tokens_fraction":round(sum(x>=70 for x in lens)/len(lens),4),
 "le25_tokens_fraction":round(sum(x<=25 for x in lens)/len(lens),4),
 "minimum_tokens":min(lens),"max_tokens":max(lens)}
manifest={"name":"semantic_v9_property_grounded_real_like_pilot",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "seed":2026101099,"target_real_distributions":probs,
 "source":str(ORIG),"patch_affected_rows":sum(patched.values()),
 "patched_phrase_counts":[{"text":k[0],"label_fragile":k[1],"n":v} for k,v in sorted(patched.items())],
 "full":{"n":len(allrows),"sha256":hashlib.sha256(OUTFULL.read_bytes()).hexdigest(),"file":str(OUTFULL)},
 "pilot":{**pstats,"sha256":hashlib.sha256(OUTPILOT.read_bytes()).hexdigest(),"file":str(OUTPILOT),
          "omissions":dict(notes)},
 "qa":["explicit package property rather than handling policy on repaired samples",
 "quota sampling by real train labels and accent","optional false clauses are omitted",
 "no normalized collisions in pilot","typed labels valid and within MAX_TOKENS"],
 "limits":["Real train marginals only, not full joint distribution",
 "No independent human audit","All examples still composed from V8 templates",
 "Do not use family challenge for training; heldout challenge unchanged",
 "No model performance claim until A/B"]}
with MAN.open("x",encoding="utf-8") as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print("V9_READY",json.dumps({"patches":sum(patched.values()),"pilot":pstats,"omissions":dict(notes),
 "manifest":str(MAN)},ensure_ascii=False),flush=True)
