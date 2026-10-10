"""Independent A/B reproduction and fresh-seed generalization audit on CPU.
Read-only original train/generator, read-only model A/B checkpoints.
Only create files inside research workspace. No official val or test.
"""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,math,random,statistics,sys,time
from pathlib import Path
import torch
from courier.nlp.neural import NeuralMissionParser
from courier.nlp.parser import TargetSpec
from courier.nlp.text import fold
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
ROOT=Path(r"D:\phenikaa")
SRC=ROOT/"results/NLP_V5_SF200_Scratch/synth_weighted_v4.py"
PATHS={"A":W/"experiments/v9_pilot_A_continuation.pt",
 "B":W/"experiments/v9_pilot_B_continuation.pt"}
OUT=W/"checkpoints/v9_independent_fresh_seed_eval_v2.json"
MD=W/"checkpoints/v9_independent_fresh_seed_eval_v2.md"
ERRORS=W/"checkpoints/v9_independent_error_examples_v2.jsonl"
for p in [OUT,MD,ERRORS]:
 if p.exists():raise FileExistsError("Refuse existing "+str(p))
torch.set_num_threads(3)
t0=time.time()
sp=importlib.util.spec_from_file_location("read_only_v4_gen_2",SRC)
G=importlib.util.module_from_spec(sp);sys.modules[sp.name]=G;sp.loader.exec_module(G)
# Precommit fresh seeds distinct from pilot and prior epoch-monitor diagnostics.
cfg={"v2":(1800,20261010711),"weighted":(1800,20261010712),"hard":(1200,20261010713)}
probes={}
for name,(n,seed) in cfg.items():
 samples=G.generate(n,seed=seed,holdout=True,mode=name)
 probes[name]=[(e.text,e.goal,e.via,e.urgent,e.fragile) for e in samples]
 print("GENERATED_NEW",name,n,seed,flush=True)
# Model-free, authored outside original generator: paired instruction order.
# A few pairs are semantically unambiguous; scope and revision are varied.
cases=[
("Giao hồ sơ tới thư viện. Trước lúc giao phải ghé trạm y tế nhận phong bì; chưa hủy chặng nhận.",
 "Giao hồ sơ tới thư viện. Thông báo cũ yêu cầu ghé trạm y tế nhận phong bì, nhưng yêu cầu này đã bị hủy; đi thẳng.", "library","clinic"),
("Chuyển bưu phẩm tới nhà ăn. Ban đầu định bỏ điểm lấy ở bãi xe, nhưng chỉ dẫn cuối cùng yêu cầu tới bãi xe lấy kiện.",
 "Chuyển bưu phẩm tới nhà ăn. Ban đầu định lấy thêm ở bãi xe, nhưng chỉ dẫn cuối cùng hủy bước lấy; kiện đã có đủ.", "canteen","parking"),
("Giao tài liệu tại ký túc xá. Lệnh đến phòng thí nghiệm lấy hồ sơ vẫn có hiệu lực dù các lệnh nhận khác đã hủy.",
 "Giao tài liệu tại ký túc xá. Lệnh đến phòng thí nghiệm lấy hồ sơ cũng đã bị hủy cùng các lệnh nhận khác.", "dorm","lab"),
("Đích giao là cổng chính. Không bỏ qua việc đến nhà thi đấu lấy phong bì, bước này vẫn bắt buộc.",
 "Đích giao là cổng chính. Không đến nhà thi đấu lấy phong bì nữa vì hàng đã nhận đủ từ đầu.", "gate","sports"),
("Mang hồ sơ đến văn phòng khoa. Thông báo mới khôi phục việc ghé giảng đường lấy thêm một thùng hàng.",
 "Mang hồ sơ đến văn phòng khoa. Thông báo mới bỏ việc ghé giảng đường lấy thêm một thùng hàng.", "office","lecture"),
("Giao thiết bị tới trạm y tế. Chặng lấy bổ sung ở thư viện chưa bị hủy và phải thực hiện.",
 "Giao thiết bị tới trạm y tế. Chặng lấy bổ sung ở thư viện đã bị hủy và không thực hiện.", "clinic","library"),
]
authored=[]
for i,(pos,neg,g,v) in enumerate(cases):
 for truth,txt in ((True,pos),(False,neg)):
  authored.append((txt,TargetSpec(g),TargetSpec(v) if truth else None,False,False))
# Add independent explicit fragile / urgent matched pairs, via absent.
flags=[
("Chuyển kiện tới thư viện. Đơn này phải giao ngay vì người nhận đang cần gấp.",
 "Chuyển kiện tới thư viện. Thời hạn đã lùi, đơn này giao bình thường.", "library","urgent"),
("Giao hộp tới nhà ăn. Hàng có thủy tinh rất dễ vỡ, phải nâng nhẹ.",
 "Giao hộp tới nhà ăn. Hộp không chứa đồ dễ vỡ, đồ bên trong chịu va đập tốt.", "canteen","fragile"),
("Mang giấy tờ đến ký túc xá. Lệnh giao hỏa tốc vẫn còn hiệu lực.",
 "Mang giấy tờ đến ký túc xá. Lệnh giao hỏa tốc đã bị rút lại, giao theo lịch.", "dorm","urgent"),
("Mang hàng tới văn phòng khoa. Thông báo cuối xác nhận trong kiện có đồ dễ nứt.",
 "Mang hàng tới văn phòng khoa. Thông báo cuối bác bỏ tin có đồ dễ nứt; kiện toàn đồ bền.", "office","fragile"),
]
for pos,neg,g,flag in flags:
 for true,txt in ((True,pos),(False,neg)):
  authored.append((txt,TargetSpec(g),None,true if flag=="urgent" else False,
                   true if flag=="fragile" else False))
probes["authored_scope_probe"]=authored
def tospec(obj):
 if obj is None:return None
 return {"type":obj.type,"ref":obj.ref,"anchor":obj.anchor}
# Guard that new test never text-collides with V9 train nor the matched
# synthetic subset used in A/B training.
pilottrain=W/"experiments/semantic_v9_real_like_pilot.jsonl"
pilotset=set(fold(json.loads(s)["text"]).strip() for s in pilottrain.open(encoding="utf-8"))
origbase=G.generate(10000,seed=2026101091,holdout=False,mode="weighted")
baseset=set(fold(x.text).strip() for x in origbase)
del origbase
overlap={}
for name,rows in probes.items():
 ts=set(fold(r[0]).strip() for r in rows)
 overlap[name]={"unique":len(ts),"overlap_V9_pool":len(ts&pilotset),
                "overlap_A_baseline_synth":len(ts&baseset)}
 print("OVERLAP",name,overlap[name],flush=True)
assert all(x["overlap_V9_pool"]==0 and x["overlap_A_baseline_synth"]==0 for x in overlap.values())
def metrics(y,rows):
 counts=collections.Counter()
 sub=collections.defaultdict(collections.Counter)
 errs=[]
 correctness=[]
 for i,(p,z) in enumerate(zip(y,rows)):
  truth={"goal":z[1],"via":z[2],"urgent":z[3],"fragile":z[4]}
  got=p.parsed
  hits={key:getattr(got,key)==value for key,value in truth.items()}
  hits["all"]=all(hits.values())
  correctness.append(hits)
  counts["n"]+=1
  for k,v in hits.items():counts[k]+=v
  for k in ("via","urgent","fragile"):
   label=bool(truth[k]);c=sub[k]
   c["n"]+=1
   c["positive"]+=label
   c["tp"]+=label and bool(getattr(got,k))
   c["fn"]+=label and not bool(getattr(got,k))
   c["fp"]+=not label and bool(getattr(got,k))
   c["tn"]+=not label and not bool(getattr(got,k))
  if not hits["all"]:
   errs.append({"index":i,"truth":{"goal":tospec(truth["goal"]),"via":tospec(truth["via"]),
             "urgent":truth["urgent"],"fragile":truth["fragile"]},
           "pred":{"goal":tospec(got.goal),"via":tospec(got.via),
              "urgent":got.urgent,"fragile":got.fragile},
           "hits":hits,"text":z[0]})
 report={"n":counts["n"],"exact":{k:round(counts[k]/counts["n"],5) for k in ("goal","via","urgent","fragile","all")},
         "confusion":{h:{k:val for k,val in dic.items()} for h,dic in sub.items()}}
 for head,c in sub.items():
  p1=c["tp"]+c["fn"];n1=c["tn"]+c["fp"]
  report["confusion"][head].update({
   "sensitivity":round(c["tp"]/p1,5) if p1 else None,
   "specificity":round(c["tn"]/n1,5) if n1 else None,
   "balanced_accuracy":round(.5*(c["tp"]/p1+c["tn"]/n1),5) if p1 and n1 else None})
 return report,correctness,errs
models={}
preds={}
scores={}
for arm,path in PATHS.items():
 parser=NeuralMissionParser.load(path)
 preds[arm]={};scores[arm]={}
 for name,rows in probes.items():
  predicted=[]
  with torch.no_grad():
   for start in range(0,len(rows),64):
    predicted+=parser.predict_batch([r[0] for r in rows[start:start+64]])
  score,hits,errors=metrics(predicted,rows)
  scores[arm][name]=score
  preds[arm][name]={"hits":hits,"errors":errors}
  print("MODEL",arm,name,score["exact"],"BACC",{h:score["confusion"][h]["balanced_accuracy"]
         for h in ("via","urgent","fragile")},flush=True)
 del parser
# McNemar exact with paired discordant successes per set; no SciPy needed.
def mcnemar_exact(b,c):
 n=b+c
 if not n:return 1.
 low=min(b,c)
 # two-sided exact Binomial(0.5), P(X<=low), clipped to 1.
 return min(1.,2*sum(math.comb(n,k) for k in range(low+1))*2.**(-n))
comparison={}
for name,rows in probes.items():
 ha=preds["A"][name]["hits"];hb=preds["B"][name]["hits"]
 cmp={"n":len(rows)}
 for label in ("all","via","urgent","fragile"):
  bonly=sum(not a[label] and b[label] for a,b in zip(ha,hb))
  aonly=sum(a[label] and not b[label] for a,b in zip(ha,hb))
  cmp[label]={"B_only_correct":bonly,"A_only_correct":aonly,
    "delta_pp":round(100*(bonly-aonly)/len(rows),3),
    "mcnemar_two_sided_p":round(mcnemar_exact(bonly,aonly),7)}
 for label in ("via","urgent","fragile"):
  ca=scores["A"][name]["confusion"][label];cb=scores["B"][name]["confusion"][label]
  cmp[label]["false_positive_delta_n"]=cb["fp"]-ca["fp"]
  cmp[label]["false_negative_delta_n"]=cb["fn"]-ca["fn"]
 comparison[name]=cmp
 # cap examples, both sides separately.
errs={}
for name in probes:
 botherr=preds["B"][name]["errors"]
 bonly_indices=[i for i,(a,b) in enumerate(zip(preds["A"][name]["hits"],preds["B"][name]["hits"]))
               if a["all"] and not b["all"]]
 bmistakes=set(bonly_indices)
 errs[name]={"n_B_wrong":len(botherr),"n_B_new_mistakes_vs_A":len(bonly_indices),
             "examples_B_new_errors":[e for e in botherr if e["index"] in bmistakes][:8]}
result={"kind":"V9_reproduced_generalization_cpu_fresh_seeds",
  "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "models":{arm:{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for arm,path in PATHS.items()},
  "fresh_seeds":{k:{"n":v[0],"seed":v[1]} for k,v in cfg.items()},
  "overlap_checks":overlap,"scores":scores,"paired_comparison":comparison,
  "selected_errors":errs,"elapsed_seconds":round(time.time()-t0,1),
  "limitations":["Synthetic fresh seeds still draw from same phrase-bank generator; not independent human OOD",
   "Authored 20-case small probe is a hand-written diagnostic and not a human-blind sample",
   "Single initial model seed, two posttraining arms only",
   "Exploratory paired p-values are unadjusted for multiple tests",
   "NO official val/test touched; NO model/code outside workspace changed"]}
with OUT.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
with ERRORS.open("x",encoding="utf-8") as f:
 for name,rec in errs.items():
  for ex in rec["examples_B_new_errors"]:
   f.write(json.dumps({"set":name,**ex},ensure_ascii=False)+"\n")
lines=["# V9 independent recheck on fresh synthetic seeds and off-generator authored mini-probe","",
       "Models A and B were evaluated on identical fresh rows, CPU only.","",
       "| Set | n | A all | B all | B−A | paired p (exploratory) | A via FPR | B via FPR |",
       "|---|---:|---:|---:|---:|---:|---:|---:|"]
for name,comp in comparison.items():
 a=scores["A"][name];b=scores["B"][name]
 def fpr(s):
  c=s["confusion"]["via"]
  return c["fp"]/(c["fp"]+c["tn"]) if c["fp"]+c["tn"] else float("nan")
 lines.append(f"| {name} | {comp['n']} | {a['exact']['all']:.2%} | {b['exact']['all']:.2%} | {comp['all']['delta_pp']:+.2f} pp | {comp['all']['mcnemar_two_sided_p']} | {fpr(a):.2%} | {fpr(b):.2%} |")
lines+=["","All metrics and confusion matrices are in the JSON checkpoint.",
        "The authored mini-probe is deliberately small and cannot prove generalization.",
        "Synthetic tests are not external human-language tests. No official validation/test was read."]
with MD.open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
print("FRESH_EVAL_COMPLETE",str(OUT),"SECS",result["elapsed_seconds"],flush=True)
