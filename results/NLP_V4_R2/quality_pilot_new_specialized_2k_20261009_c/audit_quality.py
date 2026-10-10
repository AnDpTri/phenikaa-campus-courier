"""Independent automated quality audit against original 40k and V2, with explicit rejection gates.
This does NOT certify semantic labels without human inspection.
"""
import ast,json,re,statistics,collections,random,csv
from pathlib import Path
from courier.nlp.text import tokenize,fold
from courier.nlp.neural import spec_labels,MAX_TOKENS
from courier.nlp.parser import TargetSpec
from courier.nlp.synth import GOAL_MODES,VIA_MODES,generate
D=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
ROOT=Path(r"D:\phenikaa")
OLD=ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/incoming_40000.jsonl"
TRAIN=ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/nlp_v4_curated_200000_quality_v2_reindexed.jsonl"
PATHS={"pilot_v1":D/"pilot_new_specialized_2000.jsonl","pilot_v2":D/"pilot_new_specialized_2000_v2.jsonl","pilot_v3":D/"pilot_new_specialized_2000_v3.jsonl"}
OUT=D/"independent_quality_audit_v1.json"
PATTERNS={
 "correction":r"\b(dinh chinh|nham roi|thay vao|dia chi moi|ban dau|doi lai|ban cu|tin cu|truoc kia|nham lan|cap nhat|huy|da bi huy)\b",
 "irrelevant_places":r"\b(hom qua|tuan truoc|nguoi nhan da roi|di ngang|khong dung chan|khong can ghe|khong lien quan|dia chi cu|diem khong lien quan|don khac|bo qua|dong cua|tin cu)\b",
 "spatial":r"\b(gan|xa|phia|huong|ben|tren|duoi|nhat)\b",
 "negation":r"\b(khong|dung|chua|bo qua|tranh|da huy|huy)\b",
 "before":r"\btruoc\b",
 "lay":r"\blay\b",
 "wrong_direction_phrasing":r"\bnam o ve huong\b",
 "heldout_pattern":r"\blay\b.{0,90}\bo\b.{0,90}\bxong thi\b",
}
REGEX={k:re.compile(v) for k,v in PATTERNS.items()}
def read_rows(path):
 with path.open(encoding="utf-8") as f:
  for line in f:yield json.loads(line)
def shape(iterable):
 rows=list(iterable);lengths=sorted(len(tokenize(x["text"])) for x in rows);c=collections.Counter()
 normalized=set();first8=set()
 for j in rows:
  t=fold(j["text"]);norm=" ".join(t.split());normalized.add(norm)
  first8.add(" ".join(tokenize(j["text"])[:8]))
  present=j["via"] is not None
  c["n"]+=1;c["via"]+=present;c["accented"]+=any(ord(ch)>127 for ch in j["text"])
  for k,rgx in REGEX.items():c[k]+=bool(rgx.search(t))
  c["via_without_truoc"]+=present and not REGEX["before"].search(t)
  c["none_with_lay"]+=(not present) and bool(REGEX["lay"].search(t))
 def frac(k):return round(c[k]/len(rows),5)
 return {"n":len(rows),"unique":len(normalized),"unique_first8":len(first8),
    "p10":lengths[int(len(lengths)*.1)],"p50":lengths[int(len(lengths)*.5)],
    "p90":lengths[int(len(lengths)*.9)],"p99":lengths[int(len(lengths)*.99)],"max":max(lengths),
    "via_rate":frac("via"),"accent_rate":frac("accented"),
    "features":{k:{"count":c[k],"rate":frac(k)} for k in PATTERNS},
    "via_without_truoc":c["via_without_truoc"],
    "none_with_lay":c["none_with_lay"]}
def samples_of_v2():
 for e in generate(2000,20261129,False):
  yield {"text":e.text,"via":None if e.via is None else {"type":e.via.type,"ref":e.via.ref,"anchor":e.via.anchor}}
# Correctness checks inspect exact full positive flag cue phrases from the generator's authored banks.
script=(D/"pilot_generator.py").read_text(encoding="utf-8")
tree=ast.parse(script)
banks={}
for n in tree.body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("URGENT_TRUE","FRAGILE_TRUE","URGENT_FALSE","FRAGILE_FALSE"):
  banks[n.targets[0].id]=list(ast.literal_eval(n.value))
assert len(banks)==4
changes={
"Đây là đơn khẩn, xử lý ngay khi nhận.":"Ban đầu bảo không vội, giờ đã đổi thành giao khẩn.",
"Đồ gốm bên trong có thể vỡ nếu rơi.":"Trước nói hàng bền; đính chính: đồ gốm dễ vỡ.",
"Không cần ưu tiên hỏa tốc.":"Tin cũ báo khẩn đã hủy, hiện không gấp.",
"Kiện hàng chắc chắn, không thuộc loại dễ vỡ.":"Bản trước ghi dễ vỡ, đính chính: hàng rất bền.",
}
v3_banks={k:[changes.get(p,p) for p in a] for k,a in banks.items()}
def check_positive_flags(rows,flag_banks):
 c=collections.Counter();sample=[];groups=collections.Counter()
 for x in rows:
  for field,bank in (("urgent","URGENT_TRUE"),("fragile","FRAGILE_TRUE")):
   if x[field]:
    c[field+"_positive"]+=1
    if not any(fold(s) in fold(x["text"]) for s in flag_banks[bank]):
     c[field+"_missing_evidence"]+=1
     if len(sample)<5:sample.append({"id":x["id"],"field":field,"text":x["text"][:350]})
  for name in ("goal","via"):
   s=x[name]
   if s is None:continue
   tgt=TargetSpec(s["type"],s["ref"],s["anchor"])
   assert (tgt.ref or "named") in (GOAL_MODES if name=="goal" else VIA_MODES),x["id"]
  g=x["goal"];v=x["via"];spec_labels(TargetSpec(g["type"],g["ref"],g["anchor"]),None if v is None else TargetSpec(v["type"],v["ref"],v["anchor"]),x["urgent"],x["fragile"])
  if len(tokenize(x["text"]))>MAX_TOKENS:c["truncated"]+=1
 return {"counts":dict(c),"examples":sample}
def main():
 if OUT.exists():raise RuntimeError("Refusing overwrite "+str(OUT))
 reference=shape(read_rows(OLD))
 v2=shape(samples_of_v2())
 pilots={k:shape(read_rows(v)) for k,v in PATHS.items()}
 pilot_flags={k:check_positive_flags(read_rows(v),v3_banks if k=="pilot_v3" else banks) for k,v in PATHS.items()}
 old_set={fold(x["text"]).strip() for x in read_rows(OLD)}
 curated_set={fold(x["text"]).strip() for x in read_rows(TRAIN)}
 overlapping={k:sum(fold(r["text"]).strip() in old_set or fold(r["text"]).strip() in curated_set for r in read_rows(path)) for k,path in PATHS.items()}
 gates={}
 for name,p in pilots.items():
  checks={
   "exactly_2000":p["n"]==2000,
   "unique_exact":p["unique"]==2000,
   "unseen_old_and_curated":overlapping[name]==0,
   "no_more_than_160_tokens":p["max"]<=MAX_TOKENS,
   "token_median_within_20pct_of_old":(.8*reference["p50"]<=p["p50"]<=1.2*reference["p50"]),
   "token_p90_within_20pct_of_old":(.8*reference["p90"]<=p["p90"]<=1.2*reference["p90"]),
   "accent_rate_within_12pp":abs(p["accent_rate"]-reference["accent_rate"])<=.12,
   "via_rate_within_8pp":abs(p["via_rate"]-reference["via_rate"])<=.08,
   "corrections_at_least_60pct_old":p["features"]["correction"]["rate"]>=.6*reference["features"]["correction"]["rate"],
   "irrelevant_places_at_least_50pct_old":p["features"]["irrelevant_places"]["rate"]>=.5*reference["features"]["irrelevant_places"]["rate"],
   "spatial_at_least_75pct_old":p["features"]["spatial"]["rate"]>=.75*reference["features"]["spatial"]["rate"],
   "via_without_truoc_at_least_25pct_true":p["via_without_truoc"]>=.25*p["via_rate"]*p["n"],
   "none_with_lay_at_least_10pct_none":p["none_with_lay"]>=.1*(1-p["via_rate"])*p["n"],
   "no_flag_true_without_explicit_evidence":not any(v>0 for k,v in pilot_flags[name]["counts"].items() if k.endswith("_missing_evidence")),
   "no_syntactically_wrong_direction_phrase":p["features"]["wrong_direction_phrasing"]["count"]==0,
  }
  gates[name]={"criteria":checks,"pass":all(checks.values()),"failed":[k for k,v in checks.items() if not v]}
 report={"status":"automated_audit_only","original":reference,"v2_sample_2000":v2,"pilots":pilots,"overlap_original_and_curated":overlapping,"flag_grounding":pilot_flags,"gates":gates,
 "limitations":["Template-grounded positive flags do not validate full natural-language entailment","Syntactic diversity/semantic human audit still required","No training was done","Report measures pilot 2k, not 40k"]}
 with OUT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
 for group in ("original","v2_sample_2000"):
  r=report[group];print("SUMMARY",group,json.dumps({"n":r["n"],"p50":r["p50"],"p90":r["p90"],"via_rate":r["via_rate"],"accent_rate":r["accent_rate"],"corrections":r["features"]["correction"],"irrelevant":r["features"]["irrelevant_places"],"spatial":r["features"]["spatial"]},ensure_ascii=True),flush=True)
 for name,p in pilots.items():
  print("AUDIT",name,json.dumps({"n":p["n"],"p50":p["p50"],"p90":p["p90"],"accent_rate":p["accent_rate"],"via_rate":p["via_rate"],"corrections":p["features"]["correction"],"irrelevant":p["features"]["irrelevant_places"],"spatial":p["features"]["spatial"],"grammatical_error":p["features"]["wrong_direction_phrasing"],"flag_grounding":pilot_flags[name]["counts"],"overlap":overlapping[name],"gates":gates[name]},ensure_ascii=True),flush=True)
 print("AUDIT_COMPLETE",str(OUT),flush=True)
if __name__=="__main__":main()
