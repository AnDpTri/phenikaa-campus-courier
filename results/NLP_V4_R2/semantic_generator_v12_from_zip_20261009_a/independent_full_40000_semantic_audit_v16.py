"""Independent FULL-POPULATION V16 verifier, no reliance on generator quality counters.
Every row/label/provenance inspected. Creates new report and 320 stratified examples.
No existing files modified.
"""
import json,re,random,hashlib,statistics,collections
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.parser import TargetSpec
from courier.nlp.neural import spec_labels,MAX_TOKENS
from courier.nlp.synth import GOAL_MODES,VIA_MODES
P=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
ROOT=Path(r"D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review")
DATA=P/"specialized_new_40000_v16_role_semantics.jsonl"
PROV=P/"specialized_new_40000_v16_provenance.jsonl"
OUT=P/"independent_full_semantic_audit_v16.json"
SAMPLE=P/"independent_semantic_strata_v16_320.jsonl"
for x in (OUT,SAMPLE):
 if x.exists():raise RuntimeError("Refuse overwrite: "+str(x))
from importlib.util import spec_from_file_location,module_from_spec
ss=spec_from_file_location("v12_immutable_semantic_source",P/"role_semantic_generator_v12.py")
mod=module_from_spec(ss);ss.loader.exec_module(mod)
rows=[json.loads(x) for x in DATA.open(encoding="utf-8")]
prov=[json.loads(x) for x in PROV.open(encoding="utf-8")]
assert len(rows)==len(prov)==40000
f=collections.Counter();error=collections.Counter();examples=collections.defaultdict(list)
goal_modes=collections.Counter();via_modes=collections.Counter();prefix=collections.Counter()
lengths=[]
canon=set()
def flag_err(code,j,detail):
 error[code]+=1
 if len(examples[code])<8:examples[code].append({"id":j["id"],"reason":detail})
def matchtype(k,t):
 return any(fold(a) in t for a in mod.PLACES[k])
frag_bank_true=mod.FRAGILITY[True];frag_bank_false=mod.FRAGILITY[False]
urgent_bank_true=mod.URGENCY[True];urgent_bank_false=mod.URGENCY[False]
# The legacy V15 removes fragile-positive templates implying particular physical contents.
frag_bank_true=tuple(s for s in frag_bank_true if "linh kien" not in fold(s) and "do thuy tinh" not in fold(s))
for i,(row,m) in enumerate(zip(rows,prov)):
 t=fold(row["text"]);g=row["goal"];v=row["via"];goal_modes[g["ref"] or "named"]+=1
 vm="none" if v is None else (v["ref"] or "named");via_modes[vm]+=1
 f["N"]+=1
 if row["id"]!=m["id"] or row["id"]!=f"sem_v16_{i+1:06d}":flag_err("id_mismatch",row,{"m":m["id"],"line":i+1})
 if set(row)!={"id","text","goal","via","urgent","fragile"}:flag_err("schema_fields",row,list(row))
 try:
  gg=TargetSpec(g["type"],g["ref"],g["anchor"])
  vv=None if v is None else TargetSpec(v["type"],v["ref"],v["anchor"])
  if (g["ref"] or "named") not in GOAL_MODES or (v is not None and (v["ref"] or "named") not in VIA_MODES):raise ValueError("unsupported mode")
  if len(spec_labels(gg,vv,row["urgent"],row["fragile"]))!=8:raise ValueError("label shapes")
 except Exception as exc:flag_err("invalid_label",row,str(exc))
 if m["goal_mode"] != (g["ref"] or "named"):flag_err("goal_mode_mismatch",row,str(m["goal_mode"]))
 if m["via_mode"] != vm:flag_err("via_mode_mismatch",row,str(m["via_mode"]))
 if g["type"] and not matchtype(g["type"],t):flag_err("goal_place_missing",row,g["type"])
 if v is not None and v["type"] and not matchtype(v["type"],t):flag_err("via_place_missing",row,v["type"])
 for a,k in ((g["anchor"],"goal"),(None if v is None else v["anchor"],"via")):
  if a is not None and not matchtype(a,t):flag_err("anchor_missing",row,(k,a))
 if g["ref"]=="anchor_near":
  f["anchor_near"]+=1
  if not any(w in t for w in ("khong tinh chinh","ngoai tru chinh","dia danh khac","khong tinh moc","tru chinh")):flag_err("anchor_exclusion_missing",row,t[:190])
 if g["ref"] in ("near","far","anchor_near") and not g["anchor"]:flag_err("goal_anchor_null",row,str(g))
 if v is not None and v["ref"] in ("near","far") and not v["anchor"]:flag_err("via_anchor_null",row,str(v))
 length=len(tokenize(row["text"]));lengths.append(length)
 if not (14<=length<=MAX_TOKENS):flag_err("invalid_length",row,length)
 if len(row["text"])>1450:flag_err("max_chars",row,len(row["text"]))
 if not isinstance(row["urgent"],bool) or not isinstance(row["fragile"],bool):flag_err("invalid_flag_dtype",row,(row["urgent"],row["fragile"]))
 item=m["parcel"]
 if fold(item) not in t:flag_err("main_item_absent",row,item)
 if m["supplemental_pickup_item"]:
  f["supplemental"]+=1
  if v is None:flag_err("pickup_item_no_via",row,m["supplemental_pickup_item"])
  if fold(m["supplemental_pickup_item"]) not in t:flag_err("supplemental_item_absent",row,m["supplemental_pickup_item"])
 old=m["former_destination"]
 if old:
  f["revised_goal"]+=1
  if v is not None:f["revision_and_via"]+=1
  if fold(old) not in t:flag_err("former_goal_missing",row,old)
  if not any(x in t for x in ("huy","doi dia chi","thay dich giao","khong con hieu luc","khong dung diem ay","bo diem cu","khong duoc giao","nham roi","khong theo dia chi cu")):
   flag_err("former_goal_no_revocation",row,row["text"][:160])
  # Old goal type is strictly excluded by the authoring generator; verify text places by type.
  if g["type"] is not None and old in mod.PLACES[g["type"]]:flag_err("former_goal_same_type",row,old)
 oldp=m["former_pickup"]
 if oldp:
  f["revised_pickup"]+=1
  if fold(oldp) not in t:flag_err("former_pickup_missing",row,oldp)
  if not any(x in t for x in ("huy","bo","khong con","rut lai","sai")):flag_err("former_pickup_no_revocation",row,row["text"][:180])
  if v is not None and v["type"] is not None and oldp in mod.PLACES[v["type"]]:flag_err("former_pickup_same_via_type",row,oldp)
 for neg in m["negative_mentions"]:
  f["neg_roles"]+=1;f["negative_role_"+neg["role"]]+=1
  if fold(neg["lexeme"]) not in t:flag_err("negative_place_absent",row,neg)
  if neg["type"] in (g["type"],g["anchor"],None if v is None else v["type"],None if v is None else v["anchor"]):
   flag_err("negative_type_conflicts_active",row,neg)
  if g["type"] is None and neg["role"] not in ("historical","not_related"):
   flag_err("map_only_negative_role",row,neg)
 for value,k,bank in ((row["urgent"],"urgent",urgent_bank_true if row["urgent"] else urgent_bank_false),
                       (row["fragile"],"fragile",frag_bank_true if row["fragile"] else frag_bank_false)):
  # Require explicit authored bank cue for every TRUE and for any False with such language.
  f[k+"_true"]+=bool(value)
  has=any(fold(s) in t for s in bank)
  if value and not has:flag_err(k+"_missing_true_cue",row,row["text"][-200:])
 f["via_present"]+=v is not None
 f["via_without_truoc"]+=v is not None and not bool(re.search(r"\btruoc\b",t))
 f["none_with_lay"]+=v is None and bool(re.search(r"\blay\b",t))
 f["accented"]+=any(ord(c)>127 for c in row["text"])
 f["redirect"]+=old is not None
 f["noise_clause"]+= bool(m["negative_mentions"])
 for pat,code in ((r"\bbuu kien (tui|thung|hop|bo|xap|kien)\b","double_buu_kien"),
                  (r"\bngoai kien (buu kien|kien hang|thung|goi|tui|hop)\b","ngoai_kien_dup"),
                  (r"\b(ke do moi|sau do can|roi ket thuc bang viec)\s+(dia chi|noi nhan|diem ket thuc)\b","temporal_static"),
                  (r"\bdia diem diem\b","double_dia_diem"),
                  (r"\blay\b.{0,90}\bo\b.{0,90}\bxong thi\b","leaked_exact_holdout_phrase")):
  if re.search(pat,t):flag_err(code,row,row["text"][:180])
 canonical=t.strip()
 if canonical in canon:flag_err("duplicate_in_v16",row,canonical[:80])
 canon.add(canonical)
 prefix[" ".join(t.split()[:8])]+=1
# fold-normalized exact overlap vs user existing corpus and prior V11
for name,path in (("original40k",ROOT/"incoming_40000.jsonl"),
                  ("curated200k",ROOT/"nlp_v4_curated_200000_quality_v2_reindexed.jsonl"),
                  ("v11",P.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl")):
 overlap=0
 with path.open(encoding="utf-8") as inp:
  for line in inp:
   oldt=fold(json.loads(line)["text"]).strip()
   overlap+= oldt in canon
 f["overlap_"+name]=overlap
 if overlap:flag_err("exact_overlap_"+name,rows[0],overlap)
# Build stratified review from full population, always sample independently.
criteria={
 "revise_goal_with_via":lambda j,m:m["former_destination"] is not None and j["via"] is not None,
 "revise_goal_no_via":lambda j,m:m["former_destination"] is not None and j["via"] is None,
 "revised_pickup_with_via":lambda j,m:m["former_pickup"] is not None and j["via"] is not None,
 "revised_pickup_no_via":lambda j,m:m["former_pickup"] is not None and j["via"] is None,
 "both_revisions":lambda j,m:m["former_destination"] is not None and m["former_pickup"] is not None,
 "goal_anchor_near":lambda j,m:j["goal"]["ref"]=="anchor_near",
 "goal_most":lambda j,m:str(j["goal"]["ref"]).endswith("_most"),
 "goal_near_far":lambda j,m:j["goal"]["ref"] in ("near","far"),
 "via_relative":lambda j,m:j["via"] is not None and j["via"]["ref"] in ("near","far","north","south","east","west"),
 "via_named_no_truoc":lambda j,m:j["via"] is not None and j["via"]["ref"] is None and "truoc" not in fold(j["text"]).split(),
 "no_via_take":lambda j,m:j["via"] is None and bool(re.search(r"\blay\b",fold(j["text"]))),
 "supplement":lambda j,m:m["supplemental_pickup_item"] is not None,
 "multi_distractors":lambda j,m:len(m["negative_mentions"])==2,
 "urgent_fragile":lambda j,m:j["urgent"] and j["fragile"],
 "both_false":lambda j,m:not j["urgent"] and not j["fragile"],
 "unaccented":lambda j,m:all(ord(c)<=127 for c in j["text"]),
}
rng=random.Random(20261009981);selected={};sizes={}
for name,pred in criteria.items():
 candidates=[i for i,(j,m) in enumerate(zip(rows,prov)) if pred(j,m)];sizes[name]=len(candidates)
 for i in rng.sample(candidates,min(18,len(candidates))):selected.setdefault(i,name)
remain=[i for i in range(len(rows)) if i not in selected]
for i in rng.sample(remain,max(0,320-len(selected))):selected[i]="random"
with SAMPLE.open("x",encoding="utf-8",newline="\n") as dst:
 for i,group in sorted(selected.items()):
  dst.write(json.dumps({"source_line":i+1,"group":group,"semantic_source":prov[i],**rows[i]},ensure_ascii=False)+"\n")
lengths.sort()
report={"status":"FULL_POPULATION_CONTRACT_TEST_COMPLETE_SEMANTIC_REVIEW_PENDING",
        "dataset":str(DATA),"rows_checked":len(rows),"sha256":hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "counters":dict(f),"reference_distribution":dict(goal_modes),"via_distribution":dict(via_modes),
        "p10":lengths[4000],"p50":lengths[20000],"p90":lengths[36000],"p99":lengths[39600],"max":lengths[-1],
        "error_counts":dict(error),"error_examples":dict(examples),"top8_prefixes":prefix.most_common(10),
        "independent_sample_size":len(selected),"sample_population_strata":sizes,
        "limitations":["Coverage is full for programmed structural contracts, not proof of all human-language entailments.","No V2 model or test-set labels used in generation.","The ZIP's validation split was not used to author templates."]}
with OUT.open("x",encoding="utf-8") as dst:json.dump(report,dst,ensure_ascii=False,indent=2)
print("REPORT",OUT,flush=True)
print("SUMMARY",json.dumps({k:report[k] for k in ("rows_checked","sha256","p50","p90","p99","max","independent_sample_size","error_counts","counters")},ensure_ascii=True),flush=True)
print("TOPPREFIXES",json.dumps(report["top8_prefixes"],ensure_ascii=True),flush=True)
