"""Semantic V6: balance normal/negation/revision regimes with 8-way role counterfactuals.

Based on V5 experimental helper imported read-only. ONLY creates V6 artifacts
within research workspace. No training/validation/test accessed.
"""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,random,re,sys,time
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS
WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
lib=importlib.util.spec_from_file_location("semantic_v5_helper_readonly",WORK/"candidate_semantic_v5.py")
V=importlib.util.module_from_spec(lib);sys.modules[lib.name]=V;lib.loader.exec_module(V)
SEED=2026101066
N_GROUPS=1200
REGIMES=("plain_mission","cancelled_stop","updated_instruction")
# Distinct active and absent pickup instructions from everyday campus speech.
PLAIN_TRUE=(
 "Robot cần ghé {loc} lấy thêm hàng trước khi đi giao; bước này bắt buộc.",
 "Điểm nhận phụ hiện tại là {loc}; phải tới đó nhận chứng từ.",
 "Cần đến {loc} lấy tài liệu bổ sung, rồi mới chuyển tới đích.",
 "Đơn này gồm hai chặng: nhận phong bì tại {loc}, sau đó giao.",
 "Trong lộ trình có chặng ghé {loc} nhận kiện trước khi hoàn tất giao.",
 "Lấy thêm chứng từ tại {loc} rồi mới chuyển hàng đến người nhận.",
 "Địa chỉ nhận kiện bổ sung là {loc}; robot phải tới đó.",
 "Cần nhận bưu phẩm tại {loc}; đây là chặng lấy có hiệu lực.",
)
PLAIN_FALSE=(
 "Robot đã có đủ hàng, không cần ghé {loc} lấy thêm trước khi giao.",
 "Tại {loc} không có kiện cần nhận; đơn này không có chặng nhận phụ.",
 "Không lấy tài liệu bổ sung tại {loc}; hàng đã có sẵn để giao.",
 "Đơn này chỉ có chặng giao; không nhận phong bì tại {loc}.",
 "Địa điểm {loc} không phải chặng nhận kiện trên lộ trình giao.",
 "Chứng từ đã có đủ, không phải ghé {loc} nhận thêm hàng.",
 "Địa chỉ {loc} không có kiện bổ sung cần robot tới nhận.",
 "Không có bưu phẩm cần nhận tại {loc}; bỏ qua chặng lấy này.",
)
# The original V5 branch already includes cancelled earlier pickup in both labels.
REV_TRUE=(
 "Lệnh cũ đã hủy việc lấy hàng. Chỉ dẫn mới khôi phục bước ghé {loc} nhận thêm kiện.",
 "Phiếu nháp bỏ chặng nhận ở {loc}, nhưng phiếu ký lại yêu cầu lấy hàng tại đó.",
 "Tin nhắn cũ nói bỏ điểm nhận {loc}; thông báo mới xác nhận phải ghé nơi này lấy.",
 "Ban đầu bỏ việc lấy chứng từ ở {loc}, sau đó người gửi yêu cầu khôi phục.",
 "Bản cũ gạch địa chỉ nhận {loc}, còn yêu cầu mới bắt buộc đến đó nhận hàng.",
 "Địa điểm lấy {loc} từng bị xóa; bây giờ được thêm trở lại lộ trình.",
 "Trong phiếu trước chặng ghé {loc} đã hủy; quyết định cuối cùng là tới đó lấy.",
 "Thông báo trước bỏ điểm lấy {loc}; thông báo sau khẳng định vẫn phải nhận kiện.",
)
REV_FALSE=(
 "Lệnh cũ yêu cầu lấy hàng tại {loc}. Chỉ dẫn mới hủy bước ghé đó, chỉ còn giao.",
 "Phiếu nháp có chặng nhận ở {loc}, nhưng phiếu ký bỏ yêu cầu lấy hàng tại đó.",
 "Tin nhắn cũ yêu cầu ghé điểm nhận {loc}; thông báo mới xác nhận bỏ bước lấy.",
 "Ban đầu phải nhận chứng từ ở {loc}, sau đó người gửi yêu cầu hủy.",
 "Bản cũ có địa chỉ nhận {loc}, còn yêu cầu mới không cho tới đó nhận hàng.",
 "Địa điểm lấy {loc} từng có hiệu lực; bây giờ đã bị xóa khỏi lộ trình.",
 "Trong phiếu trước chặng ghé {loc} còn hiệu lực; quyết định cuối cùng là hủy bước lấy.",
 "Thông báo trước yêu cầu lấy tại {loc}; thông báo sau khẳng định không nhận nữa.",
)
REGIME_TEMPLATES=(
 (PLAIN_TRUE,PLAIN_FALSE),
 (V.PICKUP_ACTIVE,V.PICKUP_REVOKED),
 (REV_TRUE,REV_FALSE),
)
assert all(len(z)==8 and len(x)==8 for z in REGIME_TEMPLATES for x in z)
OUT_TRAIN=WORK/"experiments/semantic_v6_balanced_train.jsonl"
OUT_CHAL=WORK/"experiments/semantic_v6_family_challenge.jsonl"
OUT_META=WORK/"checkpoints/semantic_v6_dataset_manifest.json"
for p in (OUT_TRAIN,OUT_CHAL,OUT_META):
 if p.exists():raise FileExistsError(str(p))
r=random.Random(SEED)
start=time.time()
a={"candidate_train":[],"family_challenge":[]}
counts=collections.defaultdict(collections.Counter)
regimes=collections.defaultdict(lambda:collections.Counter())
word_stats=collections.defaultdict(lambda:collections.Counter())
seen=set()
gmodes=["named"]*5+["near","far","anchor_near","north","south","east","west","south_most"]
vmodes=["named"]*7+["near","far","north","south","east","west"]
for i in range(N_GROUPS):
 fam=i%8
 regime=(i//8)%3
 is_accented=r.random()<.56
 g=V.G.Generator(seed=r.randrange(2**31),holdout=False,mode="hard")
 goal,gloc,used=V.get_target(g,r.choice(gmodes),set(),is_accented)
 via,vloc,_=V.get_target(g,r.choice(vmodes),used,is_accented)
 order=r.randrange(4)
 V.PICKUP_ACTIVE,V.PICKUP_REVOKED=REGIME_TEMPLATES[regime]
 split="candidate_train" if fam in range(6) else "family_challenge"
 group=[]
 for active in (False,True):
  for urgent in (False,True):
   for fragile in (False,True):
    row=V.make_row(i,fam,goal,via,gloc,vloc,is_accented,active,urgent,fragile,order)
    row["id"]=f"sf6_{i:05d}_{int(active)}{int(urgent)}{int(fragile)}"
    row["provenance"]["generator"]="experimental_semantic_v6"
    row["provenance"]["regime"]=REGIMES[regime]
    row["provenance"]["seed"]=SEED
    txt=fold(row["text"]).strip()
    if txt in seen:raise RuntimeError(f"DUPLICATE {i}")
    seen.add(txt)
    group.append(row)
    counts[split]["rows"]+=1
    for key in ("via","urgent","fragile"):
     counts[split][key]+=bool(row[key])
    counts[split]["accented"]+=is_accented
    w=set(re.findall(r"[a-z0-9]+",txt))
    for key in ("lay","nhan","truoc","huy","khong","vo","gap","khan","nhay","sau"):
     if key in w:
      word_stats[split][key+"_has"]+=1
      for lbl in ("via","urgent","fragile"):
       word_stats[split][key+"_"+lbl]+=bool(row[lbl])
 assert len(group)==8
 assert len({json.dumps(x["goal"],sort_keys=True) for x in group})==1
 assert sum(bool(x["via"]) for x in group)==4
 assert sum(x["urgent"] for x in group)==4
 assert sum(x["fragile"] for x in group)==4
 a[split].extend(group)
 regimes[split][REGIMES[regime]]+=8
assert sum(len(x) for x in a.values())==9600
assert len(a["candidate_train"])==7200 and len(a["family_challenge"])==2400
# New V6 must not accidentally duplicate any OLD 200K training example.
old=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch\scratch_s0_e01_seed2026110501.jsonl")
exact_overlap=0
with old.open(encoding="utf-8") as f:
 for line in f:
  exact_overlap+=fold(json.loads(line)["text"]).strip() in seen
assert exact_overlap==0
audit={}
for split,rows in a.items():
 c=counts[split]
 assert all(c[k]*2==c["rows"] for k in ("via","urgent","fragile"))
 assert len({x["group_id"] for x in rows})==len(rows)//8
 path=OUT_TRAIN if split=="candidate_train" else OUT_CHAL
 with path.open("x",encoding="utf-8",newline="\n") as f:
  for item in rows:f.write(json.dumps(item,ensure_ascii=False)+"\n")
 stats=word_stats[split]
 audit[split]={"rows":len(rows),"groups":len(rows)//8,
    "label_positive_rate":{k:round(c[k]/len(rows),4) for k in ("via","urgent","fragile")},
    "regime_counts":dict(regimes[split]),"accented_fraction":round(c["accented"]/len(rows),4),
    "word_cues":{key:{"support":stats[key+"_has"],
      **{lbl:(round(stats[key+"_"+lbl]/stats[key+"_has"],4) if stats[key+"_has"] else None)
         for lbl in ("via","urgent","fragile")}}
      for key in ("lay","nhan","truoc","huy","khong","vo","gap","khan","nhay","sau")},
    "file":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
meta={"name":"experimental_semantic_v6","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "seed":SEED,"total_groups":N_GROUPS,"rows":9600,
      "family_split":{"candidate_train":[0,1,2,3,4,5],"family_challenge":[6,7]},
      "splits":audit,"old_200k_overlap_count":exact_overlap,
      "qa":["within group 8 combinations, labels 50/50 each",
            "latest-instruction semantics encoded in active/cancelled/revised via",
            "normal delivery examples included","no normalized text collisions",
            "goals invariant within counterfactual group","token bound checked",
            "hold out two whole structural realization families"],
      "limitations":["Only machine-enforced label consistency, requires human spot review",
                     "All generated; no official validation/test loaded",
                     "No evidence of improvement before retraining and truly external evaluation",
                     "Family holdout remains same concept distribution / shared place lexicon",
                     "Do not merge family_challenge into future training",
                     "No changes outside research workspace"],
      "elapsed_seconds":round(time.time()-start,1)}
with OUT_META.open("x",encoding="utf-8") as f:json.dump(meta,f,ensure_ascii=False,indent=2)
print("V6_DATA_READY",json.dumps({"totals":{k:len(v) for k,v in a.items()},
 "regime":{k:dict(v) for k,v in regimes.items()},
 "audit":audit,"manifest":str(OUT_META),"elapsed_s":meta["elapsed_seconds"]},ensure_ascii=False),flush=True)
