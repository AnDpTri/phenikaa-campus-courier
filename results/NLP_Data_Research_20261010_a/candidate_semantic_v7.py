"""Experimental V7: counterfactual 8-way scenes, *invariant cue presence* in
each scene, realistic mix of plain/cancelled/revised pickup instructions.
Only create-new files inside research workspace. Frozen generator is read-only.
"""
from __future__ import annotations
import collections,datetime,hashlib,importlib.util,json,random,re,sys,time
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS,spec_labels
WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
SRC=WORK/"candidate_semantic_v5.py"
sp=importlib.util.spec_from_file_location("semantic_v5_helper_for_v7",SRC)
V=importlib.util.module_from_spec(sp);sys.modules[sp.name]=V;sp.loader.exec_module(V)

GOAL=V.GOAL
VIA=[
# Normal pickup; no cancellation context
(
("Không bỏ qua việc ghé {loc} lấy kiện; cần nhận thêm hàng.",
 "Địa điểm {loc} là chặng nhận bổ sung, robot cần lấy hàng tại đó.",
 "Trước khi giao, robot phải nhận tài liệu tại {loc}.",
 "Không được bỏ qua việc nhận hồ sơ tại {loc}, phải ghé lấy.",
 "Điểm lấy tại {loc} đã được xác nhận; đến nhận kiện rồi đi giao.",
 "Sau khi kiểm tra phiếu, chặng lấy tại {loc} có hiệu lực và cần nhận hàng.",
 "Không đổi điểm lấy ở {loc}; robot cần ghé nhận thêm hàng.",
 "Tại {loc} có chặng nhận thực tế trước lúc giao."),
("Không cần ghé {loc} lấy kiện; đã nhận đủ hàng.",
 "Địa điểm {loc} chỉ là mốc đường, việc nhận và lấy hàng đã xong từ điểm xuất phát.",
 "Trước khi giao, robot đã nhận đủ tài liệu ở điểm xuất phát; {loc} chỉ dùng làm mốc.",
 "Không cần nhận hồ sơ tại {loc}, robot bỏ bước ghé lấy bổ sung.",
 "Điểm lấy tại {loc} chỉ là ghi chú về một đơn khác; đã nhận đủ kiện.",
 "Sau khi kiểm tra phiếu, chặng lấy tại {loc} là dữ liệu cũ, đã nhận đủ hàng.",
 "Không dùng điểm lấy ở {loc}; robot đã nhận đủ hàng tại nơi khởi hành.",
 "Tại {loc} chỉ có ghi chú về chặng nhận của hôm trước, đơn này đã có đủ hàng.")),
# Cancelled pickup / reactivated pickup
(
("Đã hủy chặng lấy trước đây ở nơi khác; riêng chặng nhận tại {loc} vẫn hiệu lực.",
 "Không hủy lệnh ghé {loc} nhận thêm kiện; bước lấy này vẫn bắt buộc.",
 "Thông tin hủy chặng nhận tại {loc} là nhầm; robot phải tới lấy hàng.",
 "Lệnh trước nói bỏ nhận tại {loc}, nhưng lệnh mới vẫn yêu cầu nhận.",
 "Điểm lấy {loc} không thuộc danh sách bị hủy; robot cần nhận thêm kiện.",
 "Sau khi hủy lệnh khác, yêu cầu lấy tại {loc} vẫn còn, phải nhận.",
 "Chặng nhận tại {loc} từng bị hủy, hiện phục hồi và không được bỏ bước lấy.",
 "Không bỏ qua chặng lấy ở {loc}; thông báo hủy chỉ áp dụng cho lệnh cũ, cần nhận."),
("Đã hủy chặng lấy trước đây ở nơi khác; chặng nhận tại {loc} cũng hết hiệu lực.",
 "Đã hủy lệnh ghé {loc} nhận thêm kiện; bước lấy này không còn.",
 "Thông tin hủy chặng nhận tại {loc} là đúng; robot bỏ bước lấy hàng.",
 "Lệnh trước nói bỏ nhận tại {loc}, lệnh mới xác nhận tiếp tục bỏ chặng nhận.",
 "Điểm lấy {loc} thuộc danh sách bị hủy; robot không cần nhận thêm kiện.",
 "Sau khi hủy lệnh khác, yêu cầu lấy tại {loc} cũng bị hủy, khỏi nhận.",
 "Chặng nhận tại {loc} từng bị hủy, hiện vẫn hủy và không lấy.",
 "Không ghé lấy ở {loc}; thông báo hủy áp dụng chặng này, đã nhận đủ hàng.")),
# Revision/last instruction wins
(
("Trước đây hủy nhận ở {loc}, nay yêu cầu mới khôi phục chặng lấy.",
 "Thông báo mới bảo phải ghé {loc} lấy kiện, thay thế thông báo hủy trước đó.",
 "Sau khi đính chính, lệnh nhận tại {loc} có hiệu lực; không bỏ bước lấy hàng.",
 "Điểm nhận ở {loc} theo phiên bản mới, cần lấy hàng trước khi giao.",
 "Yêu cầu sửa đổi không hủy bước lấy ở {loc}; phải ghé nhận.",
 "Sau thông báo hủy trước đó, lệnh mới yêu cầu lấy tại {loc}.",
 "Bản điều chỉnh khẳng định chặng nhận tại {loc} phải làm, không bỏ bước lấy.",
 "Chỉ dẫn cuối cập nhật: nhận kiện tại {loc}, rồi giao sau."),
("Trước đây nhận ở {loc}, nay yêu cầu mới hủy chặng lấy.",
 "Thông báo mới bảo hủy việc ghé {loc} lấy kiện, thay thế thông báo trước đó.",
 "Sau khi đính chính, lệnh nhận tại {loc} hết hiệu lực; không lấy hàng.",
 "Điểm nhận ở {loc} chỉ nằm ở phiên bản cũ, đã lấy đủ hàng trước khi giao.",
 "Yêu cầu sửa đổi đã hủy bước lấy ở {loc}; không ghé nhận.",
 "Sau thông báo hủy trước đó, lệnh mới bỏ yêu cầu lấy tại {loc}.",
 "Bản điều chỉnh khẳng định chặng nhận tại {loc} đã xóa, không làm bước lấy.",
 "Chỉ dẫn cuối cập nhật: nhận kiện đã đủ, chỉ đi ngang {loc} rồi giao sau.")),
]
URGENT=(
("Yêu cầu giao gấp không bị hủy; cần ưu tiên ngay.",
 "Lệnh khẩn đang còn hiệu lực, giao ngay.",
 "Trước hạn chót, không trì hoãn, phải giao gấp.",
 "Ưu tiên cao được kích hoạt, không xếp lịch thường.",
 "Thông báo mới xác nhận trạng thái khẩn, xử lý ngay.",
 "Nay phải giao gấp, không giữ lịch cũ.",
 "Bên gửi hủy lịch thường và yêu cầu chạy hỏa tốc, ưu tiên ngay.",
 "Cập nhật cuối cùng buộc xử lý khẩn, không chờ."),
("Yêu cầu giao gấp đã bị hủy; không cần ưu tiên.",
 "Lệnh khẩn đã hết hiệu lực, giao thường.",
 "Trước hạn chót đã nới, không cần giao gấp.",
 "Ưu tiên cao đã bị gỡ, không phải xử lý đặc biệt.",
 "Thông báo mới bác bỏ trạng thái khẩn, xử lý bình thường.",
 "Nay bỏ yêu cầu giao gấp, không thay đổi lịch thường.",
 "Bên gửi hủy yêu cầu chạy hỏa tốc, chuyển lịch thường.",
 "Cập nhật cuối cùng bỏ xử lý khẩn, không cần ưu tiên."),
)
FRAGILE=(
("Trong kiện có vật dễ vỡ, không được va mạnh.",
 "Hàng nhạy va chạm, không được rung lắc.",
 "Yêu cầu bảo vệ đồ dễ bể vẫn còn hiệu lực.",
 "Thông tin có kính dễ vỡ là chính xác; không được bỏ lớp đệm.",
 "Đồ mong manh dễ hỏng; phải chống rung.",
 "Không hủy quy tắc chống va đập, vì hàng dễ vỡ.",
 "Trạng thái nhạy va chạm được xác nhận, xin nhẹ tay.",
 "Đính chính: kiện chứa đồ dễ vỡ; không được xóc."),
("Trong kiện không có vật dễ vỡ, có thể xếp thường.",
 "Hàng không nhạy va chạm, được vận chuyển thường.",
 "Yêu cầu bảo vệ đồ dễ bể đã hết hiệu lực.",
 "Thông tin có kính dễ vỡ là sai; không cần lớp đệm.",
 "Đồ mong manh dễ hỏng chỉ là ghi nhầm; hàng chịu rung.",
 "Hủy quy tắc chống va đập, vì không có hàng dễ vỡ.",
 "Trạng thái nhạy va chạm bị bác bỏ, xin vận chuyển thường.",
 "Đính chính: kiện không chứa đồ dễ vỡ; không cần chống xóc."),
)
CUES=("khong","huy","nhan","lay","truoc","sau","gap","khan","vo","nhay")
REGIME=("plain","cancelled","revision")
SEED=2026101077
GROUPS=1600
DEST={
"candidate_train":WORK/"experiments/semantic_v7_balanced_train.jsonl",
"family_challenge":WORK/"experiments/semantic_v7_family_challenge.jsonl"}
MANIFEST=WORK/"checkpoints/semantic_v7_dataset_manifest.json"
for p in (*DEST.values(),MANIFEST):
 if p.exists():raise FileExistsError(str(p))
assert len(GOAL)==len(URGENT[0])==len(URGENT[1])==len(FRAGILE[0])==len(FRAGILE[1])==8
assert all(len(v)==2 and all(len(y)==8 for y in v) for v in VIA)
def cue_set(s):
 return set(re.findall(r"[a-z0-9]+",fold(s)))
def specdict(x):
 return None if x is None else {"type":x.type,"ref":x.ref,"anchor":x.anchor}
r=random.Random(SEED);out=collections.defaultdict(list);reg=collections.defaultdict(collections.Counter)
counts=collections.defaultdict(collections.Counter);cuesplit=collections.defaultdict(lambda:collections.Counter())
modes=collections.defaultdict(collections.Counter)
all_seen=set()
start=time.time()
gmodes=["named"]*6+["near","far","anchor_near","north","south","east","west","north_most","south_most"]
vmodes=["named"]*6+["near","far","north","south","east","west"]
for i in range(GROUPS):
 fam=i%8;ri=(i//8)%3;style=r.randrange(4);accented=r.random()<.57
 g=V.G.Generator(seed=r.randrange(2**31),holdout=False,mode="hard")
 goal,gloc,used=V.get_target(g,r.choice(gmodes),set(),accented)
 via,vloc,_=V.get_target(g,r.choice(vmodes),used,accented)
 split="candidate_train" if fam<=5 else "family_challenge"
 scene=[]
 for active in (False,True):
  for urgent in (False,True):
   for fragile in (False,True):
    # Each semantic head is expressed independently: goal same for the 8 cases.
    clauses=[GOAL[fam].format(loc=gloc),VIA[ri][0 if active else 1][fam].format(loc=vloc),
             URGENT[0 if urgent else 1][fam],FRAGILE[0 if fragile else 1][fam]]
    if style==1:clauses=[clauses[1],clauses[0],clauses[2],clauses[3]]
    elif style==2:clauses=[clauses[2],clauses[0],clauses[1],clauses[3]]
    elif style==3:clauses=[clauses[0],clauses[3],clauses[1],clauses[2]]
    text=" ".join(clauses)
    if not accented:text=fold(text)
    tokens=len(tokenize(text))
    if not(12<=tokens<=MAX_TOKENS):raise ValueError(("TOKEN_LIMIT",i,fam,ri,tokens,MAX_TOKENS,text))
    label_via=via if active else None
    assert len(spec_labels(goal,label_via,urgent,fragile))==8
    canon=fold(text).strip()
    if canon in all_seen:raise ValueError(("DUPLICATE_TEXT",i,canon))
    all_seen.add(canon)
    row={"id":f"sf7_{i:05d}_{int(active)}{int(urgent)}{int(fragile)}",
         "group_id":f"scene_{i:05d}","split":split,"family_id":fam,
         "text":text,"goal":specdict(goal),"via":specdict(label_via),
         "urgent":urgent,"fragile":fragile,
         "provenance":{"generator":"semantic_v7","seed":SEED,"family_id":fam,
            "regime":REGIME[ri],"goal_mode":goal.ref or "named",
            "via_mode":via.ref or "named","accented":accented,"tokens":tokens,
            "active_via":active,"order":style}}
    scene.append(row)
 assert len(scene)==8
 goal_set={json.dumps(z["goal"],sort_keys=True) for z in scene}
 assert len(goal_set)==1
 for k in ("via","urgent","fragile"):
  assert sum(bool(row[k]) for row in scene)==4
 # Strict new invariant: no listed lexical cue may be changed by flipping
 # via/urgent/fragile labels anywhere in this 8-way scene.
 sig=[tuple(k in cue_set(x["text"]) for k in CUES) for x in scene]
 if len(set(sig))>1:
  bad={c:sum(c in cue_set(x["text"]) for x in scene) for c in CUES
       if len(set(c in cue_set(x["text"]) for x in scene))>1}
  raise AssertionError(("CUE_LEAKAGE",i,fam,REGIME[ri],bad,
                       [(z["id"],z["text"]) for z in scene]))
 out[split].extend(scene);reg[split][REGIME[ri]]+=8
 for z in scene:
  counts[split]["n"]+=1;counts[split]["accented"]+=accented
  for k in ("via","urgent","fragile"):counts[split][k]+=bool(z[k])
  modes[split]["goal_"+(goal.ref or "named")]+=1
  modes[split]["via_"+(via.ref or "named")]+=1
  cues0=cue_set(z["text"])
  for cue in CUES:
   if cue in cues0:
    cuesplit[split][cue+"_support"]+=1
    for k in ("via","urgent","fragile"):cuesplit[split][cue+"_"+k]+=bool(z[k])
assert len(out["candidate_train"])==GROUPS*6 and len(out["family_challenge"])==GROUPS*2
# Protect against direct phrase memorization: no normalized exact overlap to
# earlier V5 synthetic train baseline (first 200,000 examples).
old=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch\scratch_s0_e01_seed2026110501.jsonl")
overlap=0
with old.open(encoding="utf-8") as f:
 for s in f:overlap+=fold(json.loads(s)["text"]).strip() in all_seen
assert overlap==0
audit={}
for split,path in DEST.items():
 n=len(out[split]);c=counts[split];fe=cuesplit[split]
 assert n and all(c[k]==n//2 for k in ("via","urgent","fragile"))
 with path.open("x",encoding="utf-8",newline="\n") as f:
  for row in out[split]:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 audit[split]={"rows":n,"groups":n//8,"regime_counts":dict(reg[split]),
  "accented_fraction":round(c["accented"]/n,4),
  "mode_counts":dict(modes[split]),
  "cue_conditional":{cue:{"support":fe[cue+"_support"],
       **{key:(round(fe[cue+"_"+key]/fe[cue+"_support"],4) if fe[cue+"_support"] else None)
          for key in ("via","urgent","fragile")}} for cue in CUES},
  "label_rate":{k:c[k]/n for k in ("via","urgent","fragile")},
  "path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
manifest={"name":"semantic_v7_cue_invariant_counterfactuals","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "seed":SEED,"groups":GROUPS,"total_rows":GROUPS*8,
 "cue_presence_invariant_within_8way_scene":list(CUES),
 "split_family":{"candidate_train":[0,1,2,3,4,5],"family_challenge":[6,7]},
 "splits":audit,"old_scratch_200k_overlap":overlap,
 "qa":["All 8 label triplets per scene","3 pickup regimes with independent goal/urgent/fragile",
       "CUE INVARIANT within each scene across all 8 label combinations",
       "no normalized exact duplicates","no prior scratch epoch1 text duplicates",
       "valid spec_labels and MAX_TOKENS","family-disjoint challenge"],
 "limits":["Lexical-cue independence alone does not prove semantic comprehension",
           "Generator-generated labels require human spot QA",
           "Evaluation split uses same concepts and lexicon as train",
           "Train-only data cannot include family_challenge split",
           "No official validation/test read; no model trained or modified"],
 "seconds":round(time.time()-start,1)}
with MANIFEST.open("x",encoding="utf-8") as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print("V7_SUCCESS",json.dumps({"rows":{k:len(v) for k,v in out.items()},
"regime":{k:dict(v) for k,v in reg.items()},
"cue_conditional":{k:au["cue_conditional"] for k,au in audit.items()},
"manifest":str(MANIFEST),"seconds":manifest["seconds"]},ensure_ascii=False),flush=True)
