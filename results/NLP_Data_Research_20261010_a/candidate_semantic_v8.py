"""V8 counterfactual scenes: eliminate V7's negation-word saturation.
Creates only NEW artifacts inside research workspace. Imports old models/generator
read-only. Cue presence is invariant over all 8 label combinations in a scene.
"""
from __future__ import annotations
import ast,collections,datetime,hashlib,importlib.util,json,random,re,sys,time
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import spec_labels,MAX_TOKENS
WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
f=WORK/"candidate_semantic_v7.py"
tree=ast.parse(f.read_text(encoding="utf-8"))
saved={}
for node in tree.body:
 if isinstance(node,ast.Assign):
  for x in node.targets:
   if isinstance(x,ast.Name) and x.id in ("VIA","URGENT","FRAGILE","CUES"):
    saved[x.id]=ast.literal_eval(node.value)
assert len(saved)==4
RICH_VIA=saved["VIA"];RICH_URGENT=saved["URGENT"];RICH_FRAGILE=saved["FRAGILE"]
CUES=tuple(saved["CUES"])
lib=importlib.util.spec_from_file_location("semantic_v5_helper_for_v8",WORK/"candidate_semantic_v5.py")
V=importlib.util.module_from_spec(lib);sys.modules[lib.name]=V;lib.loader.exec_module(V)
MILD_VIA=(
(
("Robot ghé {loc} lấy kiện bổ sung trước khi giao.",
 "Điểm nhận thêm tài liệu là {loc}; robot tới lấy rồi mới giao.",
 "Chặng nhận thêm tài liệu tại {loc} diễn ra trước khi giao.",
 "Có bước ghé {loc} lấy chứng từ; robot nhận xong rồi giao.",
 "Điểm lấy thêm kiện tại {loc} đã được xác nhận; chặng nhận cần hoàn tất.",
 "Sau khi kiểm tra, địa điểm {loc} là nơi lấy hàng bổ sung.",
 "Chặng nhận bổ sung ở {loc} có hiệu lực; robot đến lấy rồi giao.",
 "Trước khi giao, đến {loc} nhận thêm một phong bì."),
("Robot đã lấy đủ kiện trước khi giao; {loc} chỉ dùng để tham chiếu.",
 "Thông tin về điểm nhận {loc} là lịch sử; tài liệu đã lấy đủ tại chỗ xuất phát.",
 "Chặng nhận thêm tài liệu trước khi giao đã hoàn tất tại chỗ xuất phát; {loc} chỉ là mốc.",
 "Thông tin ghé {loc} lấy chứng từ chỉ là ghi chú; robot đã nhận đủ từ đầu.",
 "Điểm lấy thêm kiện tại {loc} là hướng dẫn cũ; chặng nhận đã hoàn tất tại nơi xuất phát.",
 "Sau khi kiểm tra, địa điểm {loc} chỉ là mốc; hàng đã lấy đủ.",
 "Chặng nhận bổ sung ở {loc} chỉ là thông tin đơn khác; hàng đã lấy đủ.",
 "Trước khi giao, việc nhận phong bì đã xong tại điểm xuất phát; {loc} chỉ là mốc.")),
(
("Đã hủy chặng khác, riêng việc lấy tại {loc} vẫn cần làm.",
 "Lệnh hủy cũ chỉ áp dụng điểm khác; chặng nhận tại {loc} còn hiệu lực.",
 "Thông báo hủy việc lấy tại {loc} là sai; robot vẫn phải tới nhận.",
 "Trước đó báo hủy chặng nhận ở {loc}, nhưng quyết định cuối yêu cầu tới lấy.",
 "Phiếu hủy lệnh cũ, còn việc nhận tại {loc} vẫn cần làm.",
 "Sau khi hủy lệnh khác, robot vẫn phải lấy chứng từ tại {loc}.",
 "Chặng nhận tại {loc} từng bị hủy nhưng đã phục hồi, robot tới lấy.",
 "Phiếu hủy chặng cũ, song điểm nhận tại {loc} vẫn đang hoạt động."),
("Đã hủy chặng khác, việc lấy tại {loc} cũng bị hủy.",
 "Lệnh hủy cũ áp dụng cả chặng nhận tại {loc}; hàng đã đủ.",
 "Thông báo hủy việc lấy tại {loc} là chính xác; robot đã nhận đủ.",
 "Trước đó báo hủy chặng nhận ở {loc}, quyết định cuối vẫn hủy bước lấy.",
 "Phiếu hủy lệnh cũ và cả việc nhận tại {loc}.",
 "Sau khi hủy lệnh khác, robot cũng bỏ việc lấy chứng từ tại {loc}.",
 "Chặng nhận tại {loc} từng bị hủy và vẫn bị hủy, hàng đã lấy đủ.",
 "Phiếu hủy chặng cũ, đồng thời điểm nhận tại {loc} bị xóa khỏi tuyến.")),
(
("Trước kia yêu cầu hủy chặng lấy tại {loc}; lệnh mới đã phục hồi.",
 "Bản trước hủy bước ghé {loc} lấy hàng; bản mới khôi phục.",
 "Sau đính chính, việc nhận thêm tại {loc} thành bắt buộc.",
 "Chỉ dẫn trước ghi hủy nhận tại {loc}; chỉ dẫn cuối khôi phục việc lấy.",
 "Bản cập nhật phục hồi chặng nhận ở {loc}; robot phải lấy thêm kiện.",
 "Sau khi sửa phiếu, yêu cầu lấy tại {loc} trở lại hiệu lực.",
 "Thay đổi cuối cùng phục hồi bước nhận tại {loc}; cần đến lấy.",
 "Tin cuối xác nhận bước lấy tại {loc}; robot thực hiện trước khi giao."),
("Trước kia có chặng lấy tại {loc}; lệnh mới yêu cầu hủy.",
 "Bản trước có bước ghé {loc} lấy hàng; bản mới hủy.",
 "Sau đính chính, việc nhận thêm tại {loc} chỉ là ghi chú.",
 "Chỉ dẫn trước ghi nhận tại {loc}; chỉ dẫn cuối hủy việc lấy.",
 "Bản cập nhật loại bỏ chặng nhận ở {loc}; kiện đã lấy đủ.",
 "Sau khi sửa phiếu, yêu cầu lấy tại {loc} chỉ còn là ghi chú.",
 "Thay đổi cuối cùng xóa bước nhận tại {loc}; kiện đã lấy đủ.",
 "Tin cuối nói bước lấy tại {loc} đã hoàn tất tại nguồn trước khi giao.")),
)
MILD_URGENT=(
("Trạng thái giao gấp có hiệu lực; ưu tiên ngay.",
 "Lệnh khẩn đang hiệu lực; xử lý sớm.",
 "Yêu cầu giao gấp đã được kích hoạt.",
 "Ưu tiên cao đã được duyệt, xử lý ngay.",
 "Điều chỉnh mới yêu cầu giao khẩn, bắt đầu ngay.",
 "Lịch gửi được đổi thành giao gấp.",
 "Người gửi chuyển đơn sang hỏa tốc, ưu tiên xử lý.",
 "Quyết định cuối xác nhận mức khẩn, ưu tiên ngay."),
("Trạng thái giao gấp đã hết hiệu lực; theo lịch thường.",
 "Lệnh khẩn đã hết hiệu lực; theo lịch thường.",
 "Yêu cầu giao gấp chỉ là bản nháp cũ.",
 "Ưu tiên cao chỉ là đề xuất cũ, xử lý theo lịch.",
 "Điều chỉnh mới loại bỏ trạng thái khẩn, xử lý thường.",
 "Lịch gửi được đổi từ giao gấp sang giao thường.",
 "Người gửi chuyển đơn khỏi chế độ hỏa tốc, xử lý thường.",
 "Quyết định cuối gỡ mức khẩn, theo lịch."),
)
MILD_FRAGILE=(
("Kiện chứa vật dễ vỡ, cần chống xóc.",
 "Hàng nhạy va chạm, phải nâng nhẹ.",
 "Quy định vận chuyển đồ dễ bể vẫn còn hiệu lực.",
 "Nhãn dễ vỡ đã được xác nhận, robot cần kê đệm.",
 "Đồ mong manh cần chống rung.",
 "Yêu cầu chống va đập dành cho hàng dễ vỡ vẫn hiệu lực.",
 "Trạng thái nhạy va chạm đã xác nhận, phải nâng nhẹ.",
 "Bản đính chính xác nhận có đồ dễ vỡ trong kiện."),
("Kiện được ghi dễ vỡ trong phiếu cũ, thực tế là hàng bền.",
 "Hàng nhạy va chạm là mô tả nhầm; thực tế chịu rung.",
 "Quy định vận chuyển đồ dễ bể đã hết hiệu lực.",
 "Nhãn dễ vỡ đã được gỡ, robot vận chuyển bình thường.",
 "Đồ mong manh là nội dung ghi nhầm, thực tế chịu rung.",
 "Yêu cầu chống va đập dành cho hàng dễ vỡ đã hết hiệu lực.",
 "Trạng thái nhạy va chạm đã gỡ, vận chuyển bình thường.",
 "Bản đính chính xác nhận nhãn đồ dễ vỡ là nhầm."),
)
assert len(MILD_VIA)==3 and all(len(x)==2 and all(len(y)==8 for y in x) for x in MILD_VIA)
assert all(len(x)==8 for x in (*MILD_URGENT,*MILD_FRAGILE))
SEED=2026101088
GROUPS=1600
OUT={
 "candidate_train":WORK/"experiments/semantic_v8_balanced_train.jsonl",
 "family_challenge":WORK/"experiments/semantic_v8_family_challenge.jsonl"}
MANIFEST=WORK/"checkpoints/semantic_v8_dataset_manifest.json"
for p in (*OUT.values(),MANIFEST):
 if p.exists():raise FileExistsError(str(p))
def toks(s):
 return set(re.findall(r"[a-z0-9]+",fold(s)))
def specdict(q):
 return None if q is None else {"type":q.type,"ref":q.ref,"anchor":q.anchor}
rng=random.Random(SEED);t0=time.time()
rows=collections.defaultdict(list);stat=collections.defaultdict(collections.Counter)
cuecounts=collections.defaultdict(collections.Counter)
seen=set();all_groups=set()
go_modes=["named"]*6+["near","far","anchor_near","north","south","east","west","north_most","south_most"]
vi_modes=["named"]*6+["near","far","north","south","east","west"]
for i in range(GROUPS):
 fam=i%8;reg=(i//8)%3
 rich_style=((i//24)%2==0)  # deterministic 50% of each family+regime
 accented=rng.random()<.56
 order=rng.randrange(4)
 g=V.G.Generator(seed=rng.randrange(2**31),holdout=False,mode="hard")
 gs,gtext,used=V.get_target(g,rng.choice(go_modes),set(),accented)
 vs,vtext,_=V.get_target(g,rng.choice(vi_modes),used,accented)
 s="candidate_train" if fam<6 else "family_challenge"
 pv=RICH_VIA if rich_style else MILD_VIA
 uf=RICH_URGENT if rich_style else MILD_URGENT
 ff=RICH_FRAGILE if rich_style else MILD_FRAGILE
 scene=[]
 for active in (False,True):
  for urgent in (False,True):
   for fragile in (False,True):
    phrases=[
      V.GOAL[fam].format(loc=gtext),
      pv[reg][0 if active else 1][fam].format(loc=vtext),
      uf[0 if urgent else 1][fam],
      ff[0 if fragile else 1][fam]]
    if order==1:phrases=[phrases[1],phrases[0],phrases[2],phrases[3]]
    elif order==2:phrases=[phrases[2],phrases[0],phrases[1],phrases[3]]
    elif order==3:phrases=[phrases[0],phrases[3],phrases[1],phrases[2]]
    text=" ".join(phrases)
    if not accented:text=fold(text)
    n=len(tokenize(text))
    if not 12<=n<=MAX_TOKENS:raise RuntimeError(("TOKEN_LIMIT",i,n,text))
    assert len(spec_labels(gs,vs if active else None,urgent,fragile))==8
    foldtext=fold(text).strip()
    if foldtext in seen:raise RuntimeError(("DUPLICATE",i,foldtext))
    seen.add(foldtext)
    scene.append({"id":f"sf8_{i:05d}_{int(active)}{int(urgent)}{int(fragile)}",
      "group_id":f"scene_{i:05d}","family_id":fam,"split":s,
      "text":text,"goal":specdict(gs),"via":specdict(vs) if active else None,
      "urgent":urgent,"fragile":fragile,
      "provenance":{"generator":"semantic_v8","seed":SEED,
         "family_id":fam,"regime":("plain","cancelled","revision")[reg],
         "phrase_style":"negation_rich" if rich_style else "plain_lexicon",
         "goal_mode":gs.ref or "named","via_mode":vs.ref or "named",
         "accented":accented,"order":order,"tokens":n}})
 assert len(scene)==8 and len({json.dumps(x["goal"],sort_keys=True) for x in scene})==1
 for head in ("via","urgent","fragile"):
  assert sum(bool(z[head]) for z in scene)==4
 for cue in CUES:
  indicator=[cue in toks(z["text"]) for z in scene]
  if len(set(indicator))>1:
   raise AssertionError(("CUE_MISMATCH",i,fam,reg,rich_style,cue,
              [(z["id"],z["text"]) for z in scene]))
 rows[s].extend(scene)
 stat[s]["groups"]+=1;stat[s]["rich_groups"]+=rich_style
 stat[s]["mild_groups"]+=not rich_style
 stat[s]["regime_"+("plain","cancelled","revision")[reg]]+=1
 for z in scene:
  stat[s]["rows"]+=1
  stat[s]["accented"]+=accented
  for k in ("via","urgent","fragile"):stat[s][k]+=bool(z[k])
  for cue in CUES:
   if cue in toks(z["text"]):
    cuecounts[s][cue+"_support"]+=1
    for k in ("via","urgent","fragile"):cuecounts[s][cue+"_"+k]+=bool(z[k])
assert len(rows["candidate_train"])==9600 and len(rows["family_challenge"])==3200
audit={}
for s,path in OUT.items():
 n=len(rows[s]);c=stat[s];fc=cuecounts[s]
 assert all(c[k]==n//2 for k in ("via","urgent","fragile"))
 with path.open("x",encoding="utf-8",newline="\n") as f:
  for z in rows[s]:f.write(json.dumps(z,ensure_ascii=False)+"\n")
 audit[s]={"rows":n,"groups":c["groups"],"rich_groups":c["rich_groups"],
  "mild_groups":c["mild_groups"],
  "regime_groups":{reg:c["regime_"+reg] for reg in ("plain","cancelled","revision")},
  "accented_fraction":round(c["accented"]/n,4),
  "label_positive_rate":{h:c[h]/n for h in ("via","urgent","fragile")},
  "cue_conditional":{cue:{"support":fc[cue+"_support"],"prevalence":round(fc[cue+"_support"]/n,4),
    **{h:(round(fc[cue+"_"+h]/fc[cue+"_support"],4) if fc[cue+"_support"] else None)
       for h in ("via","urgent","fragile")}} for cue in CUES},
  "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"file":str(path)}
assert all(x["regime_groups"]["plain"]>=130 for x in audit.values())
manifest={"name":"semantic_v8_mixed_lexical_counterfactuals",
  "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "seed":SEED,"total_rows":GROUPS*8,"total_scenes":GROUPS,
  "split_families":{"candidate_train":[0,1,2,3,4,5],"family_challenge":[6,7]},
  "cue_invariance":list(CUES),"splits":audit,
  "qa":["8 combinations per scene","3 instruction regimes",
        "both negation-rich and normal no-negation template styles",
        "invariant cue presence across 8 labels for each scene",
        "no duplicate normalized text","bounded text length","typed labels"],
  "limitations":["No human linguistic audit","No training-based efficacy check yet",
   "Heldout families use familiar semantic concepts","Some spatial targets use negation intrinsically",
   "All writes confined to research workspace"],"elapsed_s":round(time.time()-t0,1)}
with MANIFEST.open("x",encoding="utf-8") as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print("V8_CREATED",json.dumps({"audit":{s:{"rows":v["rows"],"style":(v["rich_groups"],v["mild_groups"]),
 "khong":v["cue_conditional"]["khong"],"huy":v["cue_conditional"]["huy"],
 "nhan":v["cue_conditional"]["nhan"]} for s,v in audit.items()},
 "manifest":str(MANIFEST),"elapsed_s":manifest["elapsed_s"]},ensure_ascii=False),flush=True)
