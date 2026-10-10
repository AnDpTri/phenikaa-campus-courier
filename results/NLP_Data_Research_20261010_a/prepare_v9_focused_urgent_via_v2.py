"""V9 focused repair: paired urgent and via examples, train/diagnostic template split.
No official validation/test. Creates files inside research workspace only.
The independent diagnostic templates are never used in model training.
"""
from __future__ import annotations
import collections,datetime,hashlib,json,random,re,statistics
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS,spec_labels
from courier.nlp.parser import TargetSpec
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
TRAIN=W/"experiments/semantic_v9_focused_urgent_via_train_v2.jsonl"
CHALLENGE=W/"experiments/semantic_v9_focused_urgent_via_holdout_v2.jsonl"
MAN=W/"checkpoints/semantic_v9_focused_urgent_via_manifest_v2.json"
for p in (TRAIN,CHALLENGE,MAN):
 if p.exists():raise FileExistsError("Refusing overwrite: "+str(p))
GOALS=(("library","thư viện"),("dorm","ký túc xá"),("sports","nhà thi đấu"),
       ("clinic","trạm y tế"),("canteen","nhà ăn"),("parking","bãi xe"),
       ("lecture","giảng đường"),("lab","phòng thí nghiệm"),
       ("office","văn phòng khoa"),("gate","cổng chính"))
GOALFRAMES=("Đưa bưu kiện cuối cùng tới {goal}.",
 "Địa điểm cần giao sau mọi bước là {goal}.",
 "Bàn giao kiện tại {goal} theo phiếu chính thức.",
 "Lệnh giao cuối cùng ghi đích là {goal}.",
 "Chuyến hàng này có điểm đến là {goal}.",
 "Sau khi xử lý các bước khác, giao tại {goal}.",
 "Tới {goal} để hoàn tất đơn giao.",
 "Nơi nhận bưu phẩm được xác định là {goal}.")
FRAGILE={
 True:("Trong kiện có chai thủy tinh dễ vỡ, cần kê đệm.",
       "Hàng có món gốm dễ nứt khi va đập.",
       "Đơn hàng chứa linh kiện mong manh, rất dễ hỏng nếu xóc."),
 False:("Kiện chỉ chứa các món đồ bền chịu va chạm tốt.",
        "Hàng là vật liệu không dễ vỡ và chịu rung bình thường.",
        "Thùng chỉ chứa đồ mềm không có thứ dễ nứt vỡ.")}
VIA_CONTEXT={
 True:("Còn cần ghé {via} lấy phong bì trước khi giao.",
       "Phải tới {via} nhận thêm tài liệu rồi mới giao."),
 False:("Đã nhận đủ từ kho ban đầu, không cần đi lấy ở nơi khác.",
        "Phiếu nhận thêm đã hủy hoàn toàn; đi thẳng tới điểm giao.")}
URGENT_CONTEXT={
 True:("Đơn này đã chốt giao hỏa tốc ngay.",
       "Độ ưu tiên cuối của đơn là giao gấp."),
 False:("Đơn này được chốt giao theo lịch thường.",
        "Đã hủy chế độ khẩn; giao bình thường.")}
U_TRAIN=(
("Dù có ghi chú giao thường cũ, quyết định cuối yêu cầu giao khẩn ngay.",
 "Dù có ghi chú giao khẩn cũ, quyết định cuối yêu cầu giao thường."),
("Phiếu cũ ghi xử lý thường; tin cuối đổi sang hỏa tốc.",
 "Phiếu cũ ghi xử lý hỏa tốc; tin cuối đổi sang thường."),
("Không hủy lệnh khẩn, vẫn cần giao ngay.",
 "Đã hủy lệnh khẩn, không cần giao ngay."),
("Nhãn 'giao thường' trong bản nháp đã bị bác, chốt mức khẩn.",
 "Nhãn 'giao khẩn' trong bản nháp đã bị bác, chốt mức thường."),
("Một đơn khác giao thường nhưng đơn này phải giao hỏa tốc.",
 "Một đơn khác giao hỏa tốc nhưng đơn này được giao thường."),
("Người nhận không còn đồng ý trì hoãn, kiện này phải đến gấp.",
 "Người nhận đã đồng ý trì hoãn, kiện này không cần đến gấp."),
("Lệnh hỏa tốc chưa bị rút, ưu tiên chuyển ngay.",
 "Lệnh hỏa tốc đã bị rút, không ưu tiên chuyển ngay."),
("Dòng cuối sửa 'không gấp' thành 'gấp', thực hiện theo dòng cuối.",
 "Dòng cuối sửa 'gấp' thành 'không gấp', thực hiện theo dòng cuối."),
("Đừng theo lịch thường cũ; áp dụng lịch chuyển cấp tốc mới.",
 "Đừng theo lịch cấp tốc cũ; áp dụng lịch chuyển thường mới."),
("Ưu tiên hiện hành là khẩn; lịch thường trước đây đã bỏ.",
 "Ưu tiên hiện hành là thường; lịch khẩn trước đây đã bỏ."),
("Chỉ dẫn cuối tái kích hoạt chuyển nhanh, cần ưu tiên.",
 "Chỉ dẫn cuối vô hiệu hóa chuyển nhanh, không cần ưu tiên."),
("Lệnh giao gấp không thuộc thông báo hủy, tiếp tục thực hiện.",
 "Lệnh giao gấp thuộc thông báo hủy, thôi thực hiện."))
V_TRAIN=(
("Mặc dù bảng tin nói hủy, điểm lấy ở {via} vừa được khôi phục; phải ghé.",
 "Mặc dù bảng tin nói khôi phục, điểm lấy ở {via} vừa bị hủy; không ghé."),
("Không xóa yêu cầu nhận chứng từ tại {via}; giữ chặng này.",
 "Đã xóa yêu cầu nhận chứng từ tại {via}; bỏ chặng này."),
("Nhận thêm hồ sơ tại {via} là lệnh cuối, chưa thực hiện.",
 "Nhận thêm hồ sơ tại {via} là ghi chú cũ, đã thực hiện ở kho xuất."),
("Phiếu nháp bỏ bước nhận tại {via}; bản chính yêu cầu thực hiện.",
 "Phiếu nháp yêu cầu nhận tại {via}; bản chính bỏ bước này."),
("Lời dặn 'đi thẳng' bị thu hồi; robot cần đến {via} nhận.",
 "Lời dặn 'ghé lấy' bị thu hồi; robot đi thẳng, bỏ {via}."),
("Đến {via} để lấy bổ sung, không phải mốc tham chiếu.",
 "{via} là mốc tham chiếu, không cần đến đó lấy bổ sung."),
("Đã nhận đủ tại kho chính nhưng còn phải ghé {via} lấy kiện còn lại.",
 "Đã nhận đủ tại kho chính, không phải ghé {via} lấy kiện còn lại."),
("Lệnh hủy chỉ áp dụng điểm khác; {via} vẫn là điểm lấy bắt buộc.",
 "Lệnh hủy áp dụng luôn {via}, không còn điểm lấy bắt buộc."),
("Người giao xác nhận còn thiếu phong bì ở {via}; ghé lấy trước.",
 "Người giao xác nhận đã có đủ phong bì; {via} chỉ có trong lời nhắc cũ."),
("Dòng cuối sửa 'không nhận ở {via}' thành 'nhận ở {via}'.",
 "Dòng cuối sửa 'nhận ở {via}' thành 'không nhận ở {via}'."),
("Đừng bỏ chặng nhận ở {via}, nó vẫn hiệu lực.",
 "Bỏ chặng nhận ở {via}, nó hết hiệu lực."),
("Cần tới {via} làm điểm nhận bổ sung, khác với điểm giao.",
 "Không cần tới {via} làm điểm nhận bổ sung, chỉ đến điểm giao."))
# Heldout templates authored separately, no exact overlap with training.
U_HOLD=(
("Ban điều phối chốt mã khẩn cho đơn này sau khi xem phiếu cũ ghi bình thường.",
 "Ban điều phối chốt mã thường cho đơn này sau khi xem phiếu cũ ghi khẩn."),
("Đối với đơn hiện tại, từ 'không ưu tiên' đã bị xóa và thay bằng 'ưu tiên ngay'.",
 "Đối với đơn hiện tại, từ 'ưu tiên ngay' đã bị xóa và thay bằng 'không ưu tiên'."),
("Lệnh cuối nhất vẫn yêu cầu chuyển cấp tốc; thông báo trì hoãn bị bác bỏ.",
 "Lệnh cuối nhất cho phép chuyển theo lịch; thông báo cấp tốc bị bác bỏ."),
("Mốc phải có mặt được đẩy lên sớm, người gửi chốt giao gấp.",
 "Mốc phải có mặt được lùi xuống, người gửi chốt giao thường."),
("Điều kiện 'giao bình thường' không còn hợp lệ, bật chế độ xử lý khẩn.",
 "Điều kiện 'giao khẩn' không còn hợp lệ, bật chế độ xử lý thường."),
("Không theo yêu cầu chờ từ thông báo nháp; đơn phải đi ngay.",
 "Không theo yêu cầu đi ngay từ thông báo nháp; đơn được phép chờ."))
V_HOLD=(
("Biên bản mới ghi rõ nơi nhận bổ sung tại {via} còn phải đi tới.",
 "Biên bản mới ghi rõ nơi nhận bổ sung tại {via} đã giải quyết xong ở kho."),
("Tin trước cho bỏ {via} nhưng bản điều phối cuối buộc đến đó lấy kiện.",
 "Tin trước buộc ghé {via} nhưng bản điều phối cuối cho bỏ chặng lấy."),
("Chưa có chứng từ từ {via}; cần nhận đủ rồi mới tới đích.",
 "Chứng từ từ {via} đã có sẵn trên xe; tới đích không cần ghé lấy."),
("Vẫn lấy ở {via}, dù các bước nhận khác được loại khỏi lịch.",
 "Không lấy ở {via}, chặng này cũng bị loại khỏi lịch."),
("Từ 'bỏ qua' cạnh {via} là lỗi in; phải tới lấy gói bổ sung.",
 "Từ 'phải tới lấy' cạnh {via} là lỗi in; phải bỏ qua."),
("Người gửi phục hồi địa điểm nhận {via} vào hành trình cuối.",
 "Người gửi loại địa điểm nhận {via} khỏi hành trình cuối."))
assert all(len(p)==2 for p in (*U_TRAIN,*V_TRAIN,*U_HOLD,*V_HOLD))
rng=random.Random(202610102607)
def make_record(focus,flag,idx,holdout):
 pair=U_HOLD if holdout and focus=="urgent" else V_HOLD if holdout else U_TRAIN if focus=="urgent" else V_TRAIN
 gidx=rng.randrange(len(GOALS))
 vidx=rng.choice([i for i in range(len(GOALS)) if i!=gidx])
 gkind,gname=GOALS[gidx];vkind,vname=GOALS[vidx]
 u=rng.random()<.382 if focus=="via" else bool(flag)
 va=rng.random()<.327 if focus=="urgent" else bool(flag)
 f=rng.random()<.392
 clause=pair[idx%len(pair)][0 if flag else 1].format(via=vname)
 goal_clause=rng.choice(GOALFRAMES).format(goal=gname)
 fragile_clause=rng.choice(FRAGILE[f])
 extras=[]
 if focus=="urgent":
  extras.append(rng.choice(VIA_CONTEXT[va]).format(via=vname) if rng.random()<.8 else
     (("Ghé "+vname+" để nhận thêm kiện trước khi giao.") if va else "Không có bước lấy thêm nào còn hiệu lực."))
 else: extras.append(rng.choice(URGENT_CONTEXT[u]))
 extras.append(fragile_clause)
 order=idx%4
 if order==0:clauses=[goal_clause,clause,*extras]
 elif order==1:clauses=[clause,goal_clause,*extras]
 elif order==2:clauses=[goal_clause,*extras,clause]
 else:clauses=[extras[0],goal_clause,clause,extras[1]]
 accented=rng.random()<.706
 text=" ".join(clauses)
 if not accented:text=fold(text)
 goal=TargetSpec(gkind)
 via=TargetSpec(vkind) if va else None
 labels=spec_labels(goal,via,u,f)
 assert len(labels)==8
 toks=len(tokenize(text))
 assert 14<=toks<=MAX_TOKENS,(focus,idx,toks,text)
 return {"id":("v9fix_hold" if holdout else "v9fix_train")+f"_{focus}_{idx:04d}_{int(flag)}",
  "group_id":("held" if holdout else "train")+f"_{focus}_{idx:04d}",
  "text":text,"goal":{"type":gkind,"ref":None,"anchor":None},
  "via":({"type":vkind,"ref":None,"anchor":None} if va else None),
  "urgent":u,"fragile":f,
  "provenance":{"focus":focus,"pair_template":idx%len(pair),"accented":accented,
    "label_switch":bool(flag),"heldout":holdout,"length_tokens":toks}}
# Guarantee pair's nuisance shell matches: random state reused between flag mates,
# so only the semantically decisive paired clause and its label differ.
def create(n_groups,hold):
 rows=[]
 for focus in ("urgent","via"):
  for i in range(n_groups):
   saved=rng.getstate()
   yes=make_record(focus,True,i,hold)
   rng.setstate(saved)
   no=make_record(focus,False,i,hold)
   for attr in ("goal","fragile"):
    assert yes[attr]==no[attr],(focus,i,attr)
   if focus=="urgent":
    assert yes["via"]==no["via"]
   else:
    assert yes["urgent"]==no["urgent"]
   rows.extend((yes,no))
 return rows
train=create(240,False) # 960 rows, 480 per focus
held=create(48,True) # 192 rows, 96 per focus
for rows in (train,held):
 assert len(rows)==len({fold(z["text"]).strip() for z in rows})
# No normalized collision with existing V9 pilot or heldout V8 challenge.
existing={}
for name,fname in (("v9_pool","semantic_v9_real_like_pilot.jsonl"),
                   ("v8_family_holdout","semantic_v8_family_challenge.jsonl")):
 existing[name]=set(fold(json.loads(s)["text"]).strip() for s in
     (W/"experiments"/fname).open(encoding="utf-8"))
for setname,rows in (("train",train),("holdout",held)):
 forms={fold(z["text"]).strip() for z in rows}
 for ename,eset in existing.items():
  assert not forms&eset,(setname,ename,"overlap")
 existing[setname]=forms
assert not existing["train"]&existing["holdout"]
for path,rows in ((TRAIN,train),(CHALLENGE,held)):
 with path.open("x",encoding="utf-8",newline="\n") as out:
  for z in rows:out.write(json.dumps(z,ensure_ascii=False)+"\n")
stats={}
for k,rows in (("training_candidates",train),("independent_holdout",held)):
 cs=collections.Counter()
 lens=[]
 for z in rows:
  lens.append(z["provenance"]["length_tokens"])
  cs["urgent_pos"]+=z["urgent"]
  cs["via_pos"]+=z["via"] is not None
  cs["fragile_pos"]+=z["fragile"]
  cs["accented"]+=z["provenance"]["accented"]
  cs[z["provenance"]["focus"]]+=1
 stats[k]={"n":len(rows),"counts":dict(cs),"median_tokens":statistics.median(lens),
  "p90_tokens":sorted(lens)[int(.9*(len(lens)-1))],
  "sha256":hashlib.sha256((TRAIN if k=="training_candidates" else CHALLENGE).read_bytes()).hexdigest()}
manifest={"kind":"V9 focused semantic repair, paired training, independent challenge",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "seed":202610102607,"train":str(TRAIN),"challenge":str(CHALLENGE),
 "statistics":stats,"template_counts":{"urgent_train":len(U_TRAIN),"via_train":len(V_TRAIN),
 "urgent_holdout":len(U_HOLD),"via_holdout":len(V_HOLD)},
 "limitations":["Automatically authored counterfactuals, NOT manually approved each row",
 "Holdout uses separate templates but same construction system",
 "No official val/test ever read","Some short and narrow named-target distribution",
 "Train set must remain separate from both holdout and family challenge"]}
with MAN.open("x",encoding="utf-8") as out:json.dump(manifest,out,ensure_ascii=False,indent=2)
print("V9_FOCUSED_PREPARED",json.dumps(manifest,ensure_ascii=False),flush=True)
