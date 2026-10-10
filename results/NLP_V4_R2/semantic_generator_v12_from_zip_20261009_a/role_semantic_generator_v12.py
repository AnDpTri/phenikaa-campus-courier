"""V12 semantic-role generator: ideas from uploaded hard dataset, never copies its text.
NEW OUTPUTS ONLY; no earlier source/data or checkpoints are modified.
Builds explicitly superseded destinations/pickups and typed negative roles.
"""
from pathlib import Path
from collections import Counter
from courier.nlp.synth import GOAL_MODES,VIA_MODES
from courier.nlp.neural import spec_labels,MAX_TOKENS
from courier.nlp.text import fold,tokenize
import importlib.util,random,json,re,hashlib

HERE=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v12_from_zip_20261009_a")
BASE=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first\typed_semantic_generator.py")
OLD=Path(r"D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review")
N=2000
SEED=202610091212
OUT=HERE/"pilot_2000_v12_role_semantics.jsonl"
META=HERE/"pilot_2000_v12_provenance.jsonl"
REPORT=HERE/"pilot_2000_v12_generation_report.json"
r=random.Random(SEED)
sp=importlib.util.spec_from_file_location("v12_base_reader",BASE)
base=importlib.util.module_from_spec(sp)
sp.loader.exec_module(base)
base.rng=r
PLACES=base.PLACE
NEUTRAL=("gói hàng","thùng bưu phẩm","túi đồ","kiện hàng","hộp hàng","bưu kiện","hộp carton","túi chuyển phát","phong bì niêm phong","gói vật tư")
SUPPLEMENT=("tập biểu mẫu","phong bì chứng từ","phiếu giao hàng","bộ tài liệu phụ","thẻ ra vào","kiện giấy tờ","tập giấy bổ sung","túi tài liệu đính kèm")
DEST_CORRECTIONS=(
 "Phiếu đầu tiên yêu cầu giao {parcel} tới {old}. Chỉ thị ấy đã bị hủy.",
 "Ban đầu người gửi chọn {old} làm đích giao, nhưng vừa đổi địa chỉ và bỏ điểm cũ.",
 "Lệnh giao tới {old} không còn hiệu lực; đó là địa chỉ bị sửa.",
 "Trước đó robot được yêu cầu mang {parcel} tới {old}. Đính chính: không được giao theo lệnh cũ.",
 "Người nhận lúc đầu đặt điểm giao ở {old}; hiện tại đã hủy điểm ấy.",
 "Địa chỉ trong phiếu trước là {old}, nhưng người gửi vừa thay đích giao.",
 "Nhầm rồi, {old} là điểm giao trước khi điều chỉnh. Không dùng điểm ấy nữa.",
 "Hãy xóa khỏi lộ trình yêu cầu giao đến {old}; nơi giao chính thức nằm trong lệnh sau.",
)
PICKUP_CORRECTIONS_TRUE=(
 "Lúc đầu ghi phải lấy hàng ở {old}, nhưng điểm lấy này đã được hủy.",
 "Dự kiến nhận kiện tại {old}; xác nhận mới không còn lấy ở đó.",
 "Phiếu cũ có chặng ghé {old} nhận hàng, hiện đã bỏ chặng ấy.",
 "Đừng tới {old} lấy hàng; địa chỉ nhận trong lệnh trước là sai.",
)
PICKUP_CORRECTIONS_NONE=(
 "Lúc đầu dự kiến tới {old} lấy kiện, nhưng bên gửi đã hủy chặng đó.",
 "Điểm ghé nhận hàng {old} trong phiếu cũ không còn hiệu lực.",
 "Lệnh cũ yêu cầu ghé {old} nhận đồ; hiện nay lệnh đó đã bị rút lại.",
)
NEGATIVE_ROLE={
 "avoid":(
  "Đừng giao nhầm sang {loc}; địa chỉ này không phải đích mới.",
  "Tuyệt đối không chuyển kiện hàng tới {loc}.",
 ),
 "not_stop":(
  "Đi ngang {loc} thì không cần dừng để lấy hoặc giao hàng.",
  "Chặng này không có điểm dừng nhận hàng ở {loc}.",
 ),
 "closed":(
  "Hôm nay {loc} tạm ngừng nhận hàng nên không nằm trong lộ trình.",
  "Bộ phận ở {loc} báo đóng cửa; không ghé giao hay nhận kiện tại đó.",
 ),
 "historical":(
  "Hôm qua có một đơn khác chuyển tới {loc}, không liên quan đơn này.",
  "Chuyến trước robot từng đi {loc}; đây là thông tin lịch sử.",
 ),
 "left":(
  "Người nhận đã chuyển khỏi {loc}; địa chỉ đó chỉ là thông tin cũ.",
  "Người nhận không còn ở {loc}, đừng suy ra đó là nơi giao.",
 ),
 "not_related":(
  "Thông báo liên quan tới {loc} là của người khác, không phải chuyến này.",
  "Khu vực {loc} không liên quan đến việc giao nhận hiện tại.",
 ),
}
EXPLICIT_GOAL=(
 "Địa chỉ giao hiện có hiệu lực là: {goal}",
 "Lệnh giao cuối cùng cần làm là: {goal}",
 "Cập nhật đích đến đã được chấp nhận: {goal}",
 "Bỏ địa chỉ cũ; thực hiện chỉ dẫn giao mới: {goal}",
 "Nơi giao đúng của đơn hiện tại: {goal}",
)
PICKUP_SUPPLEMENT=(
 "Ngoài kiện {parcel}, cần ghé {via} nhận thêm {extra}, rồi mới tới nơi giao.",
 "Robot hãy nhận thêm {extra} ở {via} cho chuyến này; chặng tiếp theo là giao {parcel} tới đích.",
 "Chặng trung gian lấy {extra} nằm ở {via}. Sau đó robot mới thực hiện chặng giao {parcel}.",
 "Điểm gom thêm {extra} là {via}; khi nhận đủ thì đi giao {parcel}.",
 "Trước chặng giao {parcel}, robot cần nhận thêm {extra} tại {via}.",
 "Đến {via} lấy {extra} để đi chung chuyến, tiếp theo giao {parcel} tại đích đã chỉ định.",
)
URGENCY={
 True:(
  "Thời hạn mới yêu cầu giao ngay, không còn được trì hoãn.",
  "Ban đầu không gấp; đính chính: phải giao hỏa tốc.",
  "Đơn này cần ưu tiên xử lý khẩn cấp.",
  "Lệnh mới nâng mức ưu tiên lên giao gấp.",
 ),
 False:(
  "Không cần gấp, cứ theo lịch chuyển phát bình thường.",
  "Phiếu cũ ghi hỏa tốc, nhưng đã đính chính là không vội.",
  "Đơn này không được đánh dấu ưu tiên khẩn.",
  "Thời gian đã được nới; không cần chạy giao ngay.",
 ),
}
FRAGILITY={
 True:(
  "Trong kiện có vật dễ vỡ, cần tránh rung và va đập.",
  "Bản cũ bảo không sợ va chạm; đính chính: bên trong có đồ thủy tinh dễ bể.",
  "Lô hàng này nhạy va đập, phải nâng nhẹ khi di chuyển.",
  "Kiểm tra lại: có linh kiện dễ hỏng, cần chống va đập.",
 ),
 False:(
  "Bên trong chỉ có vật liệu bền, không có món dễ vỡ.",
  "Ban đầu đánh dấu dễ vỡ, nhưng kiểm tra lại là hàng không sợ va chạm.",
  "Không cần quy trình vận chuyển đồ mong manh.",
  "Kiện này không có hàng dễ vỡ, xử lý theo chế độ thường.",
 ),
}
NEUTRAL_END=("Nhờ robot xác nhận chuyến này.","Mình đã kiểm tra lại phiếu giao.","Chỉ thực hiện lệnh hiện tại thôi.","Cảm ơn đội vận chuyển.","Nhớ áp dụng thông tin mới nhất nhé.")
def distinct_type(used):
 choices=[t for t in PLACES if t not in used]
 if not choices:return None
 return r.choice(choices)
def place_name(kind):return r.choice(PLACES[kind])
def make_one(num):
 goal_mode=base.mode_choice(True);via_mode=base.mode_choice(False)
 goal,gph,used=base.target(goal_mode)
 via,vph=(None,None)
 if via_mode!="none":via,vph,used=base.target(via_mode,used)
 used=set(used)
 urgent=r.random()<.49
 fragile=r.random()<.48
 cargo=r.choice(NEUTRAL if r.random()<.72 else (base.CARGO_FRAGILE if fragile else base.CARGO_DURABLE))
 # Separate pickup item is an independently sampled supplementary package.
 supplemental=via is not None and r.random()<.53
 pickup_item=r.choice(SUPPLEMENT) if supplemental else cargo
 greeting=r.choice(base.GREET) if r.random()<.24 else None
 goal_text=r.choice(base.GOAL_ACTION).format(item=cargo,loc=gph)
 if via is not None:
  via_text=(r.choice(PICKUP_SUPPLEMENT).format(parcel=cargo,via=vph,extra=pickup_item)
            if supplemental else r.choice(base.VIA_ACTION).format(item=cargo,loc=vph))
 else:via_text=r.choice(base.NONE_ACTION).format(item=cargo)
 # Current true goal gets a wholly formed clause after any former destination was revoked.
 former_goal=None
 goal_revised=goal.type is not None and r.random()<.31
 blocks=[]
 if greeting:blocks.append(greeting)
 if goal_revised:
  other_kind=distinct_type(used)
  if other_kind:
   former_goal=place_name(other_kind);used.add(other_kind)
   blocks.append(r.choice(DEST_CORRECTIONS).format(old=former_goal,parcel=cargo))
   goal_text=r.choice(EXPLICIT_GOAL).format(goal=goal_text)
 old_pickup=None
 if goal.type is not None and r.random()<.18:
  other_kind=distinct_type(used)
  if other_kind:
   old_pickup=place_name(other_kind);used.add(other_kind)
   blocks.append(r.choice(PICKUP_CORRECTIONS_NONE if via is None else PICKUP_CORRECTIONS_TRUE).format(old=old_pickup))
 if r.random()<.50 and not goal_revised:
  blocks.extend((via_text,goal_text))
 else:blocks.extend((goal_text,via_text))
 negative_roles=[]
 # Old goal/pickup are already excluded; all decoys are distinct by TYPE.
 n_noise=r.choices((0,1,2),weights=(.36,.55,.09))[0]
 for _ in range(n_noise):
  kind=distinct_type(used)
  if kind is None:break
  used.add(kind)
  loc=place_name(kind)
  role=r.choice(tuple(NEGATIVE_ROLE))
  blocks.append(r.choice(NEGATIVE_ROLE[role]).format(loc=loc))
  negative_roles.append({"role":role,"type":kind,"lexeme":loc})
 if urgent or r.random()<.59:blocks.append(r.choice(URGENCY[urgent]))
 if fragile or r.random()<.59:blocks.append(r.choice(FRAGILITY[fragile]))
 if r.random()<.22:blocks.append(r.choice(NEUTRAL_END))
 text=" ".join(blocks)
 if r.random()<.025:
  # Only a non-semantic greeting typo; never corrupt an operative place, negation or flag.
  text=text.replace("Robot", "Roobt",1) if "Robot" in text else text
 accented=r.random()<.59
 if not accented:text=fold(text)
 row={"id":f"sem_v12_{num:06d}","text":text,
      "goal":base.ser(goal),"via":base.ser(via),"urgent":urgent,"fragile":fragile}
 # Strict check that semantic cue constructions survived.
 assert fold(gph) in fold(text), (num,"goal phrase missing")
 if vph:assert fold(vph) in fold(text),(num,"via phrase missing")
 if former_goal:assert fold(former_goal) in fold(text)
 if old_pickup:assert fold(old_pickup) in fold(text)
 assert len(spec_labels(goal,via,urgent,fragile))==8
 assert (goal.ref or "named") in GOAL_MODES
 assert via is None or (via.ref or "named") in VIA_MODES
 assert 14<=len(tokenize(text))<=MAX_TOKENS,(num,len(tokenize(text)))
 info={"id":row["id"],"goal_mode":goal_mode,"via_mode":via_mode,
       "former_destination":former_goal,"former_pickup":old_pickup,
       "negative_mentions":negative_roles,"parcel":cargo,"supplemental_pickup_item":pickup_item if supplemental else None,
       "accented":accented}
 return row,info
def read_hashes():
 seen=set()
 paths=[OLD/"incoming_40000.jsonl",OLD/"nlp_v4_curated_200000_quality_v2_reindexed.jsonl",
        HERE.parent/"semantic_generator_v8_20261009_first"/"specialized_new_40000_v11.jsonl"]
 for path in paths:
  with path.open(encoding="utf-8") as f:
   for line in f:
    j=json.loads(line);seen.add(fold(j["text"]).strip())
 return seen
def main():
 for p in (OUT,META,REPORT):
  if p.exists():raise RuntimeError("Never overwrite "+str(p))
 old=read_hashes();rows=[];provenance=[];seen=set();rejected=Counter()
 while len(rows)<N:
  try: row,info=make_one(len(rows)+1)
  except AssertionError as ex:
   rejected["semantic_or_length"]+=1
   if sum(rejected.values())>N*10:raise RuntimeError("Rejection limit") from ex
   continue
  key=fold(row["text"]).strip()
  if key in old or key in seen:
   rejected["duplicate"]+=1;continue
  seen.add(key);rows.append(row);provenance.append(info)
 with OUT.open("x",encoding="utf-8",newline="\n") as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
 with META.open("x",encoding="utf-8",newline="\n") as f:
  for inf in provenance:f.write(json.dumps(inf,ensure_ascii=False,separators=(",",":"))+"\n")
 sizes=sorted(len(tokenize(j["text"])) for j in rows);c=Counter()
 for row,meta in zip(rows,provenance):
  t=fold(row["text"]);c["total"]+=1
  c["via"]+=row["via"] is not None;c["former_goal"]+=meta["former_destination"] is not None
  c["former_goal_and_via"]+=meta["former_destination"] is not None and row["via"] is not None
  c["former_pickup"]+=meta["former_pickup"] is not None
  c["no_via_contains_lay"]+=row["via"] is None and bool(re.search(r"\blay\b",t))
  c["via_without_truoc"]+=row["via"] is not None and not bool(re.search(r"\btruoc\b",t))
  c["negative_role_count"]+=len(meta["negative_mentions"])
  c["supplemental_item"]+=meta["supplemental_pickup_item"] is not None
  c["accented"]+=any(ord(ch)>127 for ch in row["text"])
 stats={"n":N,"seed":SEED,"metrics":dict(c),"tokens":{"p50":sizes[N//2],"p90":sizes[int(N*.9)],"p99":sizes[int(N*.99)],"max":max(sizes)},
        "rejected":dict(rejected),"sha256":hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "status":"PILOT_PENDING_INDEPENDENT_SEMANTIC_AUDIT",
        "method":"Original phrase banks authored in new script; structural inspiration only from ZIP train schema, not sample text"}
 with REPORT.open("x",encoding="utf-8") as f:json.dump(stats,f,ensure_ascii=False,indent=2)
 print("OUTPUT",OUT,flush=True)
 print("STAT",json.dumps(stats,ensure_ascii=True),flush=True)
 for i in (0,5,120,345,701,1005,1201,1741):print("SAMPLE",json.dumps(rows[i],ensure_ascii=True),flush=True)
if __name__=="__main__":main()
