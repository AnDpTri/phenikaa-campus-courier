"""V8 typed-semantics Vietnamese courier corpus generator.
New files ONLY. No reuse or alteration of old 40k; old data used to detect overlap.
"""
from __future__ import annotations
import json,random,re,statistics,hashlib
from pathlib import Path
from collections import Counter
from courier.nlp.neural import spec_labels,MAX_TOKENS
from courier.nlp.parser import TargetSpec
from courier.nlp.text import tokenize,fold
from courier.nlp.synth import TYPES,GOAL_MODES,VIA_MODES

BASE=Path(r"D:\phenikaa\results\NLP_V4_R2\semantic_generator_v8_20261009_first")
OLD=Path(r"D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review")
N=2000
SEED=2026100918
OUTPUT=BASE/"pilot_v8_2000.jsonl"
META=BASE/"pilot_v8_stats.json"
rng=random.Random(SEED)
PLACE={
"library":("thư viện trung tâm","phòng đọc sách","kho giáo trình","khu học liệu","quầy mượn sách"),
"dorm":("ký túc xá sinh viên","khu nội trú","tòa ký túc","nhà ở sinh viên","khu lưu trú"),
"sports":("nhà thi đấu","sân bóng rổ","khu thể chất","sân tập thể thao","trung tâm thể thao"),
"clinic":("trạm y tế","phòng khám trong trường","phòng y tế trường","khu sơ cứu","trung tâm y tế"),
"canteen":("nhà ăn sinh viên","căn tin","quầy cơm","khu ẩm thực","nhà ăn trung tâm"),
"parking":("bãi gửi xe","nhà để xe","bãi đỗ xe","khu gửi xe sinh viên","bãi xe đạp"),
"lecture":("giảng đường chính","khu lớp học","tòa giảng đường","phòng học lý thuyết","tòa nhà học tập"),
"lab":("phòng thí nghiệm","xưởng thực hành","khu nghiên cứu","phòng lab","trung tâm thực nghiệm"),
"office":("văn phòng khoa","phòng đào tạo","tòa hiệu bộ","phòng hành chính","khu văn phòng"),
"gate":("cổng chính","cổng phụ","lối ra vào","chốt bảo vệ","trạm gác cổng"),
}
assert set(PLACE)==set(TYPES), (set(PLACE),set(TYPES))
CARGO_FRAGILE=("hộp ống nghiệm thủy tinh","khay mẫu vật dễ vỡ","bộ bình thủy tinh","hộp thiết bị quang học","thùng dụng cụ thủy tinh","bộ mô hình gốm","hộp lọ thuốc thử bằng kính","thùng linh kiện nhạy va đập","hộp kính quang học","bộ thiết bị phòng thí nghiệm dễ vỡ")
CARGO_DURABLE=("tập hồ sơ","túi tài liệu","hộp thẻ nhựa","bộ biểu mẫu","xấp giấy tờ","kiện sách","chồng giáo trình","lô bút viết","tập đề thi","túi áo đồng phục","hộp văn phòng phẩm","cuộn áp phích giấy")
GREET=("Nhờ robot xử lý đơn chuyển hàng sau.","Có một yêu cầu giao nhận mới.","Mình cần gửi một kiện hàng.","Đây là lệnh giao hàng đã xác nhận.","Robot nhận giúp yêu cầu này nhé.","Bên giao nhận có một chuyến mới.","Mình vừa cập nhật chuyến giao.","Robot kiểm tra lệnh vận chuyển này.")
GOAL_ACTION=(
"Giao {item} đến {loc}.",
"Chuyển {item} tới {loc}.",
"Đích giao {item} của chuyến này là {loc}.",
"Người nhận chờ {item} tại {loc}.",
"Điểm giao cuối cùng của {item} là {loc}.",
"Đưa {item} đến {loc} để bàn giao.",
"Chặng cuối của robot là chuyển {item} tới {loc}.",
"Nơi bàn giao {item} được xác định là {loc}.",
"Robot phải giao {item} tại {loc}.",
"Vận chuyển {item} tới {loc}.",
)
VIA_ACTION=(
"Kiện hàng chưa nằm trên xe; robot phải tới {loc} lấy {item} trước khi giao.",
"Điểm nhận {item} trước chặng giao là {loc}.",
"Trước khi giao, robot cần nhận {item} tại {loc}.",
"Robot hãy nhận {item} ở {loc}, rồi đi đến nơi bàn giao cuối cùng.",
"Lịch trình gồm chặng nhận {item} tại {loc}; chặng tiếp theo mới là giao hàng.",
"Robot cần đến {loc} nhận {item}. Sau đó mới thực hiện việc giao.",
"Bưu kiện {item} được giữ tại {loc}; hãy ghé lấy trước khi đi đến đích.",
"Nhận {item} ở {loc} là công việc đầu tiên, giao hàng là công việc sau đó.",
"Đầu tiên robot thu {item} tại {loc}; tiếp theo chuyển kiện tới đích.",
"Đơn này có điểm lấy hàng ở {loc}; cần hoàn thành việc lấy hàng rồi mới đi giao.",
"Chặng lấy {item} diễn ra tại {loc}, trước chặng giao cuối cùng.",
"Người gửi đang giữ {item} tại {loc}; robot cần nhận kiện ở đó trước khi giao.",
)
NONE_ACTION=(
"Kiện hàng đã được nhận tại nơi xuất phát, không cần ghé lấy thêm.",
"Robot mang sẵn {item} khi xuất phát; đi thẳng tới địa chỉ giao.",
"Không có điểm nhận hàng trung gian trong chuyến này.",
"Người gửi đã bàn giao {item} ngay tại nơi xuất phát; không cần nhận ở địa điểm khác.",
"Lấy {item} ngay tại nơi xuất phát rồi đi thẳng tới đích.",
"Hàng đã có sẵn trên xe; không cần dừng nhận hàng ở nơi khác.",
"Việc nhận {item} đã hoàn tất trước lúc khởi hành, chỉ cần đi giao.",
"Chuyến này không yêu cầu thêm chặng lấy hàng.",
)
CANCEL=(
"Địa chỉ {decoy} thuộc phiếu cũ và đã bị hủy; không giao ở đó.",
"Bản nháp nhắc tới {decoy}, nhưng đây không phải điểm giao hiện tại.",
"Lộ trình cũ từng ghé {decoy}; bỏ qua điểm đó trong chuyến này.",
"Thông tin trước đây ghi nhận hàng ở {decoy} đã bị hủy, không tới đó.",
"Robot không lấy hay giao kiện này ở {decoy}; đó là dữ liệu của đơn khác.",
"Địa điểm {decoy} là thông tin từ một đơn trước, không thuộc chuyến giao mới.",
"Người nhận từng làm việc ở {decoy}, nhưng địa chỉ đó không còn được sử dụng.",
"Đừng nhầm {decoy} là nơi nhận hàng hiện tại; đây chỉ là mốc cũ.",
"Đi ngang {decoy} không có nghĩa phải dừng hoặc lấy hàng tại đó.",
"Robot bỏ qua ghi chú về {decoy} trong bản nháp.",
)
URGENCY={
True:(
 "Đơn này cần giao gấp.",
 "Thời hạn vừa được rút ngắn; phải ưu tiên chuyển ngay.",
 "Bản cũ ghi không vội, nhưng lệnh mới yêu cầu giao khẩn.",
 "Đây là chuyến hỏa tốc, cần xử lý sớm.",
 "Có thay đổi về thời gian: yêu cầu hoàn tất ngay.",
),
False:(
 "Thời hạn bình thường, không cần giao gấp.",
 "Đơn hàng này không phải chuyến hỏa tốc.",
 "Bản cũ ghi giao ngay, nhưng lệnh mới xác nhận không gấp.",
 "Người gửi đồng ý thời gian linh hoạt.",
 "Không phải ưu tiên xử lý trước các chuyến khác.",
)}
FRAGILITY={
True:(
 "Kiện này chứa vật dễ vỡ; cần tránh va đập.",
 "Các đồ vật trong thùng dễ hỏng khi rung lắc, phải vận chuyển nhẹ.",
 "Bản cũ ghi hàng bền; kiểm tra lại xác nhận kiện hàng rất dễ vỡ.",
 "Hàng nhạy va chạm, không được xếp đè.",
 "Lưu ý có vật liệu dễ bể trong kiện.",
),
False:(
 "Kiện này không chứa đồ dễ vỡ.",
 "Hàng là vật liệu bền, không cần chế độ chống va đập đặc biệt.",
 "Bản nháp ghi dễ vỡ, nhưng đã xác nhận không cần xử lý như hàng mong manh.",
 "Có thể vận chuyển theo chế độ hàng thông thường.",
 "Kiện này không yêu cầu bảo vệ riêng đối với vật dễ vỡ.",
)}
NEUTRAL=("Phiếu giao đã được cập nhật.","Người nhận sẽ đối chiếu mã đơn.","Thông tin đơn này đã được xác nhận.","Robot kiểm tra số lượng hàng khi giao.","Người gửi đã xác nhận lộ trình.","Cảm ơn đội chuyển phát.")
def choice(t):return rng.choice(t)
def alias(kind):return choice(PLACE[kind])
def mode_choice(goal=True):
 if goal:return rng.choices(["named","near","far","anchor_near","north","south","east","west","north_most","south_most","east_most","west_most"],weights=[34,14,9,12,4,4,4,4,3,3,3,3])[0]
 return rng.choices(["none","named","near","far","north","south","east","west"],weights=[52,32,5,3,2,2,2,2])[0]
def target(mode, excluded=()):
 used=set(excluded)
 if mode.endswith("_most"):
  dirs={"north_most":"bắc","south_most":"nam","west_most":"tây","east_most":"đông"}
  return TargetSpec(None,mode,None),choice((
    f"địa danh nằm xa nhất về phía {dirs[mode]} của toàn khuôn viên",
    f"địa điểm ở cực {dirs[mode]} trên toàn bản đồ",
    f"địa danh ngoài cùng về phía {dirs[mode]} trên sơ đồ"
  )),used
 if mode=="anchor_near":
  a=choice([k for k in PLACE if k not in used]);used.add(a)
  landmark=alias(a)
  phrases=(
    f"địa danh gần {landmark} nhất tính theo số ô lưới, không tính chính {landmark}",
    f"địa điểm có khoảng cách ô lưới ngắn nhất tới {landmark}, ngoại trừ chính {landmark}",
    f"địa danh khác {landmark} gần mốc {landmark} nhất theo khoảng cách ô lưới",
  )
  return TargetSpec(None,mode,a),choice(phrases),used
 kind=choice([k for k in PLACE if k not in used]);used.add(kind)
 name=alias(kind)
 if mode=="named":return TargetSpec(kind,None,None),name,used
 if mode in ("north","south","east","west"):
  axis={"north":"bắc","south":"nam","east":"đông","west":"tây"}[mode]
  return TargetSpec(kind,mode,None),choice((
    f"{name} ở phía {axis} nhất trong các địa điểm cùng loại trên bản đồ",
    f"{name} nằm xa nhất về phía {axis} so với các địa điểm cùng loại",
    f"{name} thuộc vị trí ngoài cùng phía {axis} trong số các địa điểm cùng loại",
  )),used
 if mode in ("near","far"):
  a=choice([k for k in PLACE if k not in used]);used.add(a)
  anchor=alias(a)
  words=("gần","ngắn nhất") if mode=="near" else ("xa","dài nhất")
  phrases=(
     f"{name} {words[0]} {anchor} nhất trong các địa điểm cùng loại, tính theo đường chim bay",
     f"{name} có khoảng cách đường chim bay {words[1]} tới {anchor} trong số các địa điểm cùng loại",
     f"{name} {words[0]} {anchor} hơn các địa điểm cùng loại còn lại",
  )
  return TargetSpec(kind,mode,a),choice(phrases),used
 raise ValueError(mode)
def ser(s):return None if s is None else {"type":s.type,"ref":s.ref,"anchor":s.anchor}
def old_set():
 out=set()
 for file in (OLD/"incoming_40000.jsonl",OLD/"nlp_v4_curated_200000_quality_v2_reindexed.jsonl"):
  with file.open(encoding="utf-8") as f:
   for line in f:out.add(fold(json.loads(line)["text"]).strip())
 return out
def make_one(i):
 goal_mode=mode_choice(True);via_mode=mode_choice(False)
 goal,gph,used=target(goal_mode)
 if via_mode=="none":via,vph=None,None
 else: via,vph,used=target(via_mode,used)
 urgent=rng.random()<.47;fragile=rng.random()<.44
 cargo=choice(CARGO_FRAGILE if fragile else CARGO_DURABLE)
 # A single physical item is the *same* at pickup and dropoff.
 goal_text=choice(GOAL_ACTION).format(item=cargo,loc=gph)
 via_text=choice(VIA_ACTION if via is not None else NONE_ACTION).format(item=cargo,loc=vph)
 blocks=[goal_text,via_text]
 rng.shuffle(blocks)
 # Unambiguous disavowed places. Do not mention the true source/target aliases.
 unused=[k for k in PLACE if k not in used]
 ct=rng.choices([0,1,2],weights=[.30,.55,.15])[0]
 decoys=[]
 for _ in range(ct):
  if not unused:break
  kind=choice(unused);unused.remove(kind);decoy=alias(kind);decoys.append(decoy)
  blocks.append(choice(CANCEL).format(decoy=decoy))
 # Correction and negation labels must have visible evidence.
 if rng.random()<.86 or urgent:
  blocks.append(choice(URGENCY[urgent]))
 if rng.random()<.84 or fragile:
  blocks.append(choice(FRAGILITY[fragile]))
 if rng.random()<.20:blocks.append(choice(NEUTRAL))
 if rng.random()<.53:blocks.insert(0,choice(GREET))
 # Lexical accent distribution similar to original 40k, no half-accented mixing.
 accented=rng.random()<.59
 text=" ".join(blocks)
 if not accented:text=fold(text)
 row={"id":f"sem_v8_{i:06d}","text":text,"goal":ser(goal),"via":ser(via),"urgent":urgent,"fragile":fragile}
 assert 14 <= len(tokenize(text)) <= MAX_TOKENS,(i,len(tokenize(text)))
 for ph in (gph,vph):
  if ph:assert fold(ph) in fold(text),(i,ph)
 assert len(spec_labels(goal,via,urgent,fragile))==8
 assert (goal.ref or "named") in GOAL_MODES and (via is None or (via.ref or "named") in VIA_MODES)
 return row
def main():
 for f in (OUTPUT,META):
  if f.exists():raise RuntimeError("Refusing overwrite "+str(f))
 old=old_set();seen=set();rows=[];attempts=0;c=Counter()
 while len(rows)<N:
  attempts+=1
  if attempts>15*N:raise RuntimeError("Too many rejections")
  x=make_one(len(rows)+1);txt=fold(x["text"]).strip()
  if txt in old or txt in seen:c["duplicates_rejected"]+=1;continue
  seen.add(txt);rows.append(x)
 lengths=sorted(len(tokenize(x["text"])) for x in rows)
 for r in rows:
  v=r["via"] is not None;t=fold(r["text"]);c["n"]+=1;c["via_present"]+=v
  c["via_without_truoc"]+=v and not re.search(r"\btruoc\b",t)
  c["none_with_lay"]+=(not v) and bool(re.search(r"\blay\b",t))
  c["accents"]+=any(ord(ch)>127 for ch in r["text"])
 with OUTPUT.open("x",encoding="utf-8",newline="\n") as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
 stat={"N":N,"seed":SEED,"counts":dict(c),"tokens":{"p50":lengths[N//2],"p90":lengths[int(N*.9)],"p99":lengths[int(N*.99)],"max":lengths[-1]},"sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),"status":"PILOT_UNAUDITED_SEMANTICS"}
 with META.open("x",encoding="utf-8") as f:json.dump(stat,f,ensure_ascii=False,indent=2)
 print("CREATED",OUTPUT,flush=True)
 print("METRICS",json.dumps(stat,ensure_ascii=True),flush=True)
 for i in (0,50,200,600,800,1250,1770):
  print("SAMPLE",json.dumps(rows[i],ensure_ascii=True),flush=True)
if __name__=="__main__":main()
