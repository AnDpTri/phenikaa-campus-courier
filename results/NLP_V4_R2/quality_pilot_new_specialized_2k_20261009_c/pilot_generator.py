"""Pilot generator of NEW compositional, validated specialized Vietnamese courier missions.
Original 40k is a reference distribution ONLY; NEVER a source of copied output.
No existing artifacts are modified. Run under py -B to avoid pycache.
"""
import json, re, random, unicodedata, hashlib, statistics
from collections import Counter
from pathlib import Path
from courier.nlp.neural import spec_labels, MAX_TOKENS
from courier.nlp.synth import GOAL_MODES, VIA_MODES, TYPES
from courier.nlp.parser import TargetSpec
from courier.nlp.text import fold, tokenize

OUT=Path(r"D:\phenikaa\results\NLP_V4_R2\quality_pilot_new_specialized_2k_20261009_c")
ORIG=Path(r"D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review\incoming_40000.jsonl")
CURATED=Path(r"D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review\nlp_v4_curated_200000_quality_v2_reindexed.jsonl")
OUTPUT=OUT/"pilot_new_specialized_2000.jsonl"
REPORT=OUT/"pilot_quality_report.json"
N=2000
SEED=202610091
R=random.Random(SEED)
TYPES_LIST=list(TYPES)
PLACES={
 "library":("thư viện trung tâm","kho giáo trình","phòng đọc tầng một","quầy mượn sách","khu học liệu","thư viện nhà trường"),
 "dorm":("ký túc xá sinh viên","khu lưu trú nội trú","dãy nhà ở sinh viên","tòa ký túc","nhà ở tập thể","khu nội trú"),
 "sports":("sân tập thể thao","nhà thi đấu đa năng","khu luyện tập","sân bóng rổ","trung tâm thể chất","sân bóng đá"),
 "clinic":("phòng y tế trường","trạm chăm sóc sức khỏe","phòng khám","khu sơ cứu","trung tâm y tế","trạm y tế"),
 "canteen":("nhà ăn sinh viên","quầy cơm","khu ẩm thực","căn tin trường","bếp ăn tập thể","nhà ăn trung tâm"),
 "parking":("bãi gửi xe","khu đỗ xe máy","nhà để xe","bãi xe sinh viên","khu gửi xe đạp","bãi đỗ phương tiện"),
 "lecture":("giảng đường chính","tòa phòng học","khu lớp học","phòng học lý thuyết","nhà giảng dạy","giảng đường lớn"),
 "lab":("phòng thí nghiệm","xưởng thực hành","khu nghiên cứu","phòng lab hóa sinh","trung tâm thực nghiệm","xưởng kỹ thuật"),
 "office":("phòng hành chính","tòa hiệu bộ","văn phòng khoa","phòng đào tạo","bộ phận quản lý sinh viên","văn phòng nhà trường"),
 "gate":("cổng chính","chốt bảo vệ","lối ra vào","cổng phụ","trạm gác cổng","cổng phía sân trường"),
}
ITEMS=("hồ sơ xét duyệt","bưu kiện","hộp thiết bị","tài liệu giảng dạy","túi đồ cá nhân","kiện hàng","bộ tài liệu","thùng linh kiện","hộp thuốc","bưu phẩm","thẻ sinh viên","tập biểu mẫu","hộp vật tư","tập giấy tờ","phong bì","bộ dụng cụ","hộp thực phẩm","túi mẫu thí nghiệm","kiện sách","chồng tài liệu")
GREET=("Robot nhận lệnh giúp mình.","Có một việc giao hàng mới.","Mình cần nhờ robot hỗ trợ.","Xin chào bộ phận chuyển phát.","Thông tin đơn hàng mới đây.","Alo đội giao nhận.","Mình gửi yêu cầu này nhé.","Nhờ xử lý chuyến giao sau.","Robot đọc lại yêu cầu giúp.","Tôi có một chuyến chuyển hàng.","Bạn ơi, có đơn phát sinh.","Lệnh giao hàng vừa cập nhật.","Bên mình có kiện cần giao.","Robot giúp tôi một chuyến.","Mình gửi đơn hàng mới.")
CLOSE=("Nhờ kiểm tra lại giúp.","Cảm ơn robot nhé.","Xử lý đúng địa điểm giúp mình.","Vậy là đủ thông tin.","Nhờ làm đúng chỉ dẫn.","Đừng nhầm thông tin cũ nữa nhé.","Mình cảm ơn.","Nhớ báo khi xong.","Bạn xác nhận giúp nhé.","Cảm ơn nhiều.")
DIRECTION={
 "north":("phía bắc","phía trên","về hướng bắc"),
 "south":("phía nam","phía dưới","về hướng nam"),
 "west":("phía tây","bên trái","về hướng tây"),
 "east":("phía đông","bên phải","về hướng đông"),
}
MOST={
 "north_most":("điểm ở mép bắc xa nhất của sơ đồ","vị trí cực bắc trên bản đồ","ô nằm cao nhất toàn khuôn viên"),
 "south_most":("điểm sát rìa nam nhất của bản đồ","vị trí cực nam của sơ đồ","ô thấp nhất trên toàn bản đồ"),
 "west_most":("điểm tận cùng phía tây của sơ đồ","vị trí ngoài cùng bên trái bản đồ","ô nằm xa nhất về phía tây"),
 "east_most":("điểm tận cùng phía đông của sơ đồ","vị trí ngoài cùng bên phải bản đồ","ô nằm xa nhất về phía đông"),
}
GOAL_MODES_W=list(GOAL_MODES)
GOAL_WEIGHTS={"named":34,"near":14,"far":9,"anchor_near":13,"north":4,"south":4,"west":4,"east":4,"north_most":3.5,"south_most":3.5,"west_most":3.5,"east_most":3.5}
VIA_MODES_W=list(VIA_MODES)
VIA_WEIGHTS={"none":52,"named":32,"near":5,"far":2,"north":2.25,"south":2.25,"west":2.25,"east":2.25}
# Fully distinct syntactic constructions; no direct held-out "lay ... o ... xong thi" phrase.
VIA_FIRST=[
 "Chặng nhận hàng phụ: robot đến {via}, nhận {second}; hoàn thành chặng đó rồi {goal}",
 "Ở {via} có {second} cần gom lên xe. Sau công đoạn này, {goal}",
 "Robot lấy thêm {second} tại {via}; phần giao sau cùng là: {goal}",
 "Hãy xử lý điểm lấy {second} tại {via} rồi hãy tính đến đích nhận hàng: {goal}",
 "Có hai việc theo thứ tự: nhận {second} ở {via}; kế tiếp, {goal}",
 "Việc đầu tiên là tìm {via} để nhận {second}. Kế tiếp mới thực hiện việc giao: {goal}",
 "Lô {second} còn ở {via}. Ghé nhận lô ấy, rồi hoàn thành yêu cầu: {goal}",
 "Tại {via} đang giữ {second} của đơn này. Lấy đủ hàng tại đó; đích cuối là {goal}",
 "Lịch trình bắt đầu bằng điểm lấy {second} tại {via}, rồi kết thúc bằng việc {goal}",
 "Đón kiện phụ ở {via} để nhập cùng đơn đang mang; đến chặng sau thì {goal}",
 "Mình cần robot vào {via} gom {second}; điểm bàn giao cuối cùng: {goal}",
 "Tách ra hai chặng: điểm lấy phụ là {via}. Khi đã nhận {second}, {goal}",
 "Bên {via} đang chờ robot tới nhận {second}. Sau đó cần {goal}",
 "Trước hết, nhận {second} tại {via}; phần việc hoàn tất là {goal}",
 "Robot di chuyển tới {via} lấy thêm {second}, kế đó mới {goal}",
 "Hàng phụ {second} được lấy ở {via}; việc giao hàng là {goal}",
]
VIA_LAST=[
 "{goal} Nhưng trước khi đến đích, phải đến {via} nhận {second}.",
 "{goal} Nhớ rằng hàng chưa đủ: {second} đang ở {via}, cần ghé lấy rồi mới giao.",
 "{goal} Trong lộ trình có điểm trung gian {via} để nhận {second}; không đi thẳng.",
 "{goal} Chặng lấy {second} đặt tại {via}; hoàn tất chặng lấy mới đi giao.",
 "{goal} Phải ghé {via} thu {second} trên đường đi; đích cuối vẫn như trên.",
 "{goal} Đơn này gồm cả việc nhận {second} từ {via}, sau đó mới giao tới nơi.",
 "{goal} Xin đừng bỏ sót điểm nhận {second} ở {via}. Đây là điểm dừng trước đích.",
 "{goal} Điểm phụ để lấy {second} là {via}, không phải địa chỉ nhận cuối cùng.",
 "{goal} Khi đi, robot vòng đến {via} gom {second}, sau đó mới đến nơi giao.",
 "{goal} Lộ trình bắt buộc đi qua {via} để nhặt {second} rồi mới hoàn tất giao nhận.",
 "{goal} Một khâu nữa cần làm: lấy {second} tại {via}, rồi chuyển sang khâu giao.",
 "{goal} Trong hai công đoạn, chặng ghé {via} lấy {second} diễn ra trước chặng giao.",
]
GOAL_FRAMES=[
 "giao {item} đến {goal}.",
 "mang {item} tới {goal}.",
 "đưa {item} tới người nhận ở {goal}.",
 "nơi nhận {item} cuối cùng là {goal}.",
 "điểm kết thúc đơn hàng đặt tại {goal}.",
 "phát {item} ở {goal}.",
 "đơn này giao tại {goal}.",
 "chuyển {item} sang {goal}.",
 "địa chỉ nhận cuối cùng là {goal}.",
 "giao kiện {item} cho người trực ở {goal}.",
 "đích chuyển phát {item}: {goal}.",
 "người nhận đợi kiện {item} tại {goal}.",
 "đưa {item} về địa điểm {goal}.",
]
NONE_FRAMES=[
 "{goal} Không có điểm dừng nhận hàng trung gian nào.",
 "{goal} Hàng đã đủ từ đầu, robot đi thẳng đến nơi nhận.",
 "{goal} Đơn này chỉ có một điểm giao; không bố trí điểm ghé để lấy thêm.",
 "Đã lấy đủ {item} tại nơi xuất phát, {goal} Không phải lấy ở một địa điểm khác.",
 "{goal} Có người đã lấy {item} từ kho xuất phát rồi, robot chỉ cần đi giao.",
 "{goal} Đừng tạo thêm chặng nhận hàng ngoài điểm xuất phát.",
 "Robot nhận {item} tại nơi xuất phát. Bây giờ {goal} Không có chặng lấy thêm.",
 "{goal} Lấy hàng từ người gửi ngay tại chỗ xuất phát; không cần ghé thêm.",
 "{goal} Chỉ thực hiện hành trình đến đích; không ghé điểm gom hàng nào.",
 "{goal} Nhận sẵn {item} khi khởi hành và giao trực tiếp tới đích.",
]
DISTRACTOR=[
 "Hôm qua đã chuyển đồ qua {other}; chuyện đó thuộc đơn trước.",
 "Địa chỉ cũ ghi {other} nhưng đã bị hủy, bỏ qua địa điểm ấy.",
 "Không cần ghé {other} trong chuyến này.",
 "Đừng giao nhầm tới {other}, đây là nơi không liên quan.",
 "Người nhận trước đây làm tại {other}, nhưng đã chuyển đi.",
 "Trong bản nháp có nhắc đến {other}; dữ kiện này không còn hiệu lực.",
 "Đi ngang {other} cũng được nhưng không dừng nhận hàng ở đó.",
 "Có thông báo {other} đang đóng cửa, đừng đổi đích sang nơi ấy.",
 "Địa điểm {other} chỉ dùng làm mốc tham khảo, không phải điểm đến.",
 "Chuyến cũ từng ghé {other}; lần này không cần ghé địa chỉ cũ.",
 "Một người nhắn {other}, nhưng đó là thông tin của đơn khác.",
 "Hệ thống từng ghi sai đích là {other}; địa chỉ chính thức nằm trong lệnh này.",
 "Bạn nhớ loại {other} khỏi danh sách điểm nhận hàng nhé.",
 "Không giao và cũng không gom thêm hàng ở {other}.",
]
URGENT_TRUE=[
 "Tình trạng thời hạn: cần giao gấp.",
 "Ban đầu được báo có thể chờ, nay đổi thành ưu tiên giao ngay.",
 "Đây là đơn khẩn, xử lý ngay khi nhận.",
 "Lệnh mới thay lệnh cũ: không được chậm trễ.",
 "Phần thời gian được nâng lên mức hỏa tốc.",
 "Chuyến này phải xử lý sớm nhất có thể.",
 "Có thay đổi: từ không vội thành cần hoàn thành ngay.",
 "Mốc thời hạn rất gần, ưu tiên đơn này.",
]
URGENT_FALSE=[
 "Thời gian linh hoạt, không phải chạy gấp.",
 "Ban đầu tưởng khẩn nhưng đã xác nhận có thể làm từ từ.",
 "Đơn này không có hạn chót gấp.",
 "Không cần ưu tiên hỏa tốc.",
 "Lịch giao được nới, có thể xử lý cuối ngày.",
 "Đừng hiểu nhầm đơn này là việc khẩn.",
 "Bản cũ ghi giao ngay, bản chính thức không cần vội.",
 "Có thể vận chuyển theo nhịp bình thường.",
]
FRAGILE_TRUE=[
 "Trong kiện có vật dễ vỡ, cần chống va đập.",
 "Lúc đầu ghi hàng bền, nhưng đã đính chính là đồ thủy tinh.",
 "Chú ý bảo vệ linh kiện rất dễ hỏng khi rung.",
 "Đồ gốm bên trong có thể vỡ nếu rơi.",
 "Yêu cầu cầm nhẹ vì hàng rất mong manh.",
 "Cập nhật chất lượng hàng: cần tránh rung lắc.",
 "Đây là đồ dễ bể, không được xếp chồng mạnh.",
 "Vật chứa kính mỏng nên phải vận chuyển cẩn thận.",
]
FRAGILE_FALSE=[
 "Kiện hàng chắc chắn, không thuộc loại dễ vỡ.",
 "Lúc đầu tưởng đồ sứ, thực tế là giấy tờ thông thường.",
 "Hàng được xác nhận không cần chống va đập đặc biệt.",
 "Đã đính chính: không có vật liệu dễ bể.",
 "Đồ mềm, rơi nhẹ cũng không hỏng.",
 "Không cần chế độ vận chuyển đồ mong manh.",
 "Trạng thái dễ vỡ trong phiếu cũ là nhầm lẫn.",
 "Bên trong chỉ có tài liệu bền, không sợ va chạm.",
]
NEUTRAL=[
 "Người gửi sẽ đối chiếu mã đơn khi robot hoàn tất.",
 "Phiếu giao có kèm chữ ký của bên nhận.",
 "Chỉ xử lý các yêu cầu trong đơn này.",
 "Thông tin trên thùng đã được xác nhận.",
 "Đơn hàng đã được đóng gói.",
 "Xin bỏ qua những ghi chú không liên quan tới lộ trình.",
 "Mọi người đã thống nhất lại nội dung giao nhận.",
 "Hệ thống vừa xác nhận danh sách kiện.",
 "Robot giữ nguyên số lượng kiện theo phiếu giao.",
 "Phiếu mới có hiệu lực kể từ lúc này.",
]
def choose_type(excluded=()):
 pool=[t for t in TYPES_LIST if t not in excluded]
 assert pool,excluded
 return R.choice(pool)
def alias(t):return R.choice(PLACES[t])
def weighted(opts,weights):return R.choices(list(opts),[weights[x] for x in opts])[0]
def target(mode,avoids):
 used=set(avoids)
 if mode in MOST:
  p=R.choice(MOST[mode])
  return TargetSpec(None,mode,None),p,used
 if mode=="anchor_near":
  a=choose_type(used);used.add(a)
  p=R.choice([
   f"vị trí gần {alias(a)} nhất trên sơ đồ",
   f"điểm bất kỳ nằm sát {alias(a)} nhất, không tính chính địa điểm mốc",
   f"địa điểm có khoảng cách ngắn nhất tới {alias(a)} trên bản đồ",
   f"điểm đứng gần {alias(a)} hơn mọi điểm khác trong khuôn viên",
  ])
  return TargetSpec(None,mode,a),p,used
 kind=choose_type(used);used.add(kind);place=alias(kind)
 if mode=="named":return TargetSpec(kind,None,None),place,used
 if mode in DIRECTION:
  return TargetSpec(kind,mode,None),f"{place} nằm ở {R.choice(DIRECTION[mode])} trên sơ đồ",used
 if mode in ("near","far"):
  a=choose_type(used);used.add(a);anchor=alias(a)
  words={
  "near": [
    f"{place} nằm gần {anchor} hơn địa điểm cùng loại còn lại",
    f"trong hai {place}, chọn nơi có khoảng cách tới {anchor} ngắn hơn",
    f"{place} ở gần {anchor} nhất trong các nơi cùng loại",
    f"{place} phía gần {anchor} hơn chỗ thứ hai",
  ],
  "far": [
    f"{place} nằm xa {anchor} hơn địa điểm cùng loại còn lại",
    f"trong hai {place}, chọn nơi ở xa {anchor} hơn",
    f"{place} xa {anchor} nhất trong các nơi cùng loại",
    f"{place} phía xa {anchor} hơn chỗ thứ hai",
  ]}
  return TargetSpec(kind,mode,a),R.choice(words[mode]),used
 raise ValueError(mode)
def serialize(t):
 return None if t is None else {"type":t.type,"ref":t.ref,"anchor":t.anchor}
def validate(row, evidence):
 goal=TargetSpec(**row["goal"])
 via=None if row["via"] is None else TargetSpec(**row["via"])
 assert (goal.ref or "named") in GOAL_MODES
 assert via is None or (via.ref or "named") in VIA_MODES
 assert len(spec_labels(goal,via,row["urgent"],row["fragile"]))==8
 assert evidence["goal_phrase"].casefold() in row["text"].casefold() if evidence["accented"] else fold(evidence["goal_phrase"]) in row["text"].casefold()
 if evidence["via_phrase"]:
  assert (evidence["via_phrase"].casefold() in row["text"].casefold()) if evidence["accented"] else (fold(evidence["via_phrase"]) in row["text"].casefold())
 assert len(tokenize(row["text"]))<=MAX_TOKENS
 assert len(tokenize(row["text"]))>=14
def generate_one(i):
 gm=weighted(GOAL_MODES_W,GOAL_WEIGHTS);vm=weighted(VIA_MODES_W,VIA_WEIGHTS)
 gs,gphrase,gu=target(gm,())
 if vm!="none":vs,vphrase,vu=target(vm,gu)
 else:vs,vphrase,vu=None,None,set()
 item=R.choice(ITEMS);extra_item=R.choice(ITEMS)
 goal_sentence=R.choice(GOAL_FRAMES).format(item=item,goal=gphrase)
 if vs is None:core=R.choice(NONE_FRAMES).format(goal=goal_sentence,item=item)
 else:
  templates=VIA_FIRST if R.random()<0.58 else VIA_LAST
  core=R.choice(templates).format(goal=goal_sentence,via=vphrase,second=extra_item)
 urgent=R.random()<.46;fragile=R.random()<.43
 clauses=[R.choice(GREET),core]
 used=gu|vu
 n_dist=R.choices([0,1,2,3],[.12,.27,.40,.21])[0]
 for _ in range(n_dist):
  other=alias(choose_type(used))
  clauses.append(R.choice(DISTRACTOR).format(other=other))
 if R.random()<.94:clauses.append(R.choice(URGENT_TRUE if urgent else URGENT_FALSE))
 if R.random()<.90:clauses.append(R.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE))
 if R.random()<.60:clauses.append(R.choice(NEUTRAL))
 # Shuffle contextual clauses but preserve the syntactic core's internal action ordering.
 if len(clauses)>3:
  extras=clauses[2:];R.shuffle(extras);clauses=clauses[:2]+extras
 if R.random()<.38:
  clauses=clauses[2:3]+clauses[:2]+clauses[3:] if len(clauses)>=3 else clauses
 clauses.append(R.choice(CLOSE))
 raw=" ".join(x.strip() for x in clauses if x.strip())
 accented=R.random()<.60
 text=raw if accented else fold(raw)
 row={"id":f"specnew_{i:06d}","text":text,"goal":serialize(gs),"via":serialize(vs),"urgent":urgent,"fragile":fragile}
 validate(row,{"goal_phrase":gphrase,"via_phrase":vphrase,"accented":accented})
 info={"goal_mode":gm,"via_mode":vm,"via_phrase":vphrase,"accented":accented,"goal_phrase":gphrase}
 return row,info

def scan_old():
 unique=set();base_text=[]
 with ORIG.open(encoding="utf-8") as f:
  for line in f:
   x=json.loads(line);unique.add(fold(x["text"]).strip())
 return unique
def main():
 for p in (OUTPUT,REPORT):
  if p.exists():raise RuntimeError("Refuse overwrite: "+str(p))
 old=scan_old()
 rows=[];stats=Counter();seen=set();samples=[];tries=0
 while len(rows)<N:
  tries+=1
  if tries>N*6:raise RuntimeError("Too many rejections")
  row,info=generate_one(len(rows)+1)
  canon=fold(row["text"]).strip()
  if canon in old or canon in seen:stats["duplicate_rejected"]+=1;continue
  seen.add(canon)
  rows.append(row)
  if len(samples)<30 and (len(rows)%67==1):samples.append({"row":row,"syntactic":info})
  stats["n"]+=1
  stats["via_"+info["via_mode"]]+=1
  stats["goal_"+info["goal_mode"]]+=1
  stats["accents"]+=info["accented"]
  stats["via_has_truoc"]+=(row["via"] is not None and bool(re.search(r"\btruoc\b",fold(row["text"]))))
  stats["via_without_truoc"]+=(row["via"] is not None and not re.search(r"\btruoc\b",fold(row["text"])))
  stats["none_has_lay"]+=(row["via"] is None and bool(re.search(r"\blay\b",fold(row["text"]))))
  stats["none_total"]+=(row["via"] is None)
  stats["via_total"]+=(row["via"] is not None)
  stats["token_count_sum"]+=len(tokenize(row["text"]))
 with OUTPUT.open("x",encoding="utf-8",newline="\n") as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
 sizes=sorted(len(tokenize(x["text"])) for x in rows)
 report={
  "status":"PILOT_UNAUDITED_HUMAN_SEMANTICS",
  "description":"2k genuinely new compositional specialized missions, generated without old40k templates or training/validation/test sentences",
  "generator_seed":SEED,"count":N,
  "source_reference_only":str(ORIG),
  "stats":dict(stats),
  "token_percentiles":{"p10":sizes[int(N*.1)],"p50":sizes[int(N*.5)],"p90":sizes[int(N*.9)],"p99":sizes[int(N*.99)],"max":max(sizes)},
  "samples":samples,
  "source_modification":False,
  "train_completed":False,
  "limitations":["Not yet semantic-human audited","Syntax/templates are newly authored, not proof of original-quality parity","Validation/test data not used"],
  "sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
 }
 with REPORT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print(json.dumps({"file":str(OUTPUT),"stats":dict(stats),"tokens":report["token_percentiles"],"sha256":report["sha256"]},ensure_ascii=True),flush=True)
 for j in samples[:12]:print("SAMPLE",json.dumps(j,ensure_ascii=True),flush=True)
if __name__=="__main__":main()
