"""Candidate V2 contrastive dataset. Experimental only; NEVER attach to training."""
from __future__ import annotations
from collections import Counter
import datetime,hashlib,json,random
from pathlib import Path
from courier.nlp.parser import TargetSpec
from courier.nlp.neural import MAX_TOKENS,spec_labels
from courier.nlp.text import fold,tokenize
WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
ALIASES={
"library":("thư viện","phòng đọc sách","khu học liệu"),
"dorm":("ký túc xá","khu nội trú","nhà ở sinh viên"),
"sports":("nhà thi đấu","sân bóng rổ","khu thể chất"),
"clinic":("trạm y tế","phòng khám","phòng y tế"),
"canteen":("nhà ăn","căn tin","quầy cơm"),
"parking":("bãi gửi xe","nhà để xe","khu gửi xe"),
"lecture":("giảng đường","khu lớp học","tòa nhà học tập"),
"lab":("phòng thí nghiệm","phòng lab","xưởng thực hành"),
"office":("văn phòng khoa","phòng đào tạo","tòa hiệu bộ"),
"gate":("cổng chính","lối ra vào","cổng phụ")}
GOAL=[
"Giao kiện hàng đến {goal}.","Nơi bàn giao cuối là {goal}.",
"Robot hãy chuyển thùng tài liệu tới {goal}.",
"Đích giao chính thức là {goal}.","Đem bưu phẩm sang {goal} cho người nhận."
]
POS=[
"Chặng lấy còn hiệu lực là {via}; cần đến đó nhận thêm hàng rồi giao.",
"Robot phải ghé {via} lấy chứng từ trước khi đi giao.",
"Điểm nhận bổ sung là {via}, phải lấy phong bì tại đó.",
"Không bỏ qua việc nhận hồ sơ ở {via}. Đây là chặng lấy đang có hiệu lực.",
"Yêu cầu nhận kiện tại {via} vẫn hợp lệ; thực hiện xong mới giao.",
]
NEG=[
"Phiếu cũ ghi chặng lấy tại {via} nhưng đã hủy. Không nhận thêm hàng ở đó.",
"Robot từng được yêu cầu ghé {via} lấy chứng từ, nay lệnh đã bị hủy. Đi thẳng để giao.",
"Điểm nhận bổ sung là {via} theo bản nháp, không còn hiệu lực; không lấy phong bì tại đó.",
"Thông báo 'không bỏ qua việc nhận hồ sơ ở {via}' là nội dung cũ đã hủy. Chuyến này không có chặng lấy.",
"Yêu cầu nhận kiện tại {via} đã hết hiệu lực. Không thực hiện ghé lấy.",
]
# Include synonyms unseen in OLD generator and minimally contrasted negatives.
FRAGILE_TRUE=[
"Kiện chứa kính dễ vỡ; phải chống rung lắc.",
"Đồ trong hộp nhạy va chạm, xin mang thật nhẹ.",
"Lô hàng có đồ rất dễ nứt khi va đập.",
"Vật phẩm mong manh, cần kê đệm và tránh sóc.",
"Có đồ dễ bể, tuyệt đối không va mạnh.",
"Đây là hàng dễ hỏng khi rung lắc, cần giữ chắc.",
]
FRAGILE_FALSE=[
"Kiện đã kiểm tra, không có đồ dễ vỡ.",
"Không phải hàng mong manh, không cần chống rung.",
"Hàng không nhạy va chạm; đóng gói thường là đủ.",
"Đơn này không chứa đồ dễ bể.",
"Đồ bên trong chịu va đập tốt, không cần xử lý như hàng dễ vỡ.",
"Yêu cầu vận chuyển đồ mong manh đã được hủy; đây là hàng bền.",
]
URGENT_TRUE=[
"Đơn được đánh dấu cần giao ngay.",
"Phải giao gấp, ưu tiên chuyến này.",
"Đây là lệnh khẩn, không trì hoãn.",
"Người nhận yêu cầu chuyển tới sớm nhất.",
]
URGENT_FALSE=[
"Giao theo lịch thường, không cần ưu tiên.",
"Đơn không gấp và không có hạn khẩn.",
"Chỉ chuyển khi đến lượt thông thường.",
"Đã bỏ trạng thái hỏa tốc, giao bình thường.",
]
def one(row_id,pair_id,case,goal_type,via_type,goal_loc,via_loc,prefix,via_sentence,suffix,accent):
    text=" ".join((prefix,via_sentence,suffix))
    if not accent:text=fold(text)
    goal=TargetSpec(goal_type,None,None)
    via=TargetSpec(via_type,None,None) if case=="active_via" else None
    urgent=(row_id//2)%2==0
    fragile=(row_id//4)%2==0
    n=len(tokenize(text))
    if not 12<=n<=MAX_TOKENS:raise AssertionError(("BAD_TOKENS",n))
    assert len(spec_labels(goal,via,urgent,fragile))==8
    return {"id":f"c2_{row_id:06d}","pair_id":pair_id,"case":case,
            "text":text,"goal":{"type":goal.type,"ref":goal.ref,"anchor":goal.anchor},
            "via":({"type":via.type,"ref":via.ref,"anchor":via.anchor} if via else None),
            "urgent":urgent,"fragile":fragile,"accented":accent,
            "provenance":{"source":"new_compositional_templates_v2","not_training":True}}
def generate(n_pairs=1000,seed=2026101002):
    rng=random.Random(seed);types=list(ALIASES)
    out=[];uni=set()
    for i in range(n_pairs):
        go,via=rng.sample(types,2)
        gl=rng.choice(ALIASES[go]);vl=rng.choice(ALIASES[via])
        action=rng.randrange(len(POS))
        prefix=rng.choice(GOAL).format(goal=gl)
        urgent=(i%2==0)
        fragile=((i//2)%2==0)
        suffix=rng.choice(URGENT_TRUE if urgent else URGENT_FALSE)+" "+rng.choice(FRAGILE_TRUE if fragile else FRAGILE_FALSE)
        accent=(i%4!=0)
        for active in (True,False):
            j=i*2+int(not active)
            case="active_via" if active else "cancelled_via"
            sent=(POS if active else NEG)[action].format(via=vl)
            row=one(j,f"c2_pair_{i:05d}",case,go,via,gl,vl,prefix,sent,suffix,accent)
            # Each pair has same flag/goal, only via role changes.
            assert row["urgent"]==urgent and row["fragile"]==fragile
            canon=fold(row["text"]).strip()
            if canon in uni:raise AssertionError("DUPLICATE_NORMALIZED_TEXT")
            uni.add(canon)
            out.append(row)
    return out
if __name__=="__main__":
    items=generate()
    n=len(items);tot=Counter()
    for row in items:
        v=row["via"] is not None;w=set(fold(row["text"]).split())
        tot["via"]+=v;tot["fragile"]+=row["fragile"];tot["urgent"]+=row["urgent"]
        tot["accented"]+=row["accented"]
        for x in ("lay","nhan","huy","khong","truoc"):
            if x in w:
                tot["has_"+x]+=1;tot["via_has_"+x]+=v
    summary={"experiment":"candidate_compositional_counterfactual_v2",
        "generated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pairs":n//2,"rows":n,"positive_via_pct":tot["via"]/n*100,
        "fragile_positive_pct":tot["fragile"]/n*100,
        "urgent_positive_pct":tot["urgent"]/n*100,
        "accented_pct":tot["accented"]/n*100,
        "via_given_words":{x:round(tot["via_has_"+x]/tot["has_"+x],4) if tot["has_"+x] else None
                         for x in ("lay","nhan","huy","khong","truoc")},
        "pass_all_label_checks":True,
        "limitations":["Only named target modes; spatial counterfactual variants remain to be built",
                       "Generator-created labels verified structurally, not human-audited",
                       "Candidate data is for research only; not incorporated into active 5-epoch run"]}
    p=WORK/"experiments/counterfactual_candidate_v2.jsonl"
    with p.open("x",encoding="utf-8") as f:
        for r in items:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
    summary["candidate_jsonl"]=str(p);summary["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    out=WORK/"checkpoints/counterfactual_candidate_v2_audit.json"
    with out.open("x",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2)
    print("CANDIDATE_READY",out,json.dumps(summary,ensure_ascii=False),flush=True)
