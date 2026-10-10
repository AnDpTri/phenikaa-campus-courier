"""Experimental Semantic Fusion V5 data: role-balanced 2x2x2 counterfactuals.

Create-only in research workspace. No changes to original generator, train or checkpoint.
The challenge split is HELD OUT BY TEMPLATE FAMILY, not shuffled sentences.
This is for data quality research, NOT claimed as blind external evaluation.
"""
from __future__ import annotations
import collections, datetime, hashlib, importlib.util, json, random, re, sys, time
from pathlib import Path
from courier.nlp.neural import MAX_TOKENS, spec_labels
from courier.nlp.text import tokenize, fold

WORK=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
GENROOT=Path(r"D:\phenikaa\results\NLP_V5_SF200_Scratch")
GENFILE=GENROOT/"synth_weighted_v4.py"
spec=importlib.util.spec_from_file_location("sf_research_v5_cleanroom",GENFILE)
G=importlib.util.module_from_spec(spec);sys.modules[spec.name]=G;spec.loader.exec_module(G)

N_GROUPS=1000
SEED=2026101055
FAMILIES=8
TRAIN_FAMILIES=tuple(range(6))
CHALLENGE_FAMILIES=(6,7)
assert not (set(TRAIN_FAMILIES)&set(CHALLENGE_FAMILIES))

# Each family has its own *goal*, pickup semantics, fragility and urgency
# realizations. This prevents trivial leakage via template families.
GOAL=(
  "Mục tiêu bàn giao của đơn này: {loc}.",
  "Người nhận xác nhận nơi nhận cuối cùng là {loc}.",
  "Mang kiện tới {loc} và kết thúc chuyến tại đó.",
  "Địa chỉ được chốt để giao hàng: {loc}.",
  "Đơn vận chuyển này kết thúc khi giao tại {loc}.",
  "Chỉ dẫn giao cuối: đưa bưu phẩm đến {loc}.",
  "Điểm cần bàn giao sau mọi công đoạn là {loc}.",
  "Hãy hoàn tất lần giao hàng ở {loc}.",
)
# The two via alternatives talk about the EXACT SAME location and pickup action,
# but differ in whether the latest instruction is active.
PICKUP_ACTIVE=(
  "Trước đó đã hủy một chuyến lấy hàng khác. Riêng lần lấy bổ sung tại {loc} vẫn còn hiệu lực và phải làm.",
  "Biên bản cũ có điểm nhận đã bị bỏ. Bước nhận tài liệu tại {loc} là chỉ dẫn mới, cần thực hiện.",
  "Có một lệnh ghé lấy đã hủy, nhưng công đoạn tới {loc} nhận phong bì thì chưa hủy.",
  "Dù bỏ bước gom hàng trong phiếu trước, robot vẫn phải đến {loc} lấy gói bổ sung.",
  "Lệnh nhận cũ đã hết hạn. Lệnh thay thế yêu cầu ghé {loc} để nhận chứng từ.",
  "Đã hủy tuyến lấy hàng ban đầu; chặng nhận còn lại ở {loc} là bắt buộc.",
  "Bản nháp về việc lấy hàng đã rút lại. Bản mới yêu cầu đến {loc} lấy kiện rồi mới giao.",
  "Một yêu cầu nhận đã bị hủy. Lưu ý vẫn cần ghé {loc} nhận hàng trước khi đi giao.",
)
PICKUP_REVOKED=(
  "Trước đó đã hủy một chuyến lấy hàng khác. Lần lấy bổ sung tại {loc} cũng bị hủy; không còn điểm nhận phụ.",
  "Biên bản cũ có điểm nhận đã bị bỏ. Bước nhận tài liệu tại {loc} cũng là chỉ dẫn cũ; không thực hiện.",
  "Có một lệnh ghé lấy đã hủy, và công đoạn tới {loc} nhận phong bì cũng đã hủy.",
  "Đã bỏ bước gom hàng trong phiếu trước; yêu cầu tới {loc} lấy gói bổ sung cũng không còn.",
  "Lệnh nhận cũ đã hết hạn. Không có lệnh thay thế yêu cầu ghé {loc} nhận chứng từ.",
  "Đã hủy tuyến lấy hàng ban đầu; chặng nhận ở {loc} cũng bị xóa, chỉ còn chặng giao.",
  "Bản nháp về việc lấy hàng đã rút lại. Bản mới không yêu cầu đến {loc} lấy kiện nữa.",
  "Một yêu cầu nhận đã bị hủy. Việc ghé {loc} nhận hàng cũng không còn hiệu lực.",
)
# For every family, both bool labels retain overlapping words such as
# 'dễ vỡ', 'nhạy', 'gấp' and 'khẩn'; scope/negation determines semantics.
FRAGILE_TRUE=(
  "Lớp hàng có vật dễ vỡ, chống xóc trong suốt chuyến.",
  "Kiện có dụng cụ thủy tinh mong manh, cần tránh rung mạnh.",
  "Đồ bên trong dễ nứt nếu va đập; phải nâng nhẹ.",
  "Sau khi kiểm tra, yêu cầu bảo vệ hàng dễ vỡ vẫn còn hiệu lực.",
  "Có linh kiện nhạy với chấn động; robot cần đi êm.",
  "Thùng chứa đồ gốm dễ bể; dùng quy trình vận chuyển nhẹ tay.",
  "Lời nhắn trước nói hàng bền là sai; hiện xác nhận có vật dễ vỡ.",
  "Cập nhật mới khẳng định món hàng mong manh và dễ hư khi bị rung.",
)
FRAGILE_FALSE=(
  "Lớp hàng không có vật dễ vỡ; không cần chế độ chống xóc.",
  "Kiện không có dụng cụ thủy tinh mong manh; vận chuyển bình thường.",
  "Đồ bên trong chịu va đập tốt, không phải loại dễ nứt.",
  "Sau khi kiểm tra, yêu cầu bảo vệ hàng dễ vỡ đã bị hủy.",
  "Dòng ghi linh kiện nhạy với chấn động là nhầm; hàng này chịu rung được.",
  "Thùng không chứa đồ gốm dễ bể, không cần quy trình nhẹ tay đặc biệt.",
  "Lời nhắn trước nói có vật dễ vỡ là sai; đã xác nhận hàng bền.",
  "Cập nhật mới bác bỏ thông tin hàng mong manh; kiện chịu rung tốt.",
)
URGENT_TRUE=(
  "Lệnh gấp đã được xác nhận; ưu tiên giao ngay.",
  "Thời hạn khẩn vẫn hiệu lực; không được trì hoãn.",
  "Lịch cũ không gấp đã thay thế bằng yêu cầu giao sớm nhất.",
  "Ưu tiên cao được kích hoạt và cần xử lý tức thì.",
  "Người gửi rút lại lịch thường; chuyến này chuyển thành hỏa tốc.",
  "Cần hoàn tất ngay; đánh dấu đơn khẩn cấp.",
  "Không còn chờ theo thứ tự bình thường, phải giao gấp.",
  "Phiếu xác nhận trạng thái ưu tiên xử lý khẩn.",
)
URGENT_FALSE=(
  "Lệnh gấp đã được hủy; giao theo lịch thường.",
  "Thời hạn khẩn hết hiệu lực; có thể giao không vội.",
  "Lịch cũ gấp đã thay thế bằng yêu cầu giao bình thường.",
  "Ưu tiên cao bị hủy và không cần xử lý tức thì.",
  "Người gửi rút lại yêu cầu hỏa tốc; chuyến này theo lịch thường.",
  "Không cần hoàn tất ngay; đã xóa nhãn đơn khẩn cấp.",
  "Yêu cầu giao gấp không còn hiệu lực; cứ giao theo lịch.",
  "Phiếu bác bỏ trạng thái ưu tiên xử lý khẩn.",
)
assert len({len(x) for x in (GOAL,PICKUP_ACTIVE,PICKUP_REVOKED,FRAGILE_TRUE,FRAGILE_FALSE,URGENT_TRUE,URGENT_FALSE)})==1

def asdict(x):
    return None if x is None else {"type":x.type,"ref":x.ref,"anchor":x.anchor}

def get_target(g,mode,used,accented):
    v,text,u=g._target(mode,used,accented)
    assert text and v is not None
    return v,text,u

def make_row(group, family, gspec,vspec,gloc,vloc,accented,active,urgent,fragile,order):
    clauses=[
      GOAL[family].format(loc=gloc),
      (PICKUP_ACTIVE if active else PICKUP_REVOKED)[family].format(loc=vloc),
      (URGENT_TRUE if urgent else URGENT_FALSE)[family],
      (FRAGILE_TRUE if fragile else FRAGILE_FALSE)[family],
    ]
    if order==1:clauses=[clauses[1],clauses[0],clauses[2],clauses[3]]
    elif order==2:clauses=[clauses[2],clauses[0],clauses[1],clauses[3]]
    elif order==3:clauses=[clauses[0],clauses[3],clauses[1],clauses[2]]
    text=" ".join(clauses)
    if not accented:text=fold(text)
    tokens=len(tokenize(text))
    if not (12<=tokens<=MAX_TOKENS):
        raise ValueError(f"TOKEN_BOUND group={group} family={family} tokens={tokens} max={MAX_TOKENS} text={text}")
    via=vspec if active else None
    assert len(spec_labels(gspec,via,urgent,fragile))==8
    return {
       "id":f"sf5_{group:05d}_{int(active)}{int(urgent)}{int(fragile)}",
       "group_id":f"scene_{group:05d}",
       "family_id":family,
       "split":"candidate_train" if family in TRAIN_FAMILIES else "family_challenge",
       "text":text,
       "goal":asdict(gspec),"via":asdict(via),
       "urgent":urgent,"fragile":fragile,
       "provenance":{"generator":"experimental_semantic_v5",
          "seed":SEED,"group_id":group,"family_id":family,
          "goal_mode":gspec.ref or "named","via_mode":vspec.ref or "named",
          "via_status":"active" if active else "revoked",
          "accented":accented,"token_count":tokens,
          "counterfactual_dimensions":"via+urgent+fragile",
          "is_official_validation":False}}
def main():
    start=time.time()
    candidates=[
        WORK/"experiments/semantic_v5_balanced_train.jsonl",
        WORK/"experiments/semantic_v5_family_challenge.jsonl",
        WORK/"checkpoints/semantic_v5_dataset_manifest.json",
    ]
    for p in candidates:
        if p.exists():raise FileExistsError(str(p))
    r=random.Random(SEED)
    gmodes=["named"]*5+["near","far","anchor_near","north","south","east","west","north_most"]
    vmodes=["named"]*6+["near","far","north","south","east","west"]
    all_seen=set()
    results=collections.defaultdict(list)
    label_counts=collections.defaultdict(collections.Counter)
    feature_counts=collections.defaultdict(collections.Counter)
    group_count=collections.Counter()
    mode_counts=collections.defaultdict(collections.Counter)
    for group in range(N_GROUPS):
        fam=group%FAMILIES
        accented=r.random()<.55
        g=G.Generator(seed=r.randrange(2**31),holdout=False,mode="hard")
        gspec,gloc,used=get_target(g,r.choice(gmodes),set(),accented)
        vspec,vloc,_=get_target(g,r.choice(vmodes),used,accented)
        order=r.randrange(4)
        sample=[]
        for active in (False,True):
            for urgent in (False,True):
                for fragile in (False,True):
                    row=make_row(group,fam,gspec,vspec,gloc,vloc,accented,active,urgent,fragile,order)
                    norm=fold(row["text"]).strip()
                    if norm in all_seen:raise RuntimeError(f"DUPLICATE_TEXT group {group}")
                    all_seen.add(norm)
                    sample.append(row)
        assert len(sample)==8
        assert len({json.dumps(e["goal"],sort_keys=True) for e in sample})==1
        for dim in ("via","urgent","fragile"):
            assert sum(bool(x[dim]) for x in sample)==4,(dim,group)
        assert len({x["family_id"] for x in sample})==1
        split=sample[0]["split"];results[split].extend(sample);group_count[split]+=1
        for x in sample:
            label_counts[split]["n"]+=1
            for dim in ("via","urgent","fragile"):
                label_counts[split][dim]+=bool(x[dim])
            label_counts[split]["accented"]+=accented
            mode_counts[split]["goal_"+(gspec.ref or "named")]+=1
            mode_counts[split]["via_"+(vspec.ref or "named")]+=1
            w=set(re.findall(r"[a-z0-9]+",fold(x["text"])))
            for tok in ("lay","nhan","truoc","huy","khong","vo","gap","khan","nhay","sau"):
                if tok in w:
                    feature_counts[split]["has_"+tok]+=1
                    for dim in ("via","fragile","urgent"):
                        feature_counts[split]["has_"+tok+"_"+dim]+=bool(x[dim])
    assert sum(len(a) for a in results.values())==8*N_GROUPS
    # Family holdout means no goal/pickup/flag template string is shared as whole phrase.
    assert set(x["family_id"] for x in results["candidate_train"])==set(TRAIN_FAMILIES)
    assert set(x["family_id"] for x in results["family_challenge"])==set(CHALLENGE_FAMILIES)
    # Check normalized exact overlap against current first 200K original training rows.
    old=GENROOT/"scratch_s0_e01_seed2026110501.jsonl"
    overlaps=0
    with old.open(encoding="utf-8") as f:
        for line in f:
            overlaps+=fold(json.loads(line)["text"]).strip() in all_seen
    assert overlaps==0,"Found exact overlap with old training corpus"
    audits={}
    for split,data in results.items():
        a=label_counts[split];b=feature_counts[split]
        assert all(a[k]==a["n"]//2 for k in ("via","urgent","fragile"))
        c={"n":a["n"],"groups":group_count[split],
           "positive_rates":{k:round(a[k]/a["n"],4) for k in ("via","urgent","fragile")},
           "accent_fraction":round(a["accented"]/a["n"],4),
           "feature_conditional":{
             token:{
               "support":b["has_"+token],
               **{k:round(b["has_"+token+"_"+k]/b["has_"+token],4)
                     if b["has_"+token] else None for k in ("via","fragile","urgent")}
             } for token in ("lay","nhan","truoc","huy","khong","vo","gap","khan","nhay","sau")},
           "mode_counts":dict(mode_counts[split]),
           "avg_tokens":round(sum(x["provenance"]["token_count"] for x in data)/len(data),1)}
        audits[split]=c
    for split,p in (("candidate_train",candidates[0]),("family_challenge",candidates[1])):
        with p.open("x",encoding="utf-8",newline="\n") as f:
            for item in results[split]:f.write(json.dumps(item,ensure_ascii=False)+"\n")
        audits[split]["path"]=str(p)
        audits[split]["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    m={"experiment":"sf5_counterfactual_8way_family_split",
       "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "seed":SEED,"groups":N_GROUPS,"total_rows":N_GROUPS*8,
       "family_assignment":{"train":TRAIN_FAMILIES,"challenge":CHALLENGE_FAMILIES},
       "dataset":audits,"exact_overlap_prior_scratch_epoch1":overlaps,
       "frozen_generator_read_only_sha256":hashlib.sha256(GENFILE.read_bytes()).hexdigest(),
       "qa_checks":["8 label combinations per base scene","within-scene same goal, same target words",
                    "role complement 4/8 per dimension","MAX_TOKENS","no exact duplicates",
                    "template-family holdout","no exact overlap original 200k","all 8 spec_labels defined"],
       "limits":["No human annotation audit","Template-family holdout is still synthetic, not real external test",
                 "Semantics of spatial targets require future manual QA",
                 "Do not train on family_challenge data; avoid using it to select checkpoints",
                 "No active train/source/model changes"],
       "elapsed_s":round(time.time()-start,1)}
    with candidates[2].open("x",encoding="utf-8") as f:json.dump(m,f,ensure_ascii=False,indent=2)
    print("SEMANTIC_V5_CREATED",json.dumps({"paths":[str(p) for p in candidates],
          "rows":{k:len(v) for k,v in results.items()},
          "audit":audits,"elapsed_s":m["elapsed_s"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
