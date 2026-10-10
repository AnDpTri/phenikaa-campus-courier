"""Counterfactual VIA paired stress test; ONLY writes inside research workspace."""
import collections, datetime, hashlib, json, random, statistics, time
from pathlib import Path
from courier.nlp.neural import NeuralMissionParser, MAX_TOKENS, spec_labels
from courier.nlp.parser import TargetSpec
from courier.nlp.text import tokenize,fold
ROOT=Path(r"D:\phenikaa")
WORK=ROOT/"results/NLP_Data_Research_20261010_a"
ORIG=ROOT/"results/NLP_V5_SF200_Scratch"
rng=random.Random(202610101433)
PLACES={
"library":["thu vien","thu vien trung tam","phong doc sach"],
"dorm":["ky tuc xa","khu noi tru","nha o sinh vien"],
"sports":["nha thi dau","san bong ro","khu the chat"],
"clinic":["tram y te","phong y te","phong kham"],
"canteen":["nha an","canteen","quay com"],
"parking":["bai xe","nha de xe","khu giu xe"],
"lecture":["giang duong","khu lop hoc","nha hoc"],
"lab":["phong thi nghiem","phong lab","xuong thuc hanh"],
"office":["phong dao tao","van phong khoa","toa hieu bo"],
"gate":["cong chinh","cong phu","loi ra vao"]
}
GOALS=[
"Giao kien hang den {goal}.",
"Robot can dua thung hang toi {goal} de ban giao.",
"Dia chi giao cuoi cung la {goal}.",
"Nguoi nhan cho hang tai {goal}.",
"Chuyen goi tai lieu toi {goal}.",
"Don hang nay co diem giao la {goal}.",
]
VIA_POS=[
"Diem lay bo sung la {via}. Robot phai ghe lay roi moi giao.",
"Can nhan phong bi tai {via}; sau do moi den dia diem giao.",
"Ghe {via} lay tai lieu bo sung truoc khi giao.",
"Hay den {via} de nhan them hang, roi tiep tuc giao.",
"Khong bo qua buoc lay tai {via}; do la diem nhan bat buoc.",
"Chuyen truoc den {via} lay chung tu, sau do giao hang.",
]
VIA_NEG=[
"Diem lay bo sung cu tung la {via}, nhung lenh ghe lay da huy. Khong den do lay nua.",
"Phieu cu ghi nhan phong bi tai {via}, nhung da huy viec nhan. Chi can giao.",
"Hoi truoc bao ghe {via} lay tai lieu nhung yeu cau nay da duoc rut lai.",
"Khong ghe {via} de nhan them hang; kien da co san tu diem xuat phat.",
"Khong bo qua buoc lay tai {via} la cau trong phieu cu da huy. Bay gio khong lay nua.",
"Chuyen truoc den {via} lay chung tu la lenh cu. Huy buoc nay va giao thang.",
]
FLAG_URGENT=[
("Can giao gap, uu tien xu ly.",True),
("Khong gap, giao theo lich binh thuong.",False)
]
FLAG_FRAGILE=[
("Trong kien co do de vo, nhe tay.",True),
("Kien khong co do de vo.",False)
]
def row(t,g,v,u,f,pid,kind,template):
    assert 12<=len(tokenize(t))<=MAX_TOKENS,(len(tokenize(t)),t)
    assert len(spec_labels(g,v,u,f))==8
    return {"text":t,"goal":{"type":g.type,"ref":g.ref,"anchor":g.anchor},
            "via":({"type":v.type,"ref":v.ref,"anchor":v.anchor} if v else None),
            "urgent":u,"fragile":f,"pair_id":pid,"case":kind,"template_id":template}
def examples(num_pairs=600,seed=202610101433):
    r=random.Random(seed)
    types=list(PLACES)
    pairs=[]
    for i in range(num_pairs):
        t1,t2=r.sample(types,2)
        goal=r.choice(PLACES[t1]);via=r.choice(PLACES[t2])
        pre=r.choice(GOALS).format(goal=goal)
        x=i%len(VIA_POS)
        vpos=VIA_POS[x].format(via=via)
        vneg=VIA_NEG[x].format(via=via)
        urgent,u=r.choice(FLAG_URGENT);fragile,f=r.choice(FLAG_FRAGILE)
        suffix=" "+urgent+" "+fragile
        if r.random()<.5:
            pos=pre+" "+vpos+suffix;neg=pre+" "+vneg+suffix
        else:
            pos=vpos+" "+pre+suffix;neg=vneg+" "+pre+suffix
        g=TargetSpec(t1,None,None)
        v=TargetSpec(t2,None,None)
        pid=f"cvia_{i:04d}"
        pairs.extend([row(pos,g,v,u,f,pid,"active_via",x),
                      row(neg,g,None,u,f,pid,"cancelled_via",x)])
    return pairs
def inspect(rows, model_name, model):
    results=[];n=0;score=collections.Counter();cases=collections.defaultdict(lambda:collections.Counter())
    for start in range(0,len(rows),64):
        block=rows[start:start+64]
        pred=model.predict_batch([x["text"] for x in block])
        for r,p in zip(block,pred):
            n+=1
            truth=r["via"] is not None
            expect=r["via"]["type"] if truth else None
            got=p.parsed.via.type if p.parsed.via else None
            hit=(got==expect)
            group=cases[r["case"]]
            group["n"]+=1;group["correct_via"]+=hit
            group["false_negative"]+=truth and p.parsed.via is None
            group["false_positive"]+=(not truth) and p.parsed.via is not None
            group["goal_correct"]+=p.parsed.goal is not None and p.parsed.goal.type==r["goal"]["type"]
            score["via"]+=hit
            results.append({"pair_id":r["pair_id"],"case":r["case"],"truth_via":expect,
                            "pred_via":got,"via_correct":hit,"conf":round(p.via_confidence,5),
                            "text":r["text"]} )
    both=collections.defaultdict(list)
    for a in results:both[a["pair_id"]].append(a["via_correct"])
    fully=sum(len(v)==2 and all(v) for v in both.values())
    return {"model":model_name,"n":n,"via_acc":round(score["via"]/n,4),
            "both_counterfactual_correct_fraction":round(fully/(n/2),4),
            "cases":{k:{j:(round(value/c["n"],4) if j!="n" else value) for j,value in c.items()}
                     for k,c in cases.items()},
            "mistakes":sorted([x for x in results if not x["via_correct"]],
                              key=lambda x:-x["conf"])[:30]}
if __name__=="__main__":
    import torch
    torch.set_num_threads(2)
    rows=examples()
    outputs=[ORIG/"scratch_s0_epoch01.pt",ORIG/"scratch_s0_epoch02.pt"]
    metrics=[]
    for path in outputs:
        if not path.is_file():continue
        net=NeuralMissionParser.load(path)
        for m in net.nets:m.cpu().eval()
        with torch.no_grad():metrics.append(inspect(rows,path.name,net))
        del net
    result={"test":"paired_counterfactual_via_v1","n_pairs":len(rows)//2,
            "total_examples":len(rows),"models":metrics,
            "tested_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "limits":"Handwritten templates use synthetic place aliases; stress test, not blind external evaluation. No model tuning."}
    target=WORK/"checkpoints/counterfactual_via_v1.json"
    with target.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2)
    rows_out=WORK/"experiments/counterfactual_via_pairs_v1.jsonl"
    with rows_out.open("x",encoding="utf-8") as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print("COUNTERFACTUAL_COMPLETE",target)
    for a in metrics:print(a["model"],"via",a["via_acc"],"paired",a["both_counterfactual_correct_fraction"],"by_case",a["cases"])
