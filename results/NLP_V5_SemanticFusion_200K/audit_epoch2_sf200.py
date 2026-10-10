"""Independent epoch2 all-row QA, E1/E2 dedup, and V5 trainer preflight."""
from pathlib import Path
import collections,hashlib,json,re,importlib.util,sys
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import spec_labels,MAX_TOKENS
from courier.nlp.parser import TargetSpec
ROOT=Path(r"D:\phenikaa")
RUN=ROOT/"results/NLP_V5_SemanticFusion_200K"
E1=ROOT/"results/NLP_V4_R2/unified_weighted_200k_20261010_a/train_weighted_unified_200000_seed2026101060.jsonl"
E2=RUN/"sf200_epoch2_seed2026101061.jsonl"
ROLES=RUN/"sf200_epoch2_roles_seed2026101061.jsonl"
STATS=RUN/"sf200_epoch2_manifest.json"
REPORT=RUN/"sf200_epoch2_quality_audit.json"
if REPORT.exists():raise FileExistsError(REPORT)
stats=json.loads(STATS.read_text(encoding="utf-8"))
sp=importlib.util.spec_from_file_location("v5_check_weighted",ROOT/"results/NLP_V4_R2/unified_weighted_200k_20261010_a/synth_weighted_v4.py")
mod=importlib.util.module_from_spec(sp);sys.modules[sp.name]=mod;sp.loader.exec_module(mod)
set1=set()
with E1.open(encoding="utf-8") as f:
    for line in f:set1.add(fold(json.loads(line)["text"]).strip())
assert len(set1)==200000
seen=set();counts=collections.Counter();issues=collections.Counter();examples=collections.defaultdict(list)
first8=collections.Counter();size=0
def warn(k,j,msg):
    issues[k]+=1
    if len(examples[k])<5:examples[k].append({"id":j.get("id"),"problem":str(msg)[:180]})
with E2.open(encoding="utf-8") as a,ROLES.open(encoding="utf-8") as b:
    for row_id,(line,meta_line) in enumerate(zip(a,b),1):
        j=json.loads(line);meta=json.loads(meta_line)
        size+=1
        t=fold(j["text"]).strip()
        if j["id"]!=f"sf200_e2_{row_id:06d}" or meta["id"]!=j["id"]:warn("id",j,row_id)
        if set(j)!={"id","text","goal","via","urgent","fragile"}:warn("fields",j,list(j))
        if t in set1:warn("overlap_e1",j,t[:75])
        if t in seen:warn("overlap_in_e2",j,t[:75])
        seen.add(t)
        first8[" ".join(t.split()[:8])]+=1
        count_tokens=len(tokenize(j["text"]))
        if not 8<=count_tokens<=MAX_TOKENS:warn("max_tokens",j,count_tokens)
        gg=j["goal"];vv=j["via"]
        try:
            g=TargetSpec(gg["type"],gg["ref"],gg["anchor"])
            v=None if vv is None else TargetSpec(vv["type"],vv["ref"],vv["anchor"])
            assert len(spec_labels(g,v,j["urgent"],j["fragile"]))==8
        except Exception as e:warn("label",j,e)
        recipe=meta["recipe"];counts[recipe]+=1
        if recipe!="v2_general":
            if meta["goal_mode"]!=(gg["ref"] or "named"):warn("goal_mode",j,meta)
            if meta["via_mode"]!=("none" if vv is None else (vv["ref"] or "named")):warn("via_mode",j,meta)
            for k in ("former_destination","former_pickup"):
                if meta[k] and fold(meta[k]) not in t:warn("missing_"+k,j,meta[k])
            for n in meta["negative_roles"]:
                if fold(n["text"]) not in t:warn("negative_not_mentioned",j,n)
                if n["type"] in (gg["type"],gg["anchor"],None if vv is None else vv["type"],None if vv is None else vv["anchor"]):
                    warn("neg_conflicts_active",j,n)
            if j["urgent"] and not any(fold(q) in t for q in mod.URGENT_POS):warn("urgent_cue",j,j["text"])
            if j["fragile"] and not any(fold(q) in t for q in mod.FRAGILE_POS):warn("fragile_cue",j,j["text"])
        if re.search(r"\b(khu vuc khu|dia diem diem|buu kien buu kien|ngoai kien buu kien|tai tai|o o)\b",t):warn("bad_grammar",j,t[:100])
    if a.readline() or b.readline():raise RuntimeError("Extra unpaired record")
if size!=200000:raise RuntimeError("wrong size "+str(size))
for recipe,percentage in mod.WEIGHTED_RECIPES.items():
    if counts[recipe]!=percentage*2000:warn("recipe",{"id":"quota"},(recipe,counts[recipe],percentage*2000))
if first8.most_common(1)[0][1]>35:warn("prefix_cap",{"id":"distribution"},first8.most_common(1))
sha=hashlib.sha256(E2.read_bytes()).hexdigest()
if sha!=stats["sha256"]:warn("sha_manifest",{"id":"file"},sha)
report={"passed":not issues,"total":size,"e1_exact_overlap":issues["overlap_e1"],
        "recipes":dict(counts),"unique_first8":len(first8),
        "max_first8_count":first8.most_common(1)[0][1],
        "sha256":sha,"issues":dict(issues),"examples":dict(examples)}
with REPORT.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print("EPOCH2_QA",json.dumps(report,ensure_ascii=True),flush=True)
if not report["passed"]:raise SystemExit(2)
