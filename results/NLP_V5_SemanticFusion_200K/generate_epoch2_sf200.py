"""V5-SF200: generate epoch-2 200K from the unified weighted generator.

All outputs are NEW, created with exclusive opens. No pre-existing source,
corpus, validation, model or checkpoint is edited. Uses train banks only.
"""
from __future__ import annotations
import collections,hashlib,importlib.util,json,random,re,sys,time
from pathlib import Path
from courier.nlp.text import fold,tokenize
from courier.nlp.neural import MAX_TOKENS,spec_labels

ROOT=Path(r"D:\phenikaa")
RUN=ROOT/"results/NLP_V5_SemanticFusion_200K"
DATA=ROOT/"results/NLP_V4_R2/unified_weighted_200k_20261010_a"
OUT=RUN/"sf200_epoch2_seed2026101061.jsonl"
ROLES=RUN/"sf200_epoch2_roles_seed2026101061.jsonl"
STATS=RUN/"sf200_epoch2_manifest.json"
for target in (OUT,ROLES,STATS):
    if target.exists():raise FileExistsError("Refuse overwrite: "+str(target))
base=DATA/"synth_weighted_v4.py"
sp=importlib.util.spec_from_file_location("v5_unified_generator_epoch2",base)
generator=importlib.util.module_from_spec(sp)
sys.modules[sp.name]=generator
sp.loader.exec_module(generator)
weights=generator.WEIGHTED_RECIPES
assert sum(weights.values())==100
sources=[
 DATA/"train_weighted_unified_200000_seed2026101060.jsonl",
 ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/incoming_40000.jsonl",
 ROOT/"results/nlp_v4_data_350k/replacement_20261009_review/nlp_v4_curated_200000_quality_v2_reindexed.jsonl",
 ROOT/"results/NLP_V4_R2/semantic_generator_v12_from_zip_20261009_a/specialized_new_40000_v18_role_semantics.jsonl",
 ROOT/"results/NLP_V4_R2/unified_v2_v18_clone_20261009_a/train_unified_240000_interleaved_seed20261009.jsonl",
]
reference=set()
for path in sources:
    with path.open(encoding="utf-8") as stream:
        for line in stream: reference.add(fold(json.loads(line)["text"]).strip())
    print("REFERENCE_READY",path.name,len(reference),flush=True)
g=generator.Generator(seed=2026101061,holdout=False,mode="weighted")
names=tuple(weights)
acc={name:0 for name in names}
counts=collections.Counter()
rejected=collections.Counter()
prefix=collections.Counter()
seen=set()
bad=re.compile(r"\b(khu vuc khu|dia diem diem|dia diem vi tri|buu kien buu kien|ngoai kien buu kien|ke do moi dia chi|sau do can dia chi|tai tai|o o)\b")
start=time.time()
def target(x):
    return None if x is None else {"type":x.type,"ref":x.ref,"anchor":x.anchor}
with OUT.open("x",encoding="utf-8",newline="\n") as data_file,ROLES.open("x",encoding="utf-8",newline="\n") as role_file:
    for idx in range(200000):
        for name in names:acc[name]+=weights[name]
        recipe=max(names,key=lambda name:(acc[name],-names.index(name)))
        acc[recipe]-=100
        g._forced_recipe=recipe
        for attempt in range(700):
            ex=g.example()
            folded=fold(ex.text).strip()
            tokens=len(tokenize(ex.text))
            first8=" ".join(folded.split()[:8])
            if tokens<8 or tokens>MAX_TOKENS:
                rejected["length"]+=1;continue
            if folded in reference or folded in seen:
                rejected["duplicate"]+=1;continue
            if bad.search(folded):
                rejected["grammar"]+=1;continue
            if prefix[first8]>=35:
                rejected["prefix_cap"]+=1;continue
            role=g.last_provenance.copy()
            if recipe!="v2_general":
                if ex.urgent and not any(fold(q) in folded for q in generator.URGENT_POS):raise AssertionError("urgent")
                if ex.fragile and not any(fold(q) in folded for q in generator.FRAGILE_POS):raise AssertionError("fragile")
                for k in ("former_destination","former_pickup"):
                    if role[k] and fold(role[k]) not in folded:raise AssertionError(k)
                for nm in role["negative_roles"]:
                    if fold(nm["text"]) not in folded:raise AssertionError("negative role")
                if ex.goal.ref=="anchor_near" and not any(x in folded for x in ("khong tinh","loai tru","ngoai","khong ke","tru moc")):
                    raise AssertionError("ambiguous anchor")
            assert len(spec_labels(ex.goal,ex.via,ex.urgent,ex.fragile))==8
            seen.add(folded)
            prefix[first8]+=1
            counts[recipe]+=1
            counts["via"]+=ex.via is not None
            counts["accented"]+=any(ord(c)>127 for c in ex.text)
            record={"id":f"sf200_e2_{idx+1:06d}","text":ex.text,
                    "goal":target(ex.goal),"via":target(ex.via),
                    "urgent":ex.urgent,"fragile":ex.fragile}
            data_file.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
            role_file.write(json.dumps({"id":record["id"],"recipe":recipe,**role},ensure_ascii=False,separators=(",",":"))+"\n")
            break
        else:
            raise RuntimeError(f"Insufficient diversity for {recipe} at {idx+1}")
        if idx%25000==24999:
            print("EPOCH2_GENERATED",idx+1,"quota",dict((k,counts[k]) for k in names),"rejected",dict(rejected),flush=True)
expected={k:weights[k]*2000 for k in names}
assert all(counts[k]==v for k,v in expected.items())
summary={
 "status":"GENERATED_PENDING_INDEPENDENT_QA",
 "count":200000,"seed":2026101061,
 "recipe_counts":{k:counts[k] for k in names},
 "statistics":dict(counts),"rejected":dict(rejected),
 "unique_prefix8":len(prefix),"max_prefix8":max(prefix.values()),
 "no_overlap_with_epoch1_and_reference":True,
 "sha256":hashlib.sha256(OUT.read_bytes()).hexdigest(),
 "elapsed_seconds":round(time.time()-start,1),
}
with STATS.open("x",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2)
print("EPOCH2_READY",json.dumps(summary,ensure_ascii=True),flush=True)
