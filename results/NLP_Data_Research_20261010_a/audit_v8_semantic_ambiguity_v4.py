"""Release blocking semantic label ambiguity review.
Count policy-only fragile clauses which fail to entail fragile property.
No modification outside workspace.
"""
import collections,datetime,hashlib,json
from pathlib import Path
from courier.nlp.text import fold
W=Path(r"D:\phenikaa\results\NLP_Data_Research_20261010_a")
OUT=W/"checkpoints/semantic_v8_fragile_label_semantic_audit_v4.json"
if OUT.exists():raise FileExistsError(str(OUT))
patterns={
 "policy_protection_expired":"yeu cau bao ve do de be da het hieu luc",
 "policy_handling_expired":"quy dinh van chuyen do de be da het hieu luc",
 "policy_impact_protection_expired":"yeu cau chong va dap danh cho hang de vo da het hieu luc",
 "policy_protection_active":"yeu cau bao ve do de be van con hieu luc",
 "policy_handling_active":"quy dinh van chuyen do de be van con hieu luc"}
counts={}
for split,file in [
 ("train",W/"experiments/semantic_v8_balanced_train.jsonl"),
 ("challenge",W/"experiments/semantic_v8_family_challenge.jsonl")]:
 n=0;head=collections.Counter();byfamily=collections.Counter();examples={}
 for line in file.open(encoding="utf-8"):
  row=json.loads(line);n+=1
  norm=fold(row["text"])
  for name,phrase in patterns.items():
   if phrase in norm:
    label=bool(row["fragile"])
    head[(name,label)]+=1
    byfamily[(row["family_id"],row["provenance"]["phrase_style"],name,label)]+=1
    if name not in examples:
     examples[name]={"id":row["id"],"text":row["text"],"label_fragile":label,
                      "family_id":row["family_id"],"style":row["provenance"]["phrase_style"]}
 cand_false=sum(cnt for (name,label),cnt in head.items() if not label)
 cand_true=sum(cnt for (name,label),cnt in head.items() if label)
 counts[split]={"n":n,"potential_ambiguity_n":cand_true+cand_false,
  "potential_ambiguity_fraction":round((cand_true+cand_false)/n,4),
  "false_label_policy_only_n":cand_false,"false_label_policy_only_fraction":round(cand_false/n,4),
  "false_label_policy_only_fraction_among_false":round(cand_false/(n/2),4),
  "by_clause":[{"clause":name,"label":label,"n":cnt} for (name,label),cnt in sorted(head.items())],
  "by_family":[{"family_id":fam,"style":style,"clause":name,"label":label,"n":cnt}
    for (fam,style,name,label),cnt in sorted(byfamily.items())],
  "examples":examples,"sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
record={"type":"semantic_v8_fragile_property_vs_handling_policy_ambiguity",
 "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "counts":counts,
 "interpretation":"Mission.fragile should reflect package fragility. Revocation of a handling request/policy does NOT logically prove the item is non-fragile. Continuing a handling request is suggestive but not necessarily an assertion of item composition.",
 "release_decision":"REJECT MAIN TRAINING; REPAIR LABEL-GROUNDING POLICY CLAUSES AND GET HUMAN QA",
 "more_review_needed":["mild family 3 and family 6 negative clauses rescinding fragile status/label without explicitly asserting item robustness",
                       "recognition of spatial target semantics in complex pickup clauses",
                       "short/default-flag absent texts and independent human-authored data"],
 "caveats":["Clause matching is strict normalized substring; possible additional ambiguous instances are not captured.",
 "These are potential semantic ambiguities, not independently human-confirmed label errors.",
 "No outside file modified or validation/test data accessed."]}
with OUT.open("x",encoding="utf-8") as f:json.dump(record,f,ensure_ascii=False,indent=2)
print("AMBIGUITY_AUDIT_DONE",json.dumps({s:{k:v for k,v in z.items() if k in ('n','potential_ambiguity_n','potential_ambiguity_fraction','false_label_policy_only_n','false_label_policy_only_fraction','false_label_policy_only_fraction_among_false')} for s,z in counts.items()},ensure_ascii=False),flush=True)
print("SAVED",OUT)
