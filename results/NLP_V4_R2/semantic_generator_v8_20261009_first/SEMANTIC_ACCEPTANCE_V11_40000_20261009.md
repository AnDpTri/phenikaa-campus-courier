# V4 R2 — Final Semantic Acceptance Audit for NEW 40K V11

**Date:** 2026-10-09
**Decision:** **PASS the defined generator semantic-quality gate, with stated limitations.**
**Model training:** NOT STARTED.
**Original source/code/checkpoints:** UNMODIFIED. All outputs are newly created.

## Objective and evolution

The new generator is intended to produce 40,000 **new** specialist courier-language
examples inspired by the QUALITY/COMPLEXITY of the original incoming_40000.jsonl,
not to copy or splice the original records. V7 failed the prior semantic audit.
V8 was too verbose (median 85 tokens); V9 was shorter but not fully optimized.
V10 rebuilt the semantics-first generator with complete grammatical clauses and
passed its 2,000-example contract audit and stratified sentence inspection.
V11 increases true distractor situations and uses the V10 typed architecture
without returning to unsafe concatenation.

The generator enforces a single concrete cargo across the entire pickup/dropoff
mission, selects material-appropriate cargo for fragile/not-fragile examples, gives
explicit goal/via order, and uses typed spatial phrases. `anchor_near` always
explicitly excludes the anchor itself, consistent with `resolver.py`.

## Artifacts

- NEW 40K: `specialized_new_40000_v11.jsonl` (40,000 records)
- Reproducible runner: `run_new_specialized_40k_v11_final.py`
- Base typed generator: `typed_semantic_generator.py`
- Independent contract audit: `semantic_contract_v11_40000.json`
- Reproducible independent audit runner: `run_semantic_contract_v11_40000.py`
- 311 stratified candidate rows: `semantic_stratified_v11_300.jsonl`
  (actual sample length 311 because the quota from the strata slightly
  exceeded 300)
- Generation manifest: `specialized_new_40000_v11_stats.json`

SHA256 dataset:
`6fa1fce1e2faf7d792c7625c5d65a8edbc6168cd3a2c3258db26177bdb59fcab`

## 40K full-population contract validation

The audit executed against all **40,000** rows of the V11 dataset and returned
`ISSUES {}`, meaning **no violations were found in the implemented automated
semantic contracts**.

Checked conditions include:
- Parseable goal/via specs with currently supported target references
- Exactly one internally consistent cargo identity in each mission
- The same cargo for picking up and delivering
- Cargo belongs to the appropriate fragile/nonfragile category
- Valid nonempty anchors for applicable spatial modes
- Explicit exclusion of anchor landmark for `anchor_near`
- Location class mentioned in the text of typed goal/via
- Urgent/fragile positive instructions present when the labels are true
- No original V7 `địa điểm điểm`, `kế đó mới + địa chỉ`,
  `bưu kiện xấp giấy tờ`, or literal V2 held-out phrase
- All lengths inside the neural parser max-token threshold
- Generator rejects any folded-text exact duplicates with old 40K,
  curated 200K and within the new 40K itself

IMPORTANT: Contract checking is NOT full natural-language understanding; an
unmodeled ambiguity could remain.

## Distribution comparison on the same code/regex for both populations

| Metric | Original 40K | New V11 40K |
|---|---:|---:|
| Number of records | 40,000 | 40,000 |
| Via present | 19,138 (47.85%) | 18,947 (47.37%) |
| Accented Vietnamese | 23,604 (59.01%) | 23,502 (58.76%) |
| Median tokens | 63 | 74 |
| P90 tokens | 85 | 94 |
| Distraction/background-cue lexical proxy | 29,866 (74.67%) | 28,581 (71.45%) |
| Spatial-language lexical proxy | 30,524 (76.31%) | 32,101 (80.25%) |
| Via present WITHOUT word 'trước' | 0 | **8,818** |
| Via none WITH word 'lấy' | 0 | **9,246** |

The two lexical proxy rates are illustrative, not manual classification
of true syntactic constructs. The original 40K was not itself assumed
to be perfect.

V11 median length is 11 tokens higher than original (17.5%), whereas
its P90 is 9 tokens higher (10.6%), both within the established pilot
20% statistical tolerance band.

## Stratified semantic reading

The independent V11 audit selected **311** candidates stratified across
spatial goal variants, spatial via variants, goals of the `anchor_near` or
`*_most` families, via none with `lấy`, via present without `trước`,
true/false urgent/fragile combinations, corrected instructions, accented and
unaccented text, plus independent random examples.

**140 unique sentences from different positions throughout the 311-row selection
were directly read and semantically inspected** (contiguous selection-line
offset blocks 0–9,20–27,40–47,60–69,80–87,100–107,120–129,140–147,
160–167,180–189,200–207,220–227,240–249,260–267,280–287,295–304).
No clear inconsistent goal/via labeling, reversed pickup-delivery instruction,
false `urgent`/`fragile` evidence, or negated-location-as-active-via error was
identified in that inspected subset. Remaining 171 selected candidate sentences
were NOT manually read. This is a stratified diagnostic subset rather than a
representative survey; do not claim a general error-rate confidence interval.

The selected examples are generated data, not ground truth validated by external
human annotators or tested against a map scene. Naturalness/parity is supported
but not guaranteed.

## Precise disposition

**Accept V11 as a candidate specialist-training dataset that passes the
current semantic-quality gate.** Do not claim the entire file is 100% free of
semantic errors; do not claim that training V4 on V11 will improve public Kaggle.
The next validation stage, when requested, is fine-tuning V2 with V2 synthetic
plus this NEW 40K, and comparing via/goal/all on independent heldout and
validation without training on those evaluation examples.

No neural weights were changed during this dataset-generation or audit task.
No competition submission was performed.
