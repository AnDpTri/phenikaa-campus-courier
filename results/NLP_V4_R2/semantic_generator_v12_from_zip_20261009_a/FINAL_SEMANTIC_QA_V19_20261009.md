# Phenikaa Campus Courier NLP V4 R2 — Final Specialist Dataset Quality Audit

Date: 2026-10-09
Final candidate version: **V19**; supersedes V18 and earlier exploratory datasets.
Result: **PASS full-population automated semantic contracts and V2/V4 loader
compatibility; APPROVED FOR A CONTROLLED TRAINING EXPERIMENT, NOT certified
100% error-free by exhaustive human annotation.**

## Deliverables

- New 40,000 specialist records: `specialized_new_40000_v19_role_semantics.jsonl`
- Semantic scenario provenance: `specialized_new_40000_v19_provenance.jsonl`
- Exact new-only generator runner: `run_new_40000_v19_polished.py`
- Main generator source (read only): `role_semantic_generator_v12.py`
- Generation statistics: `specialized_new_40000_v19_generation_report.json`
- Independent full-population QA: `independent_full_semantic_audit_v19.json`
- Independent QA runner: `run_independent_full_semantic_audit_v19.py`
- Stratified human-readable sample of 320: `independent_semantic_strata_v19_320.jsonl`

Dataset SHA256:
`4f7187e05c869aad7f6c51ffb54eeaa9d11c60deec290d5718685480ca25f539`

## Origin and intent

The user-supplied `robot_hard_training_dataset(1).zip` informed the generator's
**semantic scenario structure**, not the source text of generated examples.
Concepts adapted from its *training* ontology include superseded goals,
retracted pickup stops, six negative-location mention roles, and supplementary
items received at intermediate stops. No ZIP text validation sentences or
labels were used in writing the generator templates. No original 40K rows
were copied. V2/V4-supported 10 types and existing target modes remain the
only emitted labels.

## Full 40,000-row test results

Tested all **40,000/40,000** examples with an independent QA module, reading
paired per-record provenance and training labels.

Automated violations detected: **0** within the implemented contracts.

The full-population verifier independently checks:

1. Exactly the V2/V4 schema `id,text,goal,via,urgent,fragile` and the
   correct 8 neural classification heads.
2. Valid goal/via target modes and anchor/type semantics; type evidence in
   text; explicit anchor exclusion for `anchor_near`.
3. Recalled old destination and/or pickup are *explicitly revoked* and never
   repurposed as the active goal or pickup stop.
4. Negative mentions have separate role, mention the intended place, and
   do not negate the current goal/via or their anchors.
5. Main goods mentioned; supplementary pickup items occur only with a via;
   true urgent/fragile labels have corresponding explicit language.
6. All 40K rows within 14–160 model-token limits; maximum 158 tokens.
7. Known grammatical anti-patterns, including `khu vực khu`, `địa điểm khu`,
   `địa danh khác`, `chặng trung gian lấy`, `bưu kiện thùng`, and
   invalid temporal connectors with static goal predicates: zero matches.
8. Unique accent-folded mission texts within V19, zero exact folded-text
   overlap with original `incoming_40000.jsonl`, curated 200K, prior V11,
   or previous V16/V17/V18 pilot files.

The generator rejected 16 candidate exact duplicates and 3 candidate
semantic-or-length exceptions before writing the final 40,000 rows.

## Core quantitative coverage

| Measure | V19 |
|---|---:|
| New rows | 40,000 |
| Via present | 19,117 (47.79%) |
| Prior destination revoked | 9,424 (23.56%) |
| Prior destination revoked + real via | 4,494 (11.24%) |
| Prior pickup revoked | 5,383 (13.46%) |
| Separate supplementary pickup item | 10,066 (25.17%) |
| Role-tagged negative mentions | 18,042 |
| Via present WITHOUT word `trước` | 11,397 |
| Via absent WITH word `lấy` | 8,829 |
| Accented Vietnamese | 23,541 (58.85%) |
| Median / p90 / p99 / maximum tokens | 73 / 102 / 127 / 158 |

Against the original 40K: original median 63 and p90 85 model tokens;
thus V19 remains longer, although below the model's max length.
Original `via` presence ~47.85%, accented ~59%, and 8,263 examples
with an explicit original-goal-revision lexical pattern. V19 has
9,424 *structurally guaranteed* superseded goals, but these are
different measurement methods and should not be treated as directly
comparable semantic coverage estimates.

## Real trainer compatibility smoke test

`scripts/nlp/train_neural_parser.py:labelled_prebuilt` loaded all **40,000**
rows without error. The `tensors` function produced 40,000 labels for each:
`goal_mode`, `goal_type`, `goal_anchor`, `via_mode`, `via_type`,
`via_anchor`, `urgent`, and `fragile`.

`encode_texts` succeeded on a 256-row text batch; actual encoded
shape **[256, 130, 24]**. No optimizer, weights update, or training
was invoked.

## Semantic reading versus automated tests

320 examples were independently selected across 16 strata including
both goal/pickup retractions, global extreme and anchor-near goals,
relative via, negative mention roles, supplementary items, and hard
flag combinations. **39 different rows of this selection were directly
read and inspected** during the final V19 quality review; no clear
reversal of active versus superseded target or wrong value for
urgent/fragile was found in those read rows.

This is *not* an exhaustive human semantic audit of 40,000 natural
language statements, and 0 automated violations is **not** proof of
0 true errors. Exact/folded deduplication does not certify removal of
near-paraphrase/template duplicates. Some sentences are synthetically
regular or slightly verbose. Multi-modal route quality or downstream
V4 accuracy was not benchmarked for V19.

## Training disposition

**Use V19, not V7–V18, as the specialist 40K source for a separate,
controlled V2-to-V4 R2 fine-tuning experiment.** Fresh V2 synthetic
200K/epoch must be generated and combined explicitly in a new trainer,
not with the old `prebuilt or generate` mutually exclusive branch.

Preserve validation/holdout and test separation; ZIP validation
was NOT reused for authored templates or training.

No training, Kaggle submission, or modification of existing
`D:\phenikaa` source files, checkpoints, or corpora occurred.
Earlier pilot artifacts have been preserved without edits.
