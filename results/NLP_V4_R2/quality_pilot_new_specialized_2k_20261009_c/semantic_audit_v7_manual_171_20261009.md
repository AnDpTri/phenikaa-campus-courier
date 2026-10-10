# V4 R2 — Semantic Audit of specialized 2K pilot V7 (2026-10-09)

## Verdict

**FAIL semantic-parity gate. DO NOT train on or scale up V7 as is.**

The previous independent automated quality gate (15/15) was necessary but inadequate.
This report distinguishes (a) programmatic observations over all 2,000 generated rows,
(b) semantic/linguistic human-readable inspection of 171 uniquely sampled rows, and
(c) genuine reference-label uncertainty.

All data / source / model artifacts predating this report remain unchanged. No training
or Kaggle submission was performed.

## Sources and review protocol

- Pilot: `pilot_new_specialized_2000_v7.jsonl` (2,000 rows)
- Original reference: `D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review\incoming_40000.jsonl` (40,000 rows)
- Stratified reading selection: `semantic_audit_selection_v1.jsonl` (171 unique source rows)
- Existing automated gates: `independent_quality_audit_v3.json`
- Sample strata included: goal anchor_near (10), goal near/far (10), goal *_most (10),
  goal direction (10), via near/far (10), via direction (10), via named without
  "truoc" (10), via none with "lay" (10), via none with "truoc" (10),
  via present with negative "khong can ghe" (10), urgent & fragile true (10),
  both false (10), redirect/correction (10), plus 45 random selections;
  duplicates in selection were excluded, for 171 unique reviewed sentences.

The 171 reviewed rows are a targeted, stratified sample, NOT a probability sample. 
Do not extrapolate a human-rated error frequency from 171 to the entire 2,000.

## Programmatic counts across the complete 2,000

| Finding | V7 | Interpretation |
| --- | ---: | --- |
| Unique records | 2,000/2,000 | Pass |
| Exact folded-text overlap vs original 40K / curated 200K | 0 | Pass; near-duplicate families not certified |
| Positive urgent/fragile lacking a literal authored-bank evidence phrase | 0 | Pass on narrow template-grounded criterion; no full entailment proof |
| `ve dia diem` | 177 (8.85%) | Awkward construction; zero matches in original 40K |
| `ve dia diem (diem/vi tri/o/...)` and similarly marked patterns | 64 | Risk of doubled noun phrases |
| Temporal connector followed by static destination/receiver description | 133 | High-risk compositional syntax; some are clear ungrammatical joins |
| Broad awkward connector detector | 378 | Overinclusive diagnostic, **not** 378 demonstrated errors |
| `anchor_near` goal | 234 | Semantically sensitive to excluding anchor from candidates |
| Of the 234 anchor_near rows: explicit exclusion of anchor location | 54 | Clear |
| Of the 234 anchor_near rows: no explicit exclusion | 180 | Ambiguity risk; **not** automatically 180 wrong labels |
| All recognized mechanical syntactic anti-patterns checked in 15/15 gate | 0 | Narrow gate passed; manually found new error families |

The map resolver `src/courier/nlp/resolver.py` in fact removes the anchor coordinate
before selecting an `anchor_near` destination. A phrase merely saying
"điểm có khoảng cách ngắn nhất tới X" does not always unambiguously specify this
rule, so such examples need re-authoring or explicit constraint.

## Positive examples of semantics preserved in reviewed samples

- `specnew_000696`: item picked at starting depot; correctly `via=null` despite "lấy".
- `specnew_001241`: library is delivery target while clinic is actual pickup stop;
  lecture hall explicitly excluded; goal/via parsed consistently.
- `specnew_001464`: gate (north) final goal, lecture hall intermediate via;
  urgency present.
- `specnew_000207`: goal lecture-near-lab, via dorm-west; spatial anchors and order align.
- `specnew_001312`: goal west-most, via clinic-near-library; negative distractor is separate.

These are inspected examples, not a guarantee that the whole corpus is correct.

## Failures and risks substantiated by reading full sentences

1. **`specnew_001266`**: "đưa hồ sơ xét duyệt về địa điểm điểm sát rìa nam nhất..." —
   duplicated location noun; goal semantics recoverable, but the generated sentence
   is not polished natural Vietnamese.
2. **`specnew_001921`**: "đưa túi đồ cá nhân về địa điểm điểm ở mép bắc..." —
   same systematic surface construction.
3. **`specnew_001511`**: "kế đó mới điểm kết thúc đơn hàng đặt tại..." —
   action-order connector requires a verb phrase; instead receives a standalone
   noun-predicate describing destination.
4. **`specnew_001594`**: "kế đó mới người nhận chờ túi mẫu thí nghiệm tại..." —
   comparable temporal-connector mismatch.
5. **`specnew_001519`**: "rồi kết thúc bằng việc địa chỉ nhận cuối cùng là..." —
   static proposition inserted into an action-nominal construction.
6. **`specnew_001054`**: shipping "hộp thực phẩm" while explaining "bên trong chỉ
   có tài liệu bền"; item-content mismatch raises semantic contradiction risk,
   particularly around the fragile label.
7. **`specnew_000072`**: "đưa thẻ sinh viên về địa điểm vị trí gần phòng thí
   nghiệm nhất..." — noun duplication combined with ambiguous nearest-to-anchor
   reference.
8. **`specnew_001379`**, **`specnew_000498`**: generic 'closest place to anchor'
   requests do not explicitly prohibit choosing the anchor itself, unlike the
   project's map resolver.

Semantic context sometimes rescues awkward sentences. The review avoids treating
every string match as a semantic label error.

## Comparison with original 40K

| Metric | Original 40K | Pilot V7 |
| --- | ---: | ---: |
| Median tokens | 63 | 74 |
| P90 tokens | 85 | 93 |
| Via-present rate | 47.85% | 49.05% |
| Vietnamese accents | 59.01% | 60.30% |
| Correction / redirect lexical cue rate | 48.41% | 51.30% |
| Distractor-context lexical cue rate | 64.96% | 46.60% |
| Spatial lexical cue rate | 74.33% | 81.60% |
| Presence of `ve dia diem` | 0 / 40,000 | 177 / 2,000 |

The original 40K also contains complex phrases and potential semantic ambiguity.
These comparisons do **not** certify the reference 40K as entirely error-free.
They do demonstrate the V7 generator did not reproduce its naturalness reliably.

## Decision and next engineering priorities

**DO NOT train / integrate / scale V7.** Retain the files as rejected diagnostic artifacts.

Before a new generator is fit for 40,000:

1. Author goal clauses and action-order transition clauses as compatible grammatical
   types. For example, let an action connector accept an imperative/verb-clause, not
   a raw destination-description nominal clause.
2. Remove global `ve dia diem` wrapping; avoid replacing phrases by textual
   concatenation with unmatched syntactic requirements.
3. Encode `anchor_near` with explicit exclusion of the anchor node in every
   variant, consistent with the resolver.
4. Tie fragile explanations to actual item contents; reject mutually exclusive
   material descriptions against the generated main item.
5. Increase compositional distractor diversity toward original 40K without copying
   old sentences or reinstating the "truoc" shortcut.
6. Re-audit a *fresh*, seeded and separate pilot using both automatic and independent
   sentence-level semantics; only consider training after this gate passes.

**Status: SEMANTIC_AUDIT_FAILED. Training: NOT STARTED. Original files: PRESERVED.**
