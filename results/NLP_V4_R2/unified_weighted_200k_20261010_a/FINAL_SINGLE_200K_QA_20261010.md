# Phenikaa Courier ? One Weighted 200K Unified Corpus

Date: 2026-10-10  
Decision: **ACCEPT the single 200,000-row corpus for a controlled training experiment**, subject to remaining semantic and downstream-model caveats. There is no 40,000-row separate "specialist" block in this design.

## What changed

The original `D:\phenikaa\src\courier\nlp\synth.py` was copied byte-for-byte into a NEW results subdirectory. An integrated generator `synth_weighted_v4.py` extends the clone. It preserves the exact old V2 generator through `generate(..., mode="v2")`, while `generate(...)` now defaults to **task-weighted** mode. Semantic constructions adapted from V18 include corrected targets, retracted pickup, six negative-context roles, supplemental item pickup, spatial references, and robust via/non-via counterexamples. Training JSONL is generated directly from the one new generator; no previous 40K rows or prebuilt training sets are spliced into this file.

## Exact mixture across all 200,000 records

| Training emphasis | Weight | Rows |
| --- | ---: | ---: |
| V2 lexical and linguistic diversity | 42% | 84,000 |
| Compositional/multi-condition missions | 18% | 36,000 |
| Pickup/no-pickup semantic reasoning | 15% | 30,000 |
| Spatial grounding and referent resolution | 12% | 24,000 |
| Retraction and instruction revision | 9% | 18,000 |
| Negative/background context | 4% | 8,000 |
| **Total** | **100%** | **200,000** |

These are generation **emphases**, not mutually exclusive semantic properties.
Via, spatial and flag labels occur across multiple recipe types. This mixture
encodes a domain-prior hypothesis and has NOT been optimized with V2/V4
model ablations or Kaggle feedback.

**Distribution through file:** independently verified all 10 consecutive
20,000-row windows have exactly the same proportions (8,400 / 3,600 / 3,000 /
2,400 / 1,800 / 800). No special 40K contiguous segment.

## All-row independent quality audit

- All 200,000 JSONL rows parse and match the V2/V4 label schema.
- All 200,000 targets encode into 8 currently supported neural label heads.
- Independent QA reported **0 violations of implemented structural/semantic
  role contracts**, including former target evidence, negative-role consistency,
  positive flag wording, anchor_near exclusion on the augmented branch, and
  known recurring string-grammar defects.
- Corpus has 200,000 unique accent-folded mission texts.
- Exact duplicate checks against original 40K, curated 200K, V18 and a prior
  rejected 240K experimental corpus were performed during generation.
- Repeated initial-8-token strings were limited to **at most 35** occurrences;
  91,784 distinct initial-8-token strings were observed.
- The generator dropped 1,916 grammatically flagged candidates and
  140,536 excessive-prefix-repeat candidates, plus 3 exact duplicate
  candidates. The output retains exactly 200,000 accepted records.
- Full-set median/p90/p99/max token counts: **63 / 88 / 108 / 149**.
- Model loading smoke test: the original `train_neural_parser.py` loader read
  **200,000 / 200,000** rows; `tensors()` generated exactly 200,000 labels
  for each of 8 heads, `encode_texts()` succeeded with a 256-row batch
  of shape **[256,123,24]**.
- A separate holdout=True, 5,000-example generation check with a distinct
  seed found **0 exact overlap** with this training corpus.
- Original V2's behavior was compared in both train/holdout modes with
  400 seeded examples; exactly preserved. The original V2 file checksum
  matches the byte-exact copied file.

## Population property counts (overlapping concepts)

| Feature | Rows / mentions |
| --- | ---: |
| Active via | 113,803 |
| Spatial goal | 137,344 |
| Spatial via | 33,743 |
| Superseded prior destination | 35,815 |
| Superseded prior pickup | 23,241 |
| Superseded goal with real via | 18,796 |
| Negative/background role mentions | 85,429 |
| Via without `tr??c` | 73,302 |
| No via but contains `l?y` | 30,319 |
| Urgent true | 91,341 |
| Fragile true | 85,377 |
| Vietnamese accent present | 64,925 |

The old V2 branch remains ASCII-like and intentionally includes realistic
noisy surface forms; the new branch introduces accented variants. This
improves style coverage but is not evidence that every expression is equally
natural or correct.

## Direct semantic inspection and remaining limits

A full-range independent reservoir selected **333** cases stratified across
spatial, revocation, correction, negative mentions, named-via and no-via,
flag combinations, accented/unaccented, and uniform random picks.
The selected source-line range is 690?199,868, so the inspection is not biased
toward the beginning of the file. **27 individual cases were directly read**
during final full-range review; these did not reveal a systematic
`goal/via` reversal or uncanceled old-location label.

IMPORTANT:
1. **No guarantee** that all 200,000 texts are free from subtle semantic
   entailment/linguistic errors. The legacy V2 branch deliberately contains
   typos and some unusual synthetic constructions.
2. Current 42/18/15/12/9/4 weights are an explicit task-informed prior,
   not a proven performance-optimal class distribution. A controlled V2-seed
   baseline versus task-weighted fine-tune, with held-out per-skill metrics,
   is needed before optimizing these weights.
3. Phrase family leakage / near paraphrase reuse is not fully ruled out by
   exact folded-text duplicate controls.
4. No downstream training, checkpoint modification, Kaggle submission, or
   external validation was run.

## Final artifacts (ALL create-only under results)

Directory: `D:\phenikaa\results\NLP_V4_R2\unified_weighted_200k_20261010_a`

- `train_weighted_unified_200000_seed2026101060.jsonl` ? **the only 200K training corpus**
- `synth_weighted_v4.py` ? integrated V2 clone and semantic recipe sampler
- `synth_v2_clone.py` ? exact-byte V2 source copy
- `train_weighted_unified_200000_roles_seed2026101060.jsonl` ? sidecar provenance, NOT extra training examples
- `train_weighted_unified_200000_manifest.json` ? seeds, exact quotas, hashes and source checks
- `independent_weighted_200000_quality_report.json` ? independently verified 200K-wide QA
- `independent_semantic_strata_full_range_v2_320.jsonl` ? 333 full-range review candidates
- `additional_holdout_sampling_audit_v2.json` ? full-range sampling and 5K holdout collision check
- `generate_weighted_200000.py` and `audit_weighted_200000_independent.py` ? reproducible creation/QA

Training corpus SHA256:
`da03ac6bb3498b177ed238855489212c23df895beb4b675b7920f5ea15b1645a`

**Status:** Corpus creation and data-side automated QA complete.
**Training:** NOT STARTED.  
**Original files:** UNMODIFIED.  
**Previous experimental 240K and 40K outputs:** PRESERVED, not used as training inputs.
