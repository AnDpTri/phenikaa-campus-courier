# NLP C1 integration — 2026-10-08

Source archive: `C:/Users/andan/Downloads/nlp_c1.zip`.
SHA256: `e8b8d9750cc615bdda4b2d6516023b1d64806892bfa5e9871e692babe98294c1`.

Only `src/courier/nlp`, `scripts/nlp`, and `tests/nlp` were updated. CV,
shared domain types, solver, system evaluator, dependencies and model artifacts
were not changed. Existing concurrent CV changes were preserved.

## Changes

- Imported C1 clause segmentation, additional negation/via cues and urgency cues.
- Imported the 50-case challenge fixture and challenge/stress evaluators.
- Fixed C1 capitalisation segmentation so landmark names, generic descriptions,
  and spatial anchors do not lose their roles (e.g. `Ghé Thư viện`, `Thư Viện`,
  and `gần Cổng trường`).
- Added a clause boundary for `không cần gấp` so it does not negate the goal.
- Preserved `MissionParser`, `ParsedMission`, `TargetSpec`, and `resolve` interfaces.
- Kept the archive's unchanged vocabulary, text normaliser and resolver; no new
  vocabulary was trained or generated.
- Stress evaluation rejects test/unknown splits before loading any data and
  parses each transformed text only once.
- Tightened the existing official validation regression test to require all 300
  annotated missions to retain their correct target nodes and flags.

This is still a local rule-based parser, not a trained encoder or the full NLP
upgrade proposed in the handover report.

## Reproduction

Run from `D:/phenikaa` with the existing Python 3.12 environment:

```powershell
py -3.12 -m unittest discover -s tests/nlp -v
py -3.12 scripts/nlp/evaluate_nlp.py --split train
py -3.12 scripts/nlp/evaluate_nlp.py --split validation --solver
py -3.12 scripts/nlp/challenge_nlp.py --quiet
py -3.12 scripts/nlp/stress_nlp.py --splits train,validation --seed 20261008
```

No GPU, additional dependencies, external inference service, or CV training is
required for these checks.

## Verified results

29/29 unit tests passed. The fixed challenge fixture improved from 46/50 before
integration to 50/50 after integration. This fixture is a regression set, not an
independent holdout or a test accuracy estimate.

The same stress transformations and seed were run on the original parser and
the updated parser. Values below are the existing evaluator's `all` metric:
resolved goal/via node sets and urgent/fragile flags, not full schema exact match.

| Variant | Train before | Train after | Validation before | Validation after |
| --- | ---: | ---: | ---: | ---: |
| Original | 99.95% | 99.95% | 100% | 100% |
| No accents | 99.95% | 99.95% | 100% | 100% |
| No punctuation | 92.45% | 99.85% | 91% | 100% |
| Lowercase, no punctuation | 92.45% | 99.60% | 91% | 100% |
| Typos at 5% | 99.30% | 99.30% | 99.33% | 99.33% |
| Typos at 15% | 98.65% | 98.65% | 95.67% | 95.67% |

The stress typo RNG is seeded with `20261008:<split>:<variant>`; its results
need not match a different handover script or RNG setup. Punctuation removal
can change or obscure scope: labels are reused, not proven preserved.

Uniform-cost solver validation action accuracy remains 55.70% for both annotated
and parsed missions. This is an interface regression check, not a new measurement
of the learned strategy model or the concurrently changing CV pipeline.

## Remaining limitations

- The known train via error (`train-01552`) remains; original train accuracy did
  not improve. The spelling engine was intentionally not changed for this import.
- Resolver still substitutes missing goals and may drop unresolved vias; the
  current metrics do not separately quantify every fallback or full schema field.
- Capitalisation and clause segmentation are heuristics, not guarantees for
  arbitrary language. More unseen phrasing and negative counterexamples are needed.
- Official validation and challenge results are development measurements, not
  evidence of 100% generalisation. No test content was read for this update.
- No new end-to-end CV accuracy claim is made while another thread changes CV.
