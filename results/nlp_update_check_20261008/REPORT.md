# Independent verification of NLP update 57f5dd2

Date: 2026-10-08. The checks below were rerun locally after the update; they are not copied from the commit's README.

## Reproduced results

- NLP unit tests: 29/29 passed.
- Train original: 1,999/2,000 exact mission results (99.95%).
- Validation original: 300/300 exact mission results (100%).
- Hand-written challenge fixture: 50/50; goal 50/50, via 50/50, urgent 9/9, fragile 8/8.
- Uniform-cost solver validation: 55.70% with annotated missions and 55.70% with parsed missions.

Stress results (`all`, seed 20261008):

| Variant | Train | Validation |
| --- | ---: | ---: |
| Original | 99.95% | 100% |
| No accents | 99.95% | 100% |
| No punctuation | 99.85% | 100% |
| Lowercase and no punctuation | 99.60% | 100% |
| Typos at 5% | 99.30% | 99.33% |
| Typos at 15% | 98.65% | 95.67% |

The punctuation robustness improvement is reproduced. The known train error `train-01552` remains: the real-word typo “thể tao” is not corrected to “thể thao”, so the sports via is missed.

## Interpretation and limits

- The update is still a deterministic rule-based parser, not the proposed learned local encoder.
- The 50 challenge cases are committed together with the rules and tests. They are a regression suite, not an independent holdout and not an estimate of private-test accuracy.
- Stress transformations reuse the original labels. Removing punctuation can alter scope, so the resulting scores are sensitivity probes rather than guaranteed semantic accuracy.
- Original validation has already influenced development and should not be treated as an unbiased final estimate.
- Resolver goal substitution and via dropping can still hide parser failures unless fallback diagnostics are reported separately.
- The update states that challenge text was written from general Vietnamese and not test. The local rerun verifies behavior and that stress tooling rejects the test split; it cannot independently prove the provenance of externally supplied challenge sentences.
- No test content was inspected or used during this verification.

## Decision

The update is safe as a regression improvement for missing punctuation: it preserves original train/validation performance and greatly improves the defined punctuation stress cases. Further claims about large-scale generalization require a separately created, frozen holdout or a learned-model comparison. The typo and silent-resolver limitations remain open.
