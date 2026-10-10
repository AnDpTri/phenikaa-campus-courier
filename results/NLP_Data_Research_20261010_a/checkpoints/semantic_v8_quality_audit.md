# Semantic V8 — independent QA

V8 retains full label counterfactual balance while reducing blanket negation.

| | V7 train | V8 train | V7 challenge | V8 challenge |
|---|---:|---:|---:|---:|
| Fraction containing the cue khong | 94.67% | 50.33% | 100% | 52.00% |
| P(via=True given khong) | 50% | 50% | 50% | 50% |
| Number of rows | 9600 | 9600 | 3200 | 3200 |

All ten monitored cues are invariant across the 8 counterfactual labels of each scene.
Distinct examples: 12800; scene count: 1600.
Exact overlap checks: {'old_scratch_200k': 0, 'old_v6_train': 0}.
Matched pairs similarity: {'via': {'median': 0.8669, 'min': 0.0122}, 'urgent': {'median': 0.9236, 'min': 0.0324}, 'fragile': {'median': 0.9267, 'min': 0.1974}}.
Representative positive/negative pairs saved: 48.

Remaining limitations: machine audit only; natural-language labels need manual review.
No training efficacy claim until controlled A/B against a frozen baseline.
All source files and active/legacy training left untouched.
