# NLP v2 handoff — 2026-10-09

## Training

NLP v2 completed 3 seeds × 15 epochs on CUDA (RTX 3050 Laptop). The final artifact is intentionally kept outside the handoff ZIP:

`D:\phenikaa\artifacts\nlp\neural_parser_v2.pt`

The training report is `results/nlp_neural_v2/report.json`. The hybrid parser reached 100% on the grounded validation split for all tracked fields. On the held-out phrasing split, the neural model scored 68.87% overall; the hybrid rule fallback scored 47.10% overall. These held-out figures are diagnostic, not a public leaderboard score.

## End-to-end validation

With CV style routing, NLP v2, and the current solver/repair pipeline:

- 300 validation scenes
- CV scene exact: 85.67% (257/300)
- full macro accuracy: 73.10%
- full macro on CV-exact scenes: 73.74%
- full macro on CV-inexact scenes: 69.30%

The public leaderboard evaluates a hidden scored subset, so local full-test change rates must not be interpreted as a public score bound.

## Tests and reproducibility

The environment used for this run does not have `pytest` installed (`python -m pytest` cannot run). The end-to-end validation command completed successfully and its JSON report is included. Install the project test dependencies before running the complete unit-test suite.

```powershell
py -3.12 scripts/evaluate_system.py --split validation `
  --nlp-model artifacts/nlp/neural_parser_v2.pt `
  --style-artifact artifacts/cv/style_classifier.joblib `
  --report results/nlp_neural_v2/e2e_validation.json
```

Do not include datasets, credentials, predictions, caches, or model binaries in source handoffs. Copy the NLP v2 artifact separately when needed.
