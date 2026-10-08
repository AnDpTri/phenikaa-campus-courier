# Integration status — 2026-10-08

The current pipeline combines learned CV, the Vietnamese mission parser, and
the new 48-profile candidate-action strategy ranker.

| Validation metric (300 scenes) | Previously submitted version | Candidate version |
|---|---:|---:|
| Oracle graph + oracle mission solver | 69.10% | 73.60% |
| Complete CV/NLP/solver | 68.57% | 73.10% |
| CV scene exact | 85.67% | 85.67% |

The candidate artifact supplied in the ZIP used scikit-learn 1.9.1. It was
retrained using the local 1.7.2 installation on train only; validation reproduced
73.60%. There were 41 passing integration tests before diagnostic additions;
three additional synthetic graph tests verify fallback reason classification.

Both test submissions contain 12,000 integers in 0..3. The candidate version
changed 1,886 predictions while preserving the old file. The user reported
52.39% public accuracy for the old version; no public score for the candidate
version has been reported here.

The test audit reproduced 1,051 fallback observations over 112 scenes with
identical predictions. All have unreachable mission routes on the predicted
graph. The resolver also replaced 245 goals, including 96 scenes where the
parser returned no goal. See `FALLBACK_AUDIT.md` for the complete statistics;
these cannot all be attributed to CV. Fallback chooses the
robot's heading if legal, otherwise the lowest numbered legal action.

Local comparison layout:

```text
results/solver_comparison_20261008/
  README.md
  CV_IMPROVEMENT_PLAN.md
  v1_submitted/          old submission, solver artifact and source snapshot
  v2_candidate/         new submission, solver artifact and validation report
  shared_cv_nlp/        shared CV artifacts and progress notes
  v2_fallback_audit/    completed rerun and diagnostic output
```

Datasets, model binaries, test submissions and local results remain outside Git.
Code and documentation are versioned. The old solver artifact requires its old
34-profile source snapshot, while the candidate artifact uses 48 profiles.

The strongest remaining CV issue is print-style node detection: node-set
74.32%, edge-set 75.68%, scene-exact 66.22%. Print is not scarce in train
(493/2,000 scenes). Detector augmentation currently changes brightness/color;
blur/JPEG augmentation, input resolution and robust grid fitting are the next
controlled experiments. Test contents must not be used to train or add rules.
