# Print CV experiment — 2026-10-08

## Decision

The 768px detector materially improves CV, especially print, but is not promoted to the default pipeline yet because complete-system validation decreased by 0.23 percentage points. The baseline artifact remains unchanged.

## Configuration

- DetectorNet width: 24 (641,902 parameters)
- Input: 768x768, output stride 4
- Seed: 0
- Requested epochs: 15
- Stopped after epoch 10; best checkpoint: epoch 8
- Early-stop decision: two epochs without an improvement of at least 0.0005
- Batch: 4, CUDA AMP on RTX 3050 Laptop GPU
- JPEG augmentation: probability 0.65, quality 30–85
- Gaussian blur augmentation: probability 0.45, sigma 0.35–1.35
- Best validation focal loss: 0.0116640452
- Selected node threshold: 0.30
- Test data was not used for training or model/threshold selection.

Training validation loss by epoch:

| Epoch | Loss |
| ---: | ---: |
| 1 | 0.2631 |
| 2 | 0.0540 |
| 3 | 0.0523 |
| 4 | 0.0316 |
| 5 | 0.0235 |
| 6 | 0.0151 |
| 7 | 0.0131 |
| 8 | **0.0117** |
| 9 | 0.3099 |
| 10 | 0.0118 |

The epoch-9 spike did not overwrite the best checkpoint.

## CV validation

| Metric | Baseline 512 | Candidate 768 / threshold 0.30 | Change |
| --- | ---: | ---: | ---: |
| Scene exact | 85.67% (257/300) | **90.00% (270/300)** | +4.33 pp |
| Grid shape | 97.67% | **98.33%** | +0.66 pp |
| Node set | 90.00% | **94.67%** | +4.67 pp |
| Edge set | 90.67% | **95.00%** | +4.33 pp |
| Landmark set | 98.00% | 97.33% | -0.67 pp |
| Robot position | 99.00% | **99.33%** | +0.33 pp |
| Robot heading | 95.67% | **96.00%** | +0.33 pp |
| Weather | 98.67% | **100.00%** | +1.33 pp |
| Invalid graphs | 0 | 0 | unchanged |

Scene exact by style:

| Style | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Classic | 97.26% | 94.52% | -2.74 pp |
| Night | 86.36% | **92.05%** | +5.69 pp |
| Print | 66.22% (49/74) | **85.14% (63/74)** | **+18.92 pp** |
| Sketch | 93.85% | 87.69% | -6.16 pp |

Print-specific exact metrics:

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Grid shape | 93.24% | **98.65%** | +5.41 pp |
| Node set | 74.32% | **91.89%** | +17.57 pp |
| Edge set | 75.68% | **90.54%** | +14.86 pp |
| Robot heading | 91.89% | **95.95%** | +4.06 pp |
| Scene exact | 66.22% | **85.14%** | +18.92 pp |

Threshold 0.40 was also fully evaluated. It produced 89.00% overall scene exact and 78.38% print scene exact, so it was rejected. The full threshold sweep is in `thresholds_validation.json`.

## Complete-system validation

The candidate was evaluated with the same NLP and 48-profile candidate solver.

| Scenario | Macro action accuracy |
| --- | ---: |
| Oracle graph + oracle mission | 73.60% |
| Baseline full pipeline | **73.10%** |
| Candidate CV + oracle mission | 72.83% |
| Candidate full pipeline | **72.87%** |

Although scene exact improved, full-pipeline action accuracy decreased by 0.23 percentage points (about 7 of 3,000 validation actions). Therefore this experiment is a CV improvement but not yet a submission improvement.

## Files

- `detector.pt`: candidate checkpoint; does not replace `artifacts/cv/detector.pt`.
- `thresholds_validation.json`: one-pass node threshold sweep.
- `cv_validation_threshold_030.json`: selected full-CV report.
- `cv_validation_threshold_040.json`: rejected threshold report.
- `system_validation_threshold_030.json`: end-to-end report.

## Recommended next experiment

Preserve the candidate and investigate a validation-independent per-scene lattice quality selector, or a train-only style classifier that uses the lower threshold for print and a higher threshold for classic/night. Any selector must be developed on a hard split of train and checked once on validation. The candidate must not replace the baseline until end-to-end accuracy is at least 73.10% without degrading the non-print styles excessively.
