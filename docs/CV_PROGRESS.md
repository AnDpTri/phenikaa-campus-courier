# CV implementation progress

Last updated: 2026-10-08 (Asia/Saigon)

## Scope and ownership

The CV workstream owns `src/courier/cv`, `scripts/cv`, and `tests/cv`. It does
not modify the solver or NLP implementations. The target contract is
`image -> SceneGraph`, with legend, weather, grid, edges, landmarks, robot
position, and robot heading all recovered from the image.

## Completed: weather classifier

Implemented a lightweight artifact-backed classifier using HOG, HSV
histograms, a coarse RGB thumbnail, and `LinearSVC`:

- source: `src/courier/cv/features.py`, `src/courier/cv/learned.py`;
- training: `scripts/cv/train_weather.py`;
- ablation evaluation: `scripts/cv/evaluate_cv.py --weather learned`;
- artifact: `artifacts/cv/weather_classifier.joblib` (ignored, reproducible);
- artifact size: about 45 KB;
- unit tests: 9/9 CV tests pass.

Results:

| Evaluation path | Accuracy |
|---|---:|
| Pre-extracted validation weather crops | 299/300 = 99.67% |
| Integrated pipeline on original validation images, oracle legend box | 300/300 = 100% |

The integrated run processed 300 scenes with zero invalid graphs at about
0.016 seconds per scene. Other stages were held at oracle to isolate weather.

Committed and pushed as `6b90a38` (`Add learned weather CV stage`).

## In progress: node classifier

Available crop data:

- train: 86,210 node crops;
- validation: 16,142 node crops;
- labels: empty, ten landmark types, and robot headings UP/DOWN/LEFT/RIGHT.

A global HOG/colour/pixel `LinearSVC` baseline was evaluated on a sampled
22,000-crop train set and 7,604 validation crops:

| C | Overall | Empty/object binary | Non-empty class | Robot heading |
|---:|---:|---:|---:|---:|
| 0.003 | 75.58% | 83.85% | 54.38% | 41.00% |
| 0.010 | 79.17% | 87.41% | 61.82% | 58.00% |
| 0.030 | 82.15% | 90.29% | 67.98% | 67.00% |

This baseline is intentionally not integrated: its aggregate score is inflated
by empty nodes and its landmark/heading accuracy is too weak. The next design
uses the legend from the same image as a style-specific reference:

1. detect empty versus occupied nodes;
2. match occupied landmark patches against `place:*` legend swatches;
3. detect the robot separately;
4. classify robot heading after rotation-normalized cropping.

This should generalize better because the four drawing styles and symbol
appearance are defined by each image's own legend.

## Remaining roadmap

1. Finish legend-conditioned node matching and artifact-backed inference.
2. Train edge existence/status/stairs/one-way classifiers on 142,062 train and
   26,507 validation edge crops, using road legend swatches for status mapping.
3. Implement grid shape/node-centre detection from the original image.
4. Implement legend layout detection and weather-box localization.
5. Add structural repair/fallbacks so predicted graphs always satisfy
   `validate_graph` and solver legality requirements.
6. Run end-to-end validation by style and degradation, then optimize the
   highest-impact error categories.
7. Add test-split inference and deterministic cache/artifact loading.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
py -3.12 scripts/cv/train_weather.py
py -3.12 scripts/cv/evaluate_cv.py --split validation --weather learned
py -3.12 -m unittest discover -s tests/cv -v
```
