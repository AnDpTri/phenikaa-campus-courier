# CV implementation progress

Last updated: 2026-10-08 (Asia/Saigon), learned CV pipeline completed

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

## Completed CNN pipeline (2026-10-08)

The SVM node baseline below was replaced by small CNNs. Detector training automatically
uses CUDA when available and falls back to CPU; the current crop training script
runs on CPU. Inference uses one shared
detector pass followed by crop classifiers.

### Code

| File | Role |
|---|---|
| `src/courier/cv/crops.py` | crop geometry (scaled by grid spacing) and labels, shared by training and inference |
| `src/courier/cv/nets.py` | `NodeNet` 48x48 -> empty / 10 landmarks / robot x 4 headings; `EdgeNet` 96x24 strip -> look, stairs, one-way |
| `src/courier/cv/detector.py` | `DetectorNet`: nine class-specific heatmaps at 1/4 of a 512x512 letterbox (nodes, seven semantic legend swatches, weather) plus box size |
| `src/courier/cv/grid_fit.py` | detected node centres -> (row, col): de-rotate by the dominant neighbour angle, cluster rows/columns by gaps |
| `src/courier/cv/neural.py` | `NeuralLegendReader`, `NeuralGridDetector` (one shared detector pass), `NeuralEdgeClassifier`, `NeuralNodeClassifier` |
| `scripts/cv/build_crops.py` | node/edge/swatch crops -> `data/cv/crops_<split>.npz` (train crops jittered 2 px, scale +-8%) |
| `scripts/cv/train_crop_nets.py` | trains `node_net.pt` and `edge_net.pt` |
| `scripts/cv/train_detector.py` | caches letterboxed images/heatmaps, trains `detector.pt` |
| `scripts/cv/build_weather_crops.py` | exports weather crops in the layout `train_weather.py` reads |
| `scripts/cv/evaluate_cv.py` | new `learned` option for every stage, `--all learned` |
| `scripts/evaluate_system.py` | complete CV -> NLP -> strategy evaluation |

### Facts and corrections checked on all 2,300 scenes

- JSON stores the final seven legend entries in semantic order, but their
  visual rows are shuffled in 64/300 validation scenes. A generic swatch
  heatmap followed by y-sorting therefore fails. The final detector uses seven
  class-specific swatch heatmaps.
- Every grid row and column contains at least one node, so the grid shape is
  the number of row/column clusters. `fit_grid` on ground-truth centres
  reproduces every grid exactly (2300/2300).
- A road "look" (colour / dash pattern) depends on the style, and
  `road_look` permutes looks between statuses in some images. `EdgeNet`
  predicts the look; the same net classifies the three legend swatches and the
  best permutation maps looks to statuses.

### Isolated validation results

| Stage | Validation result |
|---|---:|
| Node CNN with oracle grid | landmarks and robot position 100%; robot heading 97.33%; scene exact 97.33% |
| Edge CNN with oracle grid/legend | edge set 99.67%; status/stairs/one-way 100% |
| Weather with oracle box | 100% (300/300) |
| Grid fit from annotated centres | 100% (2300/2300) |
| Semantic detector checkpoint | best validation focal loss 0.0250 |

### Fully learned CV validation

All five stages are learned and use only the original image:

| Metric | Result |
|---|---:|
| Scene exact | **85.67% (257/300)** |
| Invalid graphs | **0/300** |
| Grid shape | 97.67% |
| Node set | 90.00% |
| Node precision / recall | 99.83% / 99.65% |
| Edge set | 90.67% |
| Edge status | 99.58% |
| Stairs / one-way | 99.89% / 99.91% |
| Landmark set | 98.00% |
| Robot position / heading | 99.00% / 95.67% |
| Weather | 98.67% |
| Runtime | 0.204 seconds/scene |

Scene exact by style: classic 97.26%, night 86.36%, print 66.22%, sketch
93.85%. Print is the remaining high-impact CV subgroup.

### Complete-system validation

| Inputs to strategy model | Macro action accuracy |
|---|---:|
| Oracle graph + oracle mission | 69.10% |
| Oracle graph + parsed NLP mission | 69.10% |
| Learned CV graph + oracle mission | 68.73% |
| Learned CV graph + parsed NLP mission | **68.57%** |

On CV-exact scenes the complete system scores 69.49%; on CV-inexact scenes it
scores 63.02%. The complete pipeline loses only 0.53 percentage points versus
the oracle-input strategy model, so the strategy solver remains the dominant
bottleneck.

Artifacts present locally (ignored by Git, reproducible):

- `weather_classifier.joblib`;
- `node_net.pt`;
- `edge_net.pt`;
- `detector.pt`.

### Reproduction

```powershell
$env:PYTHONPATH = "src"
py -3.12 -m pip install -e ".[cv]"   # torch; opencv-python < 5 (5.0 has no HOGDescriptor)
py -3.12 scripts/cv/build_crops.py --split train
py -3.12 scripts/cv/build_crops.py --split validation
py -3.12 scripts/cv/train_crop_nets.py --epochs 2
py -3.12 scripts/cv/train_detector.py --epochs 10 --device auto
py -3.12 scripts/cv/build_weather_crops.py; py -3.12 scripts/cv/train_weather.py
py -3.12 scripts/cv/evaluate_cv.py --split validation --all learned
py -3.12 scripts/evaluate_system.py --split validation
```

## Superseded: SVM node classifier baseline


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

1. Improve print-style node detection and robot heading classification.
2. Completed: test-split submission entry point in `scripts/create_submission.py`;
   the command writes each new run separately and can export fallback diagnostics.
3. Package or download ignored artifacts for clean-machine deployment.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
py -3.12 scripts/cv/train_weather.py
py -3.12 scripts/cv/evaluate_cv.py --split validation --weather learned
py -3.12 -m unittest discover -s tests/cv -v
```
