# Devil May Cry — Phenikaa Campus Courier AI Hackathon 2026

**A multimodal, graph-based AI system built under a ~48-hour hackathon deadline.**

[![Result](https://img.shields.io/badge/Result-Top%2013%20%2F%2025-24292f?style=flat-square)](https://github.com/AnDpTri/Top13-Phenikaa-AI-Hackathon-2026)
[![Peak leaderboard score](https://img.shields.io/badge/Peak%20score-71.17%25-216e39?style=flat-square)](https://github.com/AnDpTri/Top13-Phenikaa-AI-Hackathon-2026)
[![Challenge](https://img.shields.io/badge/Challenge-COURIER2-0969da?style=flat-square)](https://aihackathon.phenikaa-uni.edu.vn/)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square)](https://www.python.org/)

**Team:** Devil May Cry (DMC)  
**Challenge:** Phenikaa Campus Courier 2026 — `COURIER2` (new private-test dataset)  
**Reported result:** Top 13 out of 25 teams; **71.17% peak leaderboard macro accuracy**  
**Development window:** October 8–10, 2026 (~48 hours)

> **Project context.** This repository documents a hackathon solution developed from an empty project under severe time constraints. One developer coordinated the architecture, experiments, integration, validation, and submissions, with AI coding agents assisting in implementation and research. The result is a working experimental pipeline, **not a production-certified robotics system**.

## The challenge

For every delivery scene, the system receives:

- A **raster campus map** with a partially irregular road grid, landmarks, traffic conditions, stairs, one-way streets, road closures, a robot's position/heading, and weather. Images may be rotated, blurred, JPEG-compressed, or drawn in different styles; the road legend may be remapped per image.
- A **Vietnamese delivery instruction** that may be noisy or ambiguous: missing diacritics, typos, corrections, negations, indirect place descriptions, spatial references, urgent/fragile modifiers, and an optional intermediate stop (`via`).
- A **robot ID** from 0 to 9. Each robot follows its own fixed, hidden movement strategy; their tie-breaking rules are also unknown.

The objective is to predict **the robot's first movement**, encoded as `0 = UP`, `1 = DOWN`, `2 = LEFT`, or `3 = RIGHT`.

Each scene contains ten observations, one for each robot. The test submission contains **12,000 action predictions for 1,200 scenes**. The leaderboard metric is **macro accuracy across the ten robots**. Training contains 2,000 scenes and validation contains 300 scenes. The new `COURIER2` dataset differs from the earlier public challenge and has more complex robot policies and input distributions.

See the [challenge specification](Phenikaa_Campus_Courier_2026_v3/DE_BAI.md) for the full input schema, evaluation protocol, and restrictions.

## System architecture

```text
                     Campus map image
                            |
                  +---------v----------+
                  | Computer Vision     |
                  | heatmaps + CNNs     |
                  | grid reconstruction |
                  +---------+----------+
                            |
                        SceneGraph
                            |
Vietnamese instruction      |         Robot ID (0..9)
          |                 |                 |
 +--------v---------+       |                 |
 | Hybrid NLP       |       |                 |
 | rules + BiGRU    |       |                 |
 | confidence gates |       |                 |
 +--------+---------+       |                 |
          |                 |                 |
          +---- Map-aware --+                 |
                grounding                    |
                   |                         |
                 Mission                     |
                   |                         |
                   +-----------+-------------+
                               |
                  +------------v-------------+
                  | Legal directed graph     |
                  | 48 route-cost profiles   |
                  | Candidate-action ranker  |
                  | 10 robot-specific models |
                  | Graph-repair fallback    |
                  +------------+-------------+
                               |
                     First action (0..3)
```

The key design decision is to separate **perception**, **semantic parsing**, and **behavior inference** with explicit intermediate representations (`SceneGraph`, `Mission`, `Scene`). This makes errors measurable by stage, although errors can still propagate through the pipeline.

### 1. Computer vision — image to `SceneGraph`

**Implementation:** [`src/courier/cv/`](src/courier/cv/)

- A compact **CenterNet-style detector** produces heatmaps for road-grid intersections, semantic legend swatches, and the weather-icon region from a letterboxed 512-pixel image.
- A geometry-based **grid-fitting step** estimates map rotation and clusters detected node centers into rows and columns.
- **`EdgeNet`** classifies crops between adjacent nodes: road appearance, stairs, and one-way direction.
- **`NodeNet`** classifies node crops into empty nodes, landmark types, and the robot's heading.
- Road appearances are mapped back to their actual meanings by reading **the legend of each individual image**, rather than assuming fixed colors.
- A lightweight **`LinearSVC` weather classifier** operates on extracted image features. The main detector can use CUDA; the crop classifiers run on CPU in the supplied inference pipeline.

The recorded fully learned CV benchmark on the **300-scene validation split** reports **85.67% scene-exact reconstruction (257/300)**. Scene-exact means the complete discrete graph, landmarks, robot state, and weather match the annotations—not merely that individual crops were classified accurately.

The weakest visual style was `print` (66.22% scene-exact in the base experiment). A 768-pixel/style-routed experiment improved the validation CV score, but its public submission was worse, so it was not retained in the documented stronger public configurations.

Details: [`docs/CV_PROGRESS.md`](docs/CV_PROGRESS.md).

### 2. Vietnamese NLP — instruction to `Mission`

**Implementation:** [`src/courier/nlp/`](src/courier/nlp/)

The NLP stack deliberately combines two different approaches:

1. A **rule-based parser** using domain lexicons, token normalization, typo handling, clause analysis, negation/correction cues, spatial expressions, and intermediate-stop detection.
2. A **multi-task neural parser** with hashed word/character-trigram embeddings, a two-layer bidirectional GRU, and separate attention/classification heads for the goal, via, spatial anchors, urgency, and fragility.

A **confidence-gated hybrid** (`HybridMissionParser`) uses the rule parser as the starting point. The neural reading fills in a goal whenever the rules find none; it replaces a rule goal, via, or flag only when its confidence reaches the corresponding threshold (`goal`, `via`, `flags`). A threshold of `2.0` disables that override, which is how the urgent/fragile flags were left to the rules in the stronger configurations. Ensembles average the softmax outputs of several networks.

**Map-aware grounding** resolves relative descriptions such as the northernmost landmark or a location nearest an anchor against the landmarks extracted by CV. When the chosen goal type does not exist on the predicted map, the parser falls back to the neural reading, then to the most probable neural goal type that is present.

Synthetic Vietnamese examples, real annotated training scenes, held-out phrasing, and multiple model seeds were used to develop and evaluate NLP variants. The repository contains the generators and training scripts; several final NLP checkpoints are distributed via [GitHub Releases](https://github.com/AnDpTri/Top13-Phenikaa-AI-Hackathon-2026/releases/tag/v1.0.0).

A particularly consequential experiment compared the same V5 model with neural `via` overrides disabled versus enabled (`via` confidence threshold `0.95`). The recorded public scores changed from **65.28% to 70.39%**, with **1,148 of 12,000 output actions** changing. This is a configuration comparison, not proof that every individual neural correction was right.

Details: [`docs/NLP_NEURAL.md`](docs/NLP_NEURAL.md), [`docs/NLP_V4_CURATED200K_R1_REPORT_20261009.md`](docs/NLP_V4_CURATED200K_R1_REPORT_20261009.md), and [`docs/REPORT_SUBMISSION_1182_20261010.md`](docs/REPORT_SUBMISSION_1182_20261010.md).

### 3. Multi-robot solver — hidden behavior to first action

**Implementation:** [`src/courier/solver/`](src/courier/solver/)

The solver first builds a **legal, directed road graph**:

- Closed roads are unavailable.
- One-way directions are enforced.
- Only robot `R4` may traverse stairs.
- When a mission specifies a `via`, the route must visit it before the final `goal`.

A configurable shortest-path engine then evaluates legal first actions under **48 route-cost hypotheses**, covering step count, crowded roads, covered roads, turning, U-turns, stairs, and ten combinations fitted per robot by a coordinate search on the training split (`scripts/solver/fit_cost_weights.py`). Pathfinding tracks both **heading** and **whether the via has been reached**, so route cost depends on more than the current node.

Rather than predicting an absolute direction with a single four-class model, the final source implementation constructs **candidate-level features** for every legal first move: local road characteristics, turns, distance to the next waypoint, and **cost/regret/rank/optimality** under each profile. Each candidate row has 219 features. **One histogram gradient-boosting binary classifier per robot** scores the rows and the highest-scoring legal move is chosen. The two tracked artifacts (`candidate_strategy.joblib`, `candidate_strategy_trainval.joblib`) hold one classifier per robot; the "Ultra" artifact used for the last two submissions replaces each with a soft-voting ensemble of five seeds (see [Final submissions](#final-submissions-1201-and-1208)).

On validation using **ground-truth maps and missions** (isolating the strategy stage):

| Solver configuration | Macro accuracy |
| --- | ---: |
| Uniform shortest-path baseline | 55.70% |
| Earlier scene-level strategy model | 69.10% |
| 48-profile candidate-action ranker | **73.60%** |

These are **oracle-input solver results**, not public-test or fully end-to-end scores. Models subsequently fitted on both train and validation cannot be evaluated on that same validation split as an independent holdout.

#### Graph repair and graceful degradation

An early test diagnostic found **112/1,200 scenes** where the parsed mission could not be completed on the predicted graph, producing **1,051 robot-level fallback decisions**. Structural graph validation alone did not reveal these semantic reachability failures.

The repair module tries progressively less conservative relaxations: ignore one-way constraints, add missing grid-adjacent edges, reopen closed roads, and reconstruct missing grid nodes. It then attempts normal candidate ranking again, with a greedy fallback if needed.

On deliberately damaged validation graphs, the documented unreachable-case accuracy rose from **35.09% to 53.20%**. The isolated public repair submission improved from **55.67% to 57.56%**.

**Known limitation:** repaired-graph candidates are not always revalidated against the original inferred graph's legal first actions. Repair is a pragmatic hackathon fallback, not a safety guarantee for physical navigation.

Details: [`docs/SOLVER_REPORT.md`](docs/SOLVER_REPORT.md), [`docs/FALLBACK_AUDIT.md`](docs/FALLBACK_AUDIT.md), and [`docs/SOLVER_REPAIR.md`](docs/SOLVER_REPAIR.md).

## Results and the 48-hour iteration

The project started without an implemented solution. **52.39% was the first successful COURIER2 submission—not the starting accuracy of a pre-existing model.** The team then iterated through solver changes, graph repair, neural NLP, threshold calibration, and final model combinations.

| Date (2026, ICT) | Submission / experiment | Public macro accuracy |
| --- | --- | ---: |
| Oct 8, 10:25 | First end-to-end submission (`#1027`) | 52.39% |
| Oct 8, 12:57 | Candidate-action strategy ranker (`#1045`) | 55.67% |
| Oct 8, 15:17 | Graph repair (`#1057`) | 57.56% |
| Oct 8, 16:01 | Neural NLP V1 + repair (`#1062`) | 61.28% |
| Oct 9, 01:18 | NLP V2 + train/validation solver (`#1102`) | 64.78% |
| Oct 9, 09:08 | NLP V2 with neural via overrides (`#1107`) | 68.78% |
| Oct 10, 08:24 | V5, via overrides disabled (`#1180`) | 65.28% |
| Oct 10, 08:54 | V5, via overrides enabled (`#1182`) | 70.39% |
| Oct 10, 11:50 | V5 + five-seed "Ultra" solver (`#1201`) | 70.83% |
| Oct 10, 12:00 | V5+V9 ensemble, goal threshold 0.70 + Ultra solver (`#1208`) | **71.17%** |

**Reported final standing: 13th out of 25 teams.** The competition uses a separate hidden final evaluation; public leaderboard scores and final placement should not be treated as the same metric. Submission identifiers and scores up to `#1182` are documented in the reports under [`docs/`](docs/). The scores and times of `#1201` and `#1208` come from leaderboard screenshots; the configuration behind each is taken from the development session's command log, as described in [Final submissions](#final-submissions-1201-and-1208).

Two lessons from the submission history are worth highlighting:

- **Offline validation was not a perfect proxy for public test.** A CV routing experiment improved validation but reduced the public score; by contrast, changing NLP behavior helped on public scenes whose wording differed from the development data.
- **Operational correctness matters.** The team audited fallback rates, hashes, file differences, and even an accidental submission to the earlier `COURIER` challenge (a reported 25.44% score that is **not** a valid `COURIER2` result).

## Repository layout

```text
src/courier/
  common/                Domain types and dataset I/O
  cv/                    Detector, graph reconstruction, image classifiers
  nlp/                   Rule parser, neural parser, synthetic text, grounding
  solver/                Directed graph search, candidate ranker, repair
scripts/
  cv/                    CV training and diagnostics
  nlp/                   NLP training, evaluation, and corpus utilities
  solver/                Strategy training and evaluation
  create_submission.py   End-to-end test prediction generator
  evaluate_system.py     Four-way oracle/CV/NLP/full-system evaluation
  compare_submissions.py Submission integrity and disagreement checks
tests/                    Unit and integration tests
docs/                     Experiment reports, handoffs, and audits
artifacts/cv/             Tracked CV model checkpoints
artifacts/solver/         Tracked baseline and train+validation solver artifacts
results/                  Committed experiment snapshots and predictions
Phenikaa_Campus_Courier_2026_v3/
  DE_BAI.md               Challenge specification
```

The **original competition dataset is not bundled** in this repository. Some tests require that dataset; without it they are skipped or fail during setup. Final NLP checkpoints and an additional solver checkpoint are distributed in the GitHub Release instead of under the tracked `artifacts/nlp/` directory.

## Installation and reproducibility

### Requirements

- **Python 3.12** and `pip`.
- Dependencies defined in [`pyproject.toml`](pyproject.toml), including PyTorch, scikit-learn, OpenCV, Pillow, NumPy, and joblib.
- The **COURIER2 dataset**, provided separately, at `Phenikaa_Campus_Courier_2026_v3/delivery_public/`, containing the appropriate `train`, `validation`, and `test` splits.
- Optional NVIDIA CUDA GPU for the full-image detector. CPU inference is also supported.
- Optional [GitHub CLI](https://cli.github.com/) (`gh`) to download release checkpoints.

The commands below use **PowerShell** to match the project's development environment.

```powershell
git clone https://github.com/AnDpTri/Top13-Phenikaa-AI-Hackathon-2026.git
cd Top13-Phenikaa-AI-Hackathon-2026
py -3.12 -m pip install -e ".[solver,cv]"
$env:PYTHONPATH = "src"
```

### Download the NLP model

The CV checkpoint and `candidate_strategy_trainval.joblib` are tracked in the source repo. The documented `#1182` configuration needs this additional NLP release asset:

```powershell
New-Item -ItemType Directory -Force artifacts/nlp | Out-Null
gh release download v1.0.0 `
  --repo AnDpTri/Top13-Phenikaa-AI-Hackathon-2026 `
  --pattern "neural_parser_v5_sf200_scratch_via095.pt" `
  --dir artifacts/nlp
```

**Important:** `gh release download --dir artifacts` on its own does **not** place checkpoint files into the required `artifacts/nlp/` and `artifacts/solver/` subdirectories. Download to the correct directories explicitly.

### Generate the documented `#1182` configuration

After providing the COURIER2 dataset at the default path:

```powershell
py -3.12 scripts/create_submission.py `
  --detector-device auto `
  --nlp-model artifacts/nlp/neural_parser_v5_sf200_scratch_via095.pt `
  --nlp-goal-threshold 0.8 `
  --strategy-artifact artifacts/solver/candidate_strategy_trainval.joblib `
  --out results/reproduction_1182/predictions.json `
  --diagnostics-dir results/reproduction_1182/diagnostics
```

The NLP checkpoint stores the `via=0.95` hybrid threshold. The historical `#1182` submission report records the prediction-file SHA-256:

```text
1B400996A598AA21BEE3B727C563288CF86E0B4AFA53FC9BC28BFD7B785B2E9B
```

To check your output:

```powershell
(Get-FileHash results/reproduction_1182/predictions.json -Algorithm SHA256).Hash
```

A matching hash confirms byte-identical output to the reported `#1182` prediction file; a nonmatching hash requires checking the dataset, model versions, runtime environment, and thresholds. The script refuses to overwrite an existing output or diagnostics directory, so use a fresh path on subsequent runs.

### Final submissions (`#1201` and `#1208`)

Both use the Ultra solver artifact from the release. Download it and the V5+V9 NLP ensemble:

```powershell
gh release download v1.0.0 `
  --repo AnDpTri/Top13-Phenikaa-AI-Hackathon-2026 `
  --pattern "candidate_strategy_ultra.joblib" `
  --dir artifacts/solver
gh release download v1.0.0 `
  --repo AnDpTri/Top13-Phenikaa-AI-Hackathon-2026 `
  --pattern "neural_parser_v5_v9_ensemble_g070_v095.pt" `
  --dir artifacts/nlp
```

Configuration recorded for `#1208` (71.17%):

```powershell
py -3.12 scripts/create_submission.py `
  --detector-device auto `
  --nlp-model artifacts/nlp/neural_parser_v5_v9_ensemble_g070_v095.pt `
  --nlp-goal-threshold 0.70 `
  --strategy-artifact artifacts/solver/candidate_strategy_ultra.joblib `
  --out results/reproduction_1208/predictions.json `
  --diagnostics-dir results/reproduction_1208/diagnostics
```

Expected SHA-256 of the prediction file:

```text
6FDF71235BD08ACEF371A65DD464B8FA426C0D5AE211642572330FC5D2DF64F8
```

`#1201` (70.83%) is the same command with `--nlp-model artifacts/nlp/neural_parser_v5_sf200_scratch_via095.pt` and without `--nlp-goal-threshold`; its recorded prediction file has SHA-256 `097B9390AFA19F788DE74E3FC69C90FCF3733DD06E776EF7C3E1A1B98C28AE75`.

What has and has not been verified:

- The `#1208` command was re-run after the competition from a clean copy of the source with the released artifacts, on Windows 11 / Python 3.12.10 / PyTorch 2.14.1 (CUDA) / scikit-learn 1.7.2. It produced a file byte-identical to the one generated during the competition with that configuration (the hash above). The `#1201` command has not been re-run.
- The prediction files for these two runs were generated at 11:24 and 11:49 on Oct 10 and are not tracked in this repository. The session log identifies the 11:49 file as `#1201`. `#1208` was submitted at 12:00 and attributed in the log to the V5+V9 NLP ensemble; the 11:24 file is the only Ultra + V5+V9 output that existed at that time, but the upload itself was not logged, so this mapping is an inference.
- The V5+V9 artifact is the three V5 seeds plus one "V9" network fine-tuned from V5 seed 0. Merging those four checkpoints with `scripts/nlp/merge_checkpoints.py` reproduces the released weights exactly.
- The Ultra solver was trained by an ad-hoc command: five `HistGradientBoostingClassifier` seeds per robot (`learning_rate=0.04`, `max_iter=350`, `max_leaf_nodes=31`, `l2_regularization=0.5`, `min_samples_leaf=20`) in a soft `VotingClassifier`, fitted on train + validation. With train-only fitting it scored 74.20% oracle-input macro accuracy on validation, against 73.60% for the single-seed ranker. That training script is not checked in to this repository.

The release also contains `neural_parser_v5_v9_assault_g070_v095_f90.pt`, the same four networks with the flag threshold lowered to 0.90. The tracked diagnostics for `submission_total_assault_f90_20261010` reference `candidate_strategy_trainval.joblib`, not the Ultra solver.

### Run tests and evaluate the system

```powershell
py -3.12 -m unittest discover -s tests -v

# Requires the annotated COURIER2 validation split.
py -3.12 scripts/evaluate_system.py `
  --split validation `
  --strategy-artifact artifacts/solver/candidate_strategy.joblib
```

`evaluate_system.py` reports oracle, NLP-only, CV-only, and fully learned performance. For an **independent solver validation figure**, use a model trained on `train` only; `candidate_strategy_trainval.joblib` was refitted with validation annotations.

### Competition rules and data use

The competition specification prohibits training, manual labeling, or rule construction using test content. Training scripts in this repository use the annotated training data, synthetic examples, and validation for model/threshold selection; inference reads the test split to produce predictions. See the [original rules](Phenikaa_Campus_Courier_2026_v3/DE_BAI.md) before adapting the pipeline for another submission. External AI model APIs are not used by the submitted inference pipeline.

## Engineering trade-offs and known limitations

This is a hackathon snapshot, and several limitations are explicit:

- **Error propagation:** an incorrect grid, map landmark, semantic goal, or via changes the routing problem for all ten robots in a scene.
- **Distribution shift:** validation and public test can favor different configurations; test labels are unavailable for detailed attribution.
- **Model approximation:** the candidate ranker learns from many route-cost hypotheses but does not prove that the ten hidden policies have been exactly recovered.
- **Graph repair:** fallback relaxations help on damaged graphs but do not constitute verified, physical-robot-safe navigation; the final action should be rechecked against the original inferred legal action set in a future revision.
- **Reproducibility:** inference for the final configuration is reproducible from the released checkpoints (see above), but not every historical training run has a complete end-to-end procedure in this repository. In particular the Ultra solver's training command is not checked in, and the V5 NLP training scripts under `results/` use absolute paths from the development machine.
- **Dataset availability:** dataset-dependent integration tests require locally supplied competition data.

## Development and acknowledgments

The system was developed as a **one-person-led, AI-assisted engineering effort** over approximately 48 hours. AI coding agents assisted with source implementation, exploratory experiments, and technical reports. The human project lead owned the architecture, workstream coordination, integration decisions, experiment selection, submission management, and final result.

The documentation intentionally preserves failed experiments and handoff notes alongside successful results. They are part of the engineering record, not just presentation material.

## License

**No license file is currently included in this repository.** Until a license is explicitly added, do not assume the source or distributed model artifacts are available under MIT or another open-source license. Contact the repository owner regarding reuse permissions.
