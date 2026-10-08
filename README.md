# Phenikaa Campus Courier 2026

Clean-room implementation split into three independent modules:

1. `CV`: map image to `SceneGraph`.
2. `NLP`: Vietnamese mission text to structured mission fields.
3. `Strategy/Solver`: graph + structured mission + robot ID to the first action.

The package layout reserves an independent namespace for each workstream:

```text
src/courier/
  common/       # stable contracts and dataset I/O; shared, change carefully
  solver/       # graph search and learned robot strategy
  cv/           # owned by the CV workstream
  nlp/          # owned by the NLP workstream
scripts/
  solver/       # solver-only entry points
  cv/           # CV-only entry points
  nlp/          # NLP-only entry points
tests/
  solver/       # solver-only tests
  cv/           # CV-only tests
  nlp/          # NLP-only tests
artifacts/
  solver/       # trained solver models
  cv/           # trained CV models
```

The current implementation includes a complete learned CV pipeline, a Vietnamese
mission parser and map resolver, and a candidate-action strategy ranker. It can
evaluate validation and generate test submissions from images and mission text.

Current validation on 300 scenes: oracle-input solver 73.60%; complete
CV -> NLP -> solver 73.10%; CV scene exact 85.67%. These are development
validation scores. The previously submitted version scored 52.39% on the public
leaderboard. The user-reported score for the candidate submission is 55.67%,
with 49.44% accuracy for its weakest robot.

## Environment

```powershell
py -3.12 -m pip install -e ".[solver]"
$env:PYTHONPATH = "src"
```

`courier.common.Scene` is the output contract for CV.
`courier.common.Mission` is the output contract for NLP.
The solver consumes only those contracts. Workstreams should avoid editing
`courier.common` without coordinating the schema change; everything beneath
their own package and script/test directories is independently owned.

## Commands

```powershell
# Graph correctness tests
py -3.12 -m unittest discover -s tests -v

# Uniform shortest-route baseline
py -3.12 scripts/solver/evaluate_oracle_solver.py --split validation

# Train the candidate-action strategy and create its artifact
$env:PYTHONPATH = "src;scripts/solver"
py -3.12 scripts/solver/train_candidate_strategy.py --out artifacts/solver/candidate_strategy.joblib

# Re-evaluate the saved artifact (73.60% oracle-input validation)
py -3.12 scripts/solver/evaluate_strategy_model.py --split validation

# CV: per-stage accuracy; each stage (--legend/--weather/--grid/--edges/--nodes)
# can be switched independently, the rest stay at ground truth
py -3.12 -m pip install -e ".[cv]"
py -3.12 scripts/cv/evaluate_cv.py --split validation

# Train the first learned CV stage, then evaluate it with all other stages held at oracle
py -3.12 scripts/cv/train_weather.py
py -3.12 scripts/cv/evaluate_cv.py --split validation --weather learned

# NLP field accuracy with ground-truth landmarks, plus downstream solver parity
py -3.12 scripts/nlp/evaluate_nlp.py --split validation --solver

# Parse test missions into map-independent goal/via specs
py -3.12 scripts/nlp/parse_missions.py --split test

# Evaluate the complete pipeline
py -3.12 scripts/evaluate_system.py --detector-device cuda

# Generate a test submission; a new timestamped results folder is created
py -3.12 scripts/create_submission.py --detector-device cuda

# Collect a fallback audit in a separate run folder
py -3.12 scripts/create_submission.py --detector-device cuda `
  --out results/fallback_audit/predictions.json `
  --diagnostics-dir results/fallback_audit/diagnostics
```

`courier.nlp.MissionParser` converts text into map-independent `TargetSpec`
objects and boolean urgency/fragility flags. `courier.nlp.resolve` grounds those
specs against landmarks from CV and returns the `courier.common.Mission` used by
the solver. Validation goal, via, spatial-reference, urgency, and fragility
accuracy is 100% on the supplied 300 annotated scenes.

`courier.cv.CVPipeline` turns an image into `courier.cv.SceneGraph`;
`SceneGraph.to_scene(scene_id, mission)` joins it with the NLP output into the
`courier.common.Scene` the solver consumes.

The trained strategy layer masks illegal moves after classification, so it
cannot enter closed roads, violate one-way roads, or use stairs with a robot
other than robot 4.

Submission and candidate training commands refuse to overwrite existing output
files. Models, datasets and local comparison results are ignored by Git. Local
versions are kept separately under `results/solver_comparison_20261008`.
