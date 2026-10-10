# Solver workflow

From the workspace root, set `$env:PYTHONPATH = "src"` and run with Python 3.12.

- `evaluate_oracle_solver.py`: shortest-path baseline on ground-truth graph and mission.
- `search_cost_profile.py`: diagnostic search of a single cost coefficient.
- `train_oracle_strategy.py`: fit each robot independently, choose its model family
  using stratified cross-validation on train scenes, then evaluate on validation.
- `evaluate_strategy_model.py`: load the saved artifact and measure accuracy;
  `--report artifacts/solver/validation_report.json` saves metrics and latency.

The strategy features include legal immediate actions, heading, mission flags,
landmark candidates, and route cost/rank under 34 predefined utility profiles.
The artifact contains dataset SHA256, schema, library versions, CV scores,
selected families, and validation scores. Model replacement is atomic.

Validation has already been inspected during development. Its reported score
is a development benchmark, not an unbiased estimate from unseen data. Train CV
selects the model family; validation does not select among model candidates.

The graph solver raises a descriptive error for missing targets or unreachable
missions. The strategy layer exposes `predict_scene_with_diagnostics` with
confidence and a fallback reason. Missing/unreachable targets trigger a legal
heading-first fallback with confidence zero. A graph without any legal outgoing
action raises an error for the pipeline to handle. Classifier confidence is
normalized over legal actions and is not calibrated.
