# Solver workflow

From the workspace root, set `$env:PYTHONPATH = "src"` and run with Python 3.12.

- `evaluate_oracle_solver.py`: shortest-path baseline on ground-truth graph and mission.
- `search_cost_profile.py`: diagnostic search of a single cost coefficient.
- `fit_cost_weights.py`: coordinate search for additive route-cost weights, optionally by mission condition.
- `train_oracle_strategy.py`: train the legacy scene-level strategy models.
- `train_candidate_strategy.py`: train one candidate-action ranker per robot.
- `evaluate_strategy_model.py`: evaluate either artifact kind through `load_strategy`.

The current strategy feature library contains 48 route-cost profiles. The
candidate ranker converts every legal first action into a row containing scene
context, immediate action properties, relative turn/distance features, and the
action's regret/rank/optimality under every profile. Ten small
`HistGradientBoostingClassifier` models score the candidates, one per robot.

Current benchmark with oracle graph and oracle mission on 300 validation scenes:

| Method | Macro accuracy |
| --- | ---: |
| Uniform shortest path | 55.70% |
| Legacy scene classifier, 34 profiles | 69.10% |
| Legacy scene classifier, 48 profiles | 69.47% |
| Candidate ranker, 48 profiles | **73.60%** |

Train and evaluate the candidate model:

```powershell
$env:PYTHONPATH = "src;scripts/solver"
py -3.12 scripts/solver/train_oracle_strategy.py
py -3.12 scripts/solver/train_candidate_strategy.py
py -3.12 scripts/solver/evaluate_strategy_model.py `
  --model artifacts/solver/candidate_strategy.joblib `
  --report artifacts/solver/candidate_validation_report.json
```

For a final artifact, `train_candidate_strategy.py --fit-on train+validation`
is available, but its validation score is no longer held out. Test data must
never be used for training or rule development.

Feature caches include a schema and source/data fingerprint. Artifacts include
the dataset digest, feature schema, library versions, training split and
validation metrics. Model replacement is atomic.

The graph solver raises a descriptive error for missing targets or unreachable
missions. Both strategy implementations expose
`predict_scene_with_diagnostics`; missing or unreachable targets trigger a
legal heading-first fallback with confidence zero. A graph without any legal
outgoing action remains a hard error for the pipeline to handle.
