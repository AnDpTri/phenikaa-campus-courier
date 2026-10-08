"""Train the candidate-level strategy (one binary ranker per robot over legal actions).

Reuses the scene feature cache of train_oracle_strategy.py. Hyperparameters are
fixed in advance; validation is only reported. `--fit-on train+validation`
refits on both splits for the final submission artifact (its validation score is
then no longer a held-out estimate).
"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from courier.common import load_dataset
from courier.solver.candidates import (
    CandidateStrategyModel,
    best_per_scene,
    candidate_dataset,
    candidate_feature_names,
)
from train_oracle_strategy import dataset_digest, load_or_build_features


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=40,
        random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--out", type=Path, help="New artifact path; existing files are never overwritten")
    parser.add_argument("--cache", type=Path, default=Path("data/solver/oracle_features.npz"))
    parser.add_argument("--fit-on", choices=("train", "train+validation"), default="train")
    args = parser.parse_args()
    if args.out is None:
        run_name = "candidate_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = Path("results") / run_name / "candidate_strategy.joblib"
    if args.out.exists():
        parser.error(f"artifact already exists: {args.out}; choose a new --out path")

    train = load_dataset(args.data, "train")
    validation = load_dataset(args.data, "validation")
    train_normal, train_legged, validation_normal, validation_legged = load_or_build_features(
        args.cache, train.scenes, validation.scenes, args.data
    )
    y_train = np.asarray(train.labels, dtype=np.int8).reshape(-1, 10)
    y_validation = np.asarray(validation.labels, dtype=np.int8).reshape(-1, 10)

    models, scores = [], []
    for robot in range(10):
        fit_x = train_legged if robot == 4 else train_normal
        held_x = validation_legged if robot == 4 else validation_normal
        fit_y = y_train[:, robot]
        if args.fit_on == "train+validation":
            fit_x = np.concatenate([fit_x, held_x])
            fit_y = np.concatenate([fit_y, y_validation[:, robot]])
        x, y, _, _ = candidate_dataset(fit_x, fit_y)
        model = make_model(20261008 + robot).fit(x, y)
        vx, _, groups, actions = candidate_dataset(held_x, None)
        predicted = best_per_scene(model.predict_proba(vx)[:, 1], groups, actions, len(held_x))
        score = float((predicted == y_validation[:, robot]).mean())
        models.append(model)
        scores.append(score)
        print(f"R{robot}: validation {score:.4f}", flush=True)
    macro = float(np.mean(scores))
    print(f"validation macro={macro:.4f} (fit on {args.fit_on})")

    artifact = {
        "kind": CandidateStrategyModel.ARTIFACT_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": dataset_digest(args.data),
        "feature_names": candidate_feature_names(),
        "fit_on": args.fit_on,
        "models": models,
        "validation_per_robot": scores,
        "validation_macro": macro,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(args.out)
    print(f"saved candidate strategy to {args.out}")


if __name__ == "__main__":
    main()
