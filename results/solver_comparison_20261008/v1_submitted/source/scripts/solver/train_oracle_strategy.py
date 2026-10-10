"""Train and evaluate the first oracle strategy model."""

from __future__ import annotations

import argparse
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from courier.common import load_dataset
from courier.solver.features import build_feature_matrix, feature_names
from courier.solver.graph import OracleSolver


def macro_and_per_robot(labels: np.ndarray, predictions: np.ndarray):
    per_robot = (labels == predictions).mean(axis=0)
    return float(per_robot.mean()), per_robot


def legal_actions(scenes, robot: int):
    return tuple(
        frozenset(int(arc.action) for arc in OracleSolver.adjacency(scene, robot)[scene.robot_rc])
        for scene in scenes
    )


def predict_legal_one(model, features, legal_by_scene):
    probabilities = model.predict_proba(features)
    columns = {int(action): column for column, action in enumerate(model.classes_)}
    predictions = np.empty(len(features), dtype=np.int8)
    for index, legal in enumerate(legal_by_scene):
        predictions[index] = max(
            legal,
            key=lambda action: (
                probabilities[index, columns[action]] if action in columns else -1.0,
                -action,
            ),
        )
    return predictions


def feature_schema_id() -> str:
    payload = "\n".join(feature_names()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_or_build_features(cache_path: Path, train_scenes, validation_scenes):
    schema = feature_schema_id()
    if cache_path.exists():
        with np.load(cache_path) as cached:
            if str(cached["schema"]) == schema:
                print(f"loading cached graph features from {cache_path}")
                return tuple(cached[name] for name in ("train_normal", "train_legged", "validation_normal", "validation_legged"))
        print("feature schema changed; rebuilding cache")

    print("extracting graph features...")
    matrices = (
        build_feature_matrix(train_scenes, legged=False),
        build_feature_matrix(train_scenes, legged=True),
        build_feature_matrix(validation_scenes, legged=False),
        build_feature_matrix(validation_scenes, legged=True),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        schema=np.asarray(schema),
        train_normal=matrices[0],
        train_legged=matrices[1],
        validation_normal=matrices[2],
        validation_legged=matrices[3],
    )
    print(f"saved feature cache to {cache_path}")
    return matrices


def dataset_digest(data_root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("train/scenes.json", "train/labels.json"):
        path = data_root / name
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/solver/oracle_strategy.joblib"))
    parser.add_argument("--cache", type=Path, default=Path("data/solver/oracle_features.npz"))
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=2, help="CPU worker limit, including native thread pools")
    args = parser.parse_args()
    if args.jobs < 1 or args.cv_folds < 2:
        parser.error("--jobs must be positive and --cv-folds must be at least two")
    threadpool_limits(limits=args.jobs)

    train = load_dataset(args.data, "train")
    validation = load_dataset(args.data, "validation")
    assert train.scenes and train.labels and validation.scenes and validation.labels

    train_normal, train_legged, validation_normal, validation_legged = load_or_build_features(
        args.cache, train.scenes, validation.scenes
    )
    y_train = np.asarray(train.labels, dtype=np.int8).reshape(-1, 10)
    y_validation = np.asarray(validation.labels, dtype=np.int8).reshape(-1, 10)

    model_factories = {
        "extra_trees": lambda seed: ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=args.jobs
        ),
        "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=300, max_leaf_nodes=23, l2_regularization=1.0,
            random_state=seed,
        ),
        "random_forest": lambda seed: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=args.jobs
        ),
    }
    train_legal = tuple(legal_actions(train.scenes, robot) for robot in range(10))
    validation_legal = tuple(legal_actions(validation.scenes, robot) for robot in range(10))
    selected_models = []
    selected_families = []
    cv_scores = []
    validation_predictions = np.empty_like(y_validation)

    for robot in range(10):
        features = train_legged if robot == 4 else train_normal
        validation_features = validation_legged if robot == 4 else validation_normal
        labels = y_train[:, robot]
        family_scores = {}
        splitter = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=20261008 + robot,
        )
        folds = tuple(splitter.split(features, labels))
        for family, factory in model_factories.items():
            out_of_fold = np.empty(len(labels), dtype=np.int8)
            for fold, (fit_indices, holdout_indices) in enumerate(folds):
                seed = 20261008 + robot * 100 + fold
                model = factory(seed)
                model.fit(features[fit_indices], labels[fit_indices])
                print(f"R{robot} {family} fold {fold + 1}/{args.cv_folds} fitted", flush=True)
                holdout_legal = tuple(train_legal[robot][index] for index in holdout_indices)
                out_of_fold[holdout_indices] = predict_legal_one(
                    model,
                    features[holdout_indices],
                    holdout_legal,
                )
            family_scores[family] = float((out_of_fold == labels).mean())

        selected_family = max(family_scores, key=lambda name: (family_scores[name], name))
        final_model = model_factories[selected_family](20261008 + robot)
        final_model.fit(features, labels)
        validation_predictions[:, robot] = predict_legal_one(
            final_model,
            validation_features,
            validation_legal[robot],
        )
        selected_models.append(final_model)
        selected_families.append(selected_family)
        cv_scores.append(family_scores)
        print(
            f"R{robot}: selected={selected_family}; "
            + ", ".join(f"{name}={score:.4f}" for name, score in sorted(family_scores.items()))
        )

    validation_macro, validation_per_robot = macro_and_per_robot(y_validation, validation_predictions)
    print(f"validation macro={validation_macro:.4f}; per_robot={[round(x, 4) for x in validation_per_robot]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": dataset_digest(args.data),
        "feature_schema": feature_schema_id(),
        "feature_names": feature_names(),
        "family": "per_robot_train_cv",
        "families": tuple(selected_families),
        "cv_folds": args.cv_folds,
        "cv_scores": tuple(cv_scores),
        "models": selected_models,
        "validation_macro": validation_macro,
        "validation_per_robot": validation_per_robot,
    }
    temporary_path = args.out.with_suffix(args.out.suffix + ".tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(args.out)
    print(f"saved per-robot CV artifact to {args.out}")


if __name__ == "__main__":
    main()
