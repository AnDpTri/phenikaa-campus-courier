"""Train and validate the lightweight weather-icon classifier."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from courier.cv import load_rgb
from courier.cv.features import WEATHER_FEATURE_VERSION, extract_weather_features
from courier.cv.learned import SklearnWeatherClassifier


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = json.loads((root / "weather" / f"metadata_{split}.json").read_text(encoding="utf-8"))
    features = np.stack([extract_weather_features(load_rgb(root / row["image"])) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    scene_ids = tuple(row["scene_id"] for row in rows)
    return features, labels, scene_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cv"), help="root containing weather metadata/crops")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/cv/weather_classifier.joblib"))
    parser.add_argument("--c", type=float, default=0.1, help="LinearSVC regularization parameter")
    args = parser.parse_args()

    train_x, train_y, _ = load_split(args.data, "train")
    validation_x, validation_y, validation_ids = load_split(args.data, "validation")
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=args.c, class_weight="balanced", dual="auto", max_iter=10_000),
    )
    model.fit(train_x, train_y)
    train_prediction = model.predict(train_x)
    validation_prediction = model.predict(validation_x)
    score = float(accuracy_score(validation_y, validation_prediction))

    print("train labels:", dict(Counter(train_y)))
    print(f"train accuracy:      {accuracy_score(train_y, train_prediction):.4f}")
    print(f"validation accuracy: {score:.4f}")
    print("validation confusion [dry, rain]:")
    print(confusion_matrix(validation_y, validation_prediction, labels=("dry", "rain")))
    mistakes = [
        (scene_id, expected, predicted)
        for scene_id, expected, predicted in zip(validation_ids, validation_y, validation_prediction)
        if expected != predicted
    ]
    print("validation mistakes:", mistakes)

    artifact = {
        "kind": SklearnWeatherClassifier.ARTIFACT_KIND,
        "artifact_version": SklearnWeatherClassifier.ARTIFACT_VERSION,
        "feature_version": WEATHER_FEATURE_VERSION,
        "crop_pad": 4.0,
        "model": model,
        "validation_accuracy": score,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)
    print(f"saved: {args.artifact}")


if __name__ == "__main__":
    main()
