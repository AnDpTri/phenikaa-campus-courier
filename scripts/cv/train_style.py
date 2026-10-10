"""Train the drawing-style classifier on train; report validation accuracy.

    PYTHONPATH=src python scripts/cv/train_style.py --out artifacts/cv/style_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from courier.cv import load_rgb
from courier.cv.style import StyleClassifier, style_features


def load_split(data: Path, split: str):
    scenes = json.loads((data / split / "scenes.json").read_text(encoding="utf-8"))
    features = np.stack([style_features(load_rgb(data / split / scene["image"])) for scene in scenes])
    return features, np.asarray([scene["style"] for scene in scenes])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cv/style_classifier.joblib"))
    args = parser.parse_args()

    x_train, y_train = load_split(args.data, "train")
    x_val, y_val = load_split(args.data, "validation")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    model.fit(x_train, y_train)
    predicted = model.predict(x_val)
    print(f"train accuracy {np.mean(model.predict(x_train) == y_train):.4f}")
    print(f"validation accuracy {np.mean(predicted == y_val):.4f} ({int(np.sum(predicted == y_val))}/{len(y_val)})")
    confusion = Counter(zip(y_val, predicted))
    for (truth, guess), count in sorted(confusion.items()):
        if truth != guess:
            print(f"  {truth} -> {guess}: {count}")
    StyleClassifier(model).save(args.out, validation_accuracy=float(np.mean(predicted == y_val)))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
