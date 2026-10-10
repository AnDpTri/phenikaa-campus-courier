"""Drawing-style classifier and a CV pipeline that routes each image by style.

The 768 px detector (results/cv_experiments/print_768_aug_20261008) is much better
on print images (scene exact 66% -> 85%) but worse on classic and sketch, so the
routed pipeline sends only images classified as print to it.

Features: per-channel colour statistics, an HSV histogram and a small grey
thumbnail of the whole image; logistic regression trained on train only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from .pipeline import CVPipeline
from .types import CVInput, SceneGraph

STYLES = ("classic", "night", "print", "sketch")
ARTIFACT_KIND = "courier.cv.style_classifier"


def style_features(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    hist /= max(hist.sum(), 1.0)
    grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edges = cv2.Canny((grey * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    stats = np.concatenate(
        [
            small.reshape(-1, 3).mean(0) / 255.0,
            small.reshape(-1, 3).std(0) / 255.0,
            [grey.mean(), grey.std(), (grey < 0.25).mean(), (grey > 0.85).mean(), edges.mean()],
            np.histogram(grey, bins=16, range=(0, 1))[0] / grey.size,
        ]
    )
    thumbnail = cv2.resize(grey, (16, 16), interpolation=cv2.INTER_AREA).ravel()
    return np.concatenate([stats, hist, thumbnail]).astype(np.float32)


class StyleClassifier:
    def __init__(self, model) -> None:
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "StyleClassifier":
        artifact = joblib.load(path)
        if artifact.get("kind") != ARTIFACT_KIND:
            raise ValueError(f"{path} is not a style classifier artifact")
        return cls(artifact["model"])

    def save(self, path: str | Path, **metadata) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": ARTIFACT_KIND, "model": self.model, **metadata}, path)

    def predict(self, image: np.ndarray) -> str:
        return str(self.model.predict(style_features(image)[None])[0])


@dataclass
class StyleRoutedPipeline:
    """Use `routes[style]` when the classifier predicts that style, else `default`."""

    classifier: StyleClassifier
    default: CVPipeline
    routes: dict[str, CVPipeline]
    last_style: str | None = None

    def extract(self, cv_input: CVInput) -> SceneGraph:
        self.last_style = self.classifier.predict(cv_input.image)
        return self.routes.get(self.last_style, self.default).extract(cv_input)


def with_print_detector(
    default: CVPipeline,
    style_artifact: str | Path,
    print_detector: str | Path,
    threshold: float = 0.30,
    semantic_threshold: float = 0.15,
    device: str = "auto",
) -> StyleRoutedPipeline:
    """Route print images to a pipeline whose legend/grid stages use `print_detector`."""
    import dataclasses

    from .neural import NeuralGridDetector, NeuralLegendReader, SharedDetector

    shared = SharedDetector(print_detector, threshold=threshold, semantic_threshold=semantic_threshold, device=device)
    printed = dataclasses.replace(default, legend=NeuralLegendReader(shared), grid=NeuralGridDetector(shared))
    return StyleRoutedPipeline(StyleClassifier.load(style_artifact), default, {"print": printed})
