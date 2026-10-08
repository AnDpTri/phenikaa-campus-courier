"""Artifact-backed implementations of learned CV stages."""

from __future__ import annotations

from pathlib import Path

import joblib

from .features import WEATHER_FEATURE_VERSION, extract_weather_features
from .imaging import crop_box
from .types import CVInput, WEATHERS, LegendReading


class SklearnWeatherClassifier:
    """Classify the weather icon using a small scikit-learn artifact."""

    ARTIFACT_KIND = "courier.cv.weather"
    ARTIFACT_VERSION = 1

    def __init__(self, artifact: dict):
        if artifact.get("kind") != self.ARTIFACT_KIND:
            raise ValueError("Not a courier weather-classifier artifact")
        if artifact.get("artifact_version") != self.ARTIFACT_VERSION:
            raise ValueError("Unsupported weather artifact version")
        if artifact.get("feature_version") != WEATHER_FEATURE_VERSION:
            raise ValueError("Weather feature schema does not match the artifact")
        self.model = artifact["model"]
        self.crop_pad = float(artifact.get("crop_pad", 4.0))

    @classmethod
    def load(cls, path: str | Path) -> "SklearnWeatherClassifier":
        return cls(joblib.load(path))

    def classify(self, cv_input: CVInput, legend: LegendReading) -> str:
        if legend.weather_box is None:
            raise ValueError(f"{cv_input.scene_id}: weather icon was not located")
        patch = crop_box(cv_input.image, legend.weather_box, pad=self.crop_pad)
        features = extract_weather_features(patch).reshape(1, -1)
        prediction = str(self.model.predict(features)[0])
        if prediction not in WEATHERS:
            raise ValueError(f"{cv_input.scene_id}: invalid weather prediction {prediction!r}")
        return prediction
