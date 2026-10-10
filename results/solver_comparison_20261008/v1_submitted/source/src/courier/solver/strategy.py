"""Learned oracle strategy layer with graph-legality enforcement."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np

from courier.common.domain import Scene
from .features import extract_scene_features, feature_names
from .graph import OracleSolver


@dataclass(frozen=True, slots=True)
class StrategyPrediction:
    action: int
    confidence: float
    fallback_reason: str | None = None


class OracleStrategyModel:
    def __init__(self, artifact: dict):
        self.family = artifact.get("family", "per_robot")
        self.models = artifact["models"]
        self.families = tuple(artifact.get("families", (self.family,) * len(self.models)))
        stored_names = tuple(artifact["feature_names"])
        if stored_names != feature_names():
            raise ValueError("Model feature schema does not match the current code")
        if len(self.models) != 10:
            raise ValueError("Expected exactly ten robot models")
        if len(self.families) != 10:
            raise ValueError("Expected exactly ten robot model-family entries")

    @classmethod
    def load(cls, path: str | Path) -> "OracleStrategyModel":
        return cls(joblib.load(path))

    @staticmethod
    def _legal_actions(scene: Scene, robot_id: int) -> frozenset[int]:
        actions = frozenset(int(arc.action) for arc in OracleSolver.adjacency(scene, robot_id).get(scene.robot_rc, ()))
        if not actions:
            raise ValueError(f"{scene.scene_id}/R{robot_id}: no legal outgoing action")
        return actions

    @staticmethod
    def _features_or_error(scene: Scene, *, legged: bool):
        try:
            return extract_scene_features(scene, legged=legged), None
        except ValueError as error:
            return None, str(error)

    @staticmethod
    def _fallback(scene: Scene, legal: frozenset[int], reason: str) -> StrategyPrediction:
        # Keep moving only along legal arcs. No target inference is invented.
        heading = int(scene.robot_heading)
        action = heading if heading in legal else min(legal)
        return StrategyPrediction(action, 0.0, reason)

    def predict_scene(self, scene: Scene) -> tuple[int, ...]:
        return tuple(result.action for result in self.predict_scene_with_diagnostics(scene))

    def predict_scene_with_diagnostics(self, scene: Scene) -> tuple[StrategyPrediction, ...]:
        normal, normal_error = self._features_or_error(scene, legged=False)
        legged, legged_error = self._features_or_error(scene, legged=True)
        predictions = []
        for robot_id, model in enumerate(self.models):
            features = legged if robot_id == 4 else normal
            error = legged_error if robot_id == 4 else normal_error
            legal = self._legal_actions(scene, robot_id)
            if features is None:
                predictions.append(self._fallback(scene, legal, error or "feature extraction failed"))
                continue
            probabilities = model.predict_proba(features.reshape(1, -1))[0]
            by_action = {int(action): float(probability) for action, probability in zip(model.classes_, probabilities)}
            action = max(legal, key=lambda action: (by_action.get(action, -1.0), -action))
            mass = sum(by_action.get(candidate, 0.0) for candidate in legal)
            confidence = by_action.get(action, 0.0) / mass if mass else 0.0
            predictions.append(StrategyPrediction(action, confidence))
        return tuple(predictions)

    def predict_scenes(self, scenes: tuple[Scene, ...]) -> np.ndarray:
        if not scenes:
            return np.empty((0, 10), dtype=np.int8)
        normal_rows = [self._features_or_error(scene, legged=False) for scene in scenes]
        legged_rows = [self._features_or_error(scene, legged=True) for scene in scenes]
        predictions = np.empty((len(scenes), 10), dtype=np.int8)
        for robot_id, model in enumerate(self.models):
            rows = legged_rows if robot_id == 4 else normal_rows
            valid_indices = [index for index, (features, _) in enumerate(rows) if features is not None]
            by_index = {}
            class_columns = {}
            if valid_indices:
                features = np.stack([rows[index][0] for index in valid_indices])
                probabilities = model.predict_proba(features)
                by_index = dict(zip(valid_indices, probabilities))
                class_columns = {int(action): column for column, action in enumerate(model.classes_)}
            for scene_index, scene in enumerate(scenes):
                legal = self._legal_actions(scene, robot_id)
                if scene_index not in by_index:
                    predictions[scene_index, robot_id] = self._fallback(
                        scene, legal, rows[scene_index][1] or "feature extraction failed"
                    ).action
                    continue
                probability = by_index[scene_index]
                predictions[scene_index, robot_id] = max(
                    legal,
                    key=lambda action: (
                        probability[class_columns[action]] if action in class_columns else -1.0,
                        -action,
                    ),
                )
        return predictions
