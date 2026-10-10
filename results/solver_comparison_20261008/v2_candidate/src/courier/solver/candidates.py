"""Candidate-level strategy: score every legal first action, pick the best.

The scene-level model classifies four absolute directions from one scene row.
Here each legal action becomes its own row: the shared scene context, that
action's immediate features (status, turn relative to the heading, distance to
the next waypoint) and its regret/rank/optimality under every cost profile. One
binary classifier per robot learns "is this the robot's move", so training sees
2-4 rows per scene and never needs to learn each absolute direction separately.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from courier.common.domain import Scene
from .features import PROFILE_LIBRARY, extract_scene_features, feature_names
from .graph import OracleSolver
from .strategy import StrategyPrediction

_NAMES = feature_names()
_COLUMN = {name: index for index, name in enumerate(_NAMES)}
_IMMEDIATE = (
    "legal", "normal", "crowded", "covered", "stairs", "straight", "left", "right", "uturn",
    "next_degree", "waypoint_manhattan",
)
_PROFILE_PARTS = ("regret", "rank", "best")


def _scene_columns() -> list[int]:
    """Columns that describe the whole scene (everything before the per-action blocks)."""
    first_action = _COLUMN["a0_legal"]
    return list(range(first_action))


SCENE_COLUMNS = _scene_columns()


def candidate_feature_names() -> tuple[str, ...]:
    names = [_NAMES[index] for index in SCENE_COLUMNS]
    names += [f"cand_{key}" for key in _IMMEDIATE]
    names += [f"cand_abs_{action}" for action in range(4)]
    names += ["legal_count", "cand_waypoint_manhattan_excess"]
    names += [f"cand_{profile}_{part}" for profile, _ in PROFILE_LIBRARY for part in _PROFILE_PARTS]
    return tuple(names)


def candidate_rows(scene_features: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Rows for the legal actions of one scene feature vector, and those actions."""
    x = scene_features
    legal = tuple(action for action in range(4) if x[_COLUMN[f"a{action}_legal"]] > 0)
    nearest = min(x[_COLUMN[f"a{action}_waypoint_manhattan"]] for action in legal) if legal else 0.0
    scene_part = x[SCENE_COLUMNS]
    rows = []
    for action in legal:
        row = [scene_part]
        row.append(np.asarray([x[_COLUMN[f"a{action}_{key}"]] for key in _IMMEDIATE]))
        row.append(np.asarray([float(action == other) for other in range(4)]))
        row.append(np.asarray([len(legal), x[_COLUMN[f"a{action}_waypoint_manhattan"]] - nearest]))
        row.append(
            np.asarray(
                [x[_COLUMN[f"{profile}_a{action}_{part}"]] for profile, _ in PROFILE_LIBRARY for part in _PROFILE_PARTS]
            )
        )
        rows.append(np.concatenate(row))
    width = len(candidate_feature_names())
    return (np.stack(rows).astype(np.float32) if rows else np.empty((0, width), np.float32)), legal


def candidate_dataset(scene_matrix: np.ndarray, labels: np.ndarray | None):
    """Stack candidate rows of many scenes: (X, y, scene index, action)."""
    xs, ys, groups, actions = [], [], [], []
    for index, features in enumerate(scene_matrix):
        rows, legal = candidate_rows(features)
        xs.append(rows)
        groups.extend([index] * len(legal))
        actions.extend(legal)
        if labels is not None:
            ys.extend(int(labels[index] == action) for action in legal)
    return np.concatenate(xs), np.asarray(ys, dtype=np.int8), np.asarray(groups), np.asarray(actions)


def best_per_scene(probabilities: np.ndarray, groups: np.ndarray, actions: np.ndarray, count: int) -> np.ndarray:
    """Highest-probability action per scene; ties go to the lower action id."""
    best = np.full(count, -1, dtype=np.int8)
    score = np.full(count, -np.inf)
    for probability, group, action in zip(probabilities, groups, actions):
        if probability > score[group] or (probability == score[group] and action < best[group]):
            score[group], best[group] = probability, action
    return best


class CandidateStrategyModel:
    ARTIFACT_KIND = "courier.solver.candidate_ranker"

    def __init__(self, artifact: dict):
        if artifact.get("kind") != self.ARTIFACT_KIND:
            raise ValueError("Not a candidate-ranker artifact")
        if tuple(artifact["feature_names"]) != candidate_feature_names():
            raise ValueError("Candidate feature schema does not match the current code")
        if len(artifact["models"]) != 10:
            raise ValueError("Expected exactly ten robot models")
        self.models = artifact["models"]
        self.family = "candidate_ranker"
        self.families = ("candidate_ranker",) * 10

    @classmethod
    def load(cls, path: str | Path) -> "CandidateStrategyModel":
        return cls(joblib.load(path))

    def predict_scene_with_diagnostics(self, scene: Scene) -> tuple[StrategyPrediction, ...]:
        predictions = []
        cached: dict[bool, tuple[np.ndarray, tuple[int, ...]] | str] = {}
        for robot_id, model in enumerate(self.models):
            legged = robot_id == 4
            legal = frozenset(int(arc.action) for arc in OracleSolver.adjacency(scene, robot_id).get(scene.robot_rc, ()))
            if not legal:
                raise ValueError(f"{scene.scene_id}/R{robot_id}: no legal outgoing action")
            if legged not in cached:
                try:
                    cached[legged] = candidate_rows(extract_scene_features(scene, legged=legged))
                except ValueError as error:
                    cached[legged] = str(error)
            entry = cached[legged]
            if isinstance(entry, str) or not entry[1]:
                heading = int(scene.robot_heading)
                action = heading if heading in legal else min(legal)
                predictions.append(StrategyPrediction(action, 0.0, entry if isinstance(entry, str) else "no candidates"))
                continue
            rows, actions = entry
            probabilities = model.predict_proba(rows)[:, 1]
            order = max(range(len(actions)), key=lambda i: (probabilities[i], -actions[i]))
            confidence = float(probabilities[order] / max(probabilities.sum(), 1e-9))
            predictions.append(StrategyPrediction(int(actions[order]), confidence))
        return tuple(predictions)

    def predict_scene(self, scene: Scene) -> tuple[int, ...]:
        return tuple(prediction.action for prediction in self.predict_scene_with_diagnostics(scene))

    def predict_scenes(self, scenes: tuple[Scene, ...]) -> np.ndarray:
        return np.asarray([self.predict_scene(scene) for scene in scenes], dtype=np.int8).reshape(-1, 10)


def load_strategy(path: str | Path):
    """Load either strategy artifact kind."""
    from .strategy import OracleStrategyModel

    artifact = joblib.load(path)
    if isinstance(artifact, dict) and artifact.get("kind") == CandidateStrategyModel.ARTIFACT_KIND:
        return CandidateStrategyModel(artifact)
    return OracleStrategyModel(artifact)

