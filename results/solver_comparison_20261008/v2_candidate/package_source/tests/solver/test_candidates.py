from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from courier.common import load_dataset
from courier.solver import CandidateStrategyModel, OracleSolver, load_strategy
from courier.solver.candidates import (
    best_per_scene,
    candidate_dataset,
    candidate_feature_names,
    candidate_rows,
)
from courier.solver.features import extract_scene_features

DATA_ROOT = Path(__file__).parents[2] / "Phenikaa_Campus_Courier_2026_v3" / "delivery_public"
ARTIFACT = Path(__file__).parents[2] / "artifacts" / "solver" / "candidate_strategy.joblib"


@unittest.skipUnless((DATA_ROOT / "train" / "scenes.json").exists(), "dataset not available")
class CandidateRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load_dataset(DATA_ROOT, "train")

    def test_one_row_per_legal_action(self) -> None:
        assert self.train.scenes is not None
        for scene in self.train.scenes[:20]:
            for legged, robot in ((False, 0), (True, 4)):
                rows, actions = candidate_rows(extract_scene_features(scene, legged=legged))
                legal = {int(arc.action) for arc in OracleSolver.adjacency(scene, robot)[scene.robot_rc]}
                self.assertEqual(set(actions), legal)
                self.assertEqual(rows.shape, (len(legal), len(candidate_feature_names())))

    def test_dataset_marks_exactly_the_label(self) -> None:
        assert self.train.scenes is not None and self.train.labels is not None
        scenes = self.train.scenes[:30]
        matrix = np.stack([extract_scene_features(scene, legged=False) for scene in scenes])
        labels = np.asarray(self.train.labels).reshape(-1, 10)[:30, 0]
        _, y, groups, actions = candidate_dataset(matrix, labels)
        for index in range(len(scenes)):
            positives = actions[(groups == index) & (y == 1)]
            self.assertEqual(positives.tolist(), [labels[index]])

    def test_best_per_scene_breaks_ties_to_lower_action(self) -> None:
        picked = best_per_scene(np.array([0.5, 0.5, 0.2]), np.array([0, 0, 1]), np.array([3, 1, 2]), 2)
        self.assertEqual(picked.tolist(), [1, 2])

    def test_saved_candidate_strategy_predicts_legal_actions(self) -> None:
        if not ARTIFACT.exists():
            self.skipTest("candidate strategy artifact has not been trained")
        model = load_strategy(ARTIFACT)
        self.assertIsInstance(model, CandidateStrategyModel)
        assert self.train.scenes is not None
        for scene in self.train.scenes[:10]:
            for robot, action in enumerate(model.predict_scene(scene)):
                legal = {int(arc.action) for arc in OracleSolver.adjacency(scene, robot)[scene.robot_rc]}
                self.assertIn(action, legal)


if __name__ == "__main__":
    unittest.main()
