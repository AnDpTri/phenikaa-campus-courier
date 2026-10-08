from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from courier.common import ACTION_DELTAS, Action, CostProfile, Edge, Mission, Scene, load_dataset
from courier.solver import OracleSolver, OracleStrategyModel, load_strategy
from courier.solver.features import feature_names


DATA_ROOT = Path(__file__).parents[2] / "Phenikaa_Campus_Courier_2026_v3" / "delivery_public"


class OracleSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load_dataset(DATA_ROOT, "train")

    def test_all_labels_are_legal_actions(self) -> None:
        assert self.train.scenes is not None and self.train.labels is not None
        for scene_index, scene in enumerate(self.train.scenes):
            for robot_id in range(10):
                label = self.train.labels[scene_index * 10 + robot_id]
                adjacency = OracleSolver.adjacency(scene, robot_id)
                dr, dc = ACTION_DELTAS[label]
                target = (scene.robot_rc[0] + dr, scene.robot_rc[1] + dc)
                self.assertIn(target, {arc.target for arc in adjacency[scene.robot_rc]})

    def test_robot_zero_always_uses_a_shortest_route(self) -> None:
        assert self.train.scenes is not None and self.train.labels is not None
        solver = OracleSolver(CostProfile())
        for scene_index, scene in enumerate(self.train.scenes):
            label = self.train.labels[scene_index * 10]
            scores = solver.score_actions(scene, robot_id=0)
            self.assertAlmostEqual(scores[label], min(scores.values()))

    def test_saved_strategy_only_predicts_legal_actions(self) -> None:
        model_path = Path(__file__).parents[2] / "artifacts" / "solver" / "candidate_strategy.joblib"
        if not model_path.exists():
            self.skipTest("strategy artifact has not been trained")
        assert self.train.scenes is not None
        model = load_strategy(model_path)
        for scene in self.train.scenes[:10]:
            predictions = model.predict_scene(scene)
            for robot_id, prediction in enumerate(predictions):
                legal = {int(arc.action) for arc in OracleSolver.adjacency(scene, robot_id)[scene.robot_rc]}
                self.assertIn(prediction, legal)

    @staticmethod
    def make_scene(*, edges, landmarks, robot_rc=(0, 0), heading=Action.RIGHT, goal="library", via=None):
        nodes = frozenset({point for edge in edges for point in (edge.a, edge.b)})
        return Scene(
            scene_id="synthetic",
            grid_rows=2,
            grid_cols=2,
            nodes=nodes,
            edges=tuple(edges),
            landmarks=tuple(landmarks),
            robot_rc=robot_rc,
            robot_heading=heading,
            weather="dry",
            mission=Mission(
                text="synthetic",
                goal=goal,
                goal_ref=None,
                via=via,
                via_ref=None,
                urgent=False,
                fragile=False,
            ),
        )

    def test_closed_oneway_and_stairs_are_enforced(self) -> None:
        scene = self.make_scene(
            edges=(
                Edge((0, 0), (0, 1), "closed", False, None),
                Edge((0, 0), (1, 0), "normal", True, None),
                Edge((0, 1), (1, 1), "normal", False, (1, 1)),
            ),
            landmarks=(("library", (1, 1)),),
        )
        wheeled = OracleSolver.adjacency(scene, robot_id=0)
        legged = OracleSolver.adjacency(scene, robot_id=4)
        self.assertEqual(wheeled[(0, 0)], ())
        self.assertEqual({arc.target for arc in legged[(0, 0)]}, {(1, 0)})
        self.assertEqual({arc.target for arc in wheeled[(0, 1)]}, {(1, 1)})
        self.assertNotIn((0, 1), {arc.target for arc in wheeled[(1, 1)]})

    def test_via_is_visited_before_goal(self) -> None:
        scene = self.make_scene(
            edges=(
                Edge((0, 0), (0, 1), "normal", False, None),
                Edge((0, 0), (1, 0), "normal", False, None),
                Edge((1, 0), (1, 1), "normal", False, None),
                Edge((1, 1), (0, 1), "normal", False, None),
            ),
            landmarks=(("library", (0, 1)), ("clinic", (1, 0))),
            via="clinic",
        )
        result = OracleSolver(CostProfile()).solve(scene, robot_id=0)
        self.assertEqual(result.action, Action.DOWN)

    def test_no_route_is_reported(self) -> None:
        scene = self.make_scene(
            edges=(Edge((0, 0), (0, 1), "closed", False, None),),
            landmarks=(("library", (0, 1)),),
        )
        with self.assertRaisesRegex(ValueError, "no route"):
            OracleSolver(CostProfile()).solve(scene, robot_id=0)

    def test_starting_at_via_marks_it_visited(self) -> None:
        scene = self.make_scene(
            edges=(Edge((0, 0), (0, 1), "normal", False, (0, 1)),),
            landmarks=(("clinic", (0, 0)), ("library", (0, 1))),
            via="clinic",
        )
        result = OracleSolver().solve(scene, robot_id=0)
        self.assertEqual(result.action, Action.RIGHT)
        self.assertEqual(result.cost, 1.0)

    def test_cached_adjacency_preserves_costs(self) -> None:
        assert self.train.scenes is not None
        solver = OracleSolver(CostProfile(crowded=2.0, turn=0.5))
        for scene in self.train.scenes[:10]:
            adjacency = solver.adjacency(scene, robot_id=0)
            self.assertEqual(
                solver.score_actions(scene, robot_id=0),
                solver.score_actions(scene, robot_id=0, adjacency=adjacency),
            )

    def test_missing_target_fallback_is_explicit_and_legal(self) -> None:
        scene = self.make_scene(
            edges=(Edge((0, 0), (0, 1), "normal", False, None),),
            landmarks=(),
        )
        model = OracleStrategyModel({
            "family": "test", "models": [None] * 10, "feature_names": feature_names(),
        })
        results = model.predict_scene_with_diagnostics(scene)
        self.assertTrue(all(result.action == Action.RIGHT for result in results))
        self.assertTrue(all(result.confidence == 0.0 for result in results))
        self.assertTrue(all("no goal candidates" in result.fallback_reason for result in results))
        np.testing.assert_array_equal(model.predict_scenes((scene,)), np.full((1, 10), Action.RIGHT))
        self.assertEqual(model.predict_scenes(()).shape, (0, 10))

    def test_missing_via_does_not_silently_skip_it(self) -> None:
        scene = self.make_scene(
            edges=(Edge((0, 0), (0, 1), "normal", False, None),),
            landmarks=(("library", (0, 1)),),
            via="clinic",
        )
        with self.assertRaisesRegex(ValueError, "no via candidates"):
            OracleSolver().solve(scene, robot_id=0)


if __name__ == "__main__":
    unittest.main()
