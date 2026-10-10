from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

from courier.common import load_dataset
from courier.solver import OracleSolver, load_strategy
from courier.solver.repair import greedy_action, mission_reachable, repair_scene

DATA_ROOT = Path(__file__).parents[2] / "Phenikaa_Campus_Courier_2026_v3" / "delivery_public"
ARTIFACT = Path(__file__).parents[2] / "artifacts" / "solver" / "candidate_strategy.joblib"


def _cut_robot_off(scene):
    """Drop every road touching the robot except one, then cut that one's far side."""
    keep = next(edge for edge in scene.edges if scene.robot_rc in (edge.a, edge.b))
    neighbour = keep.b if keep.a == scene.robot_rc else keep.a
    edges = tuple(
        edge for edge in scene.edges
        if edge == keep or (scene.robot_rc not in (edge.a, edge.b) and neighbour not in (edge.a, edge.b))
    )
    return dataclasses.replace(scene, edges=edges)


@unittest.skipUnless((DATA_ROOT / "validation" / "scenes.json").exists(), "dataset not available")
class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenes = load_dataset(DATA_ROOT, "validation").scenes

    def test_ground_truth_missions_are_reachable(self) -> None:
        for scene in self.scenes[:50]:
            self.assertTrue(mission_reachable(scene, 0))
            self.assertTrue(mission_reachable(scene, 4))

    def test_repair_restores_reachability_and_keeps_existing_roads(self) -> None:
        repaired_count = 0
        for scene in self.scenes[:40]:
            broken = _cut_robot_off(scene)
            if mission_reachable(broken, 0):
                continue
            result = repair_scene(broken, 0)
            self.assertIsNotNone(result)
            repaired, step = result
            self.assertTrue(mission_reachable(repaired, 0))
            pairs = {frozenset((e.a, e.b)) for e in repaired.edges}
            self.assertTrue(all(frozenset((e.a, e.b)) in pairs for e in broken.edges))
            repaired_count += 1
        self.assertGreater(repaired_count, 0)

    def test_one_way_is_relaxed_first(self) -> None:
        scene = self.scenes[0]
        reversed_roads = dataclasses.replace(
            scene, edges=tuple(dataclasses.replace(e, oneway_to=e.a) for e in scene.edges)
        )
        if not mission_reachable(reversed_roads, 0):
            self.assertEqual(repair_scene(reversed_roads, 0)[1], "drop_oneway")

    def test_greedy_action_is_legal(self) -> None:
        for scene in self.scenes[:20]:
            legal = frozenset(int(a.action) for a in OracleSolver.adjacency(scene, 0)[scene.robot_rc])
            self.assertIn(greedy_action(scene, legal), legal)

    @unittest.skipUnless(ARTIFACT.exists(), "strategy artifact not available")
    def test_strategy_repairs_instead_of_guessing(self) -> None:
        strategy = load_strategy(ARTIFACT)
        for scene in self.scenes[:40]:
            broken = _cut_robot_off(scene)
            if mission_reachable(broken, 0):
                continue
            results = strategy.predict_scene_with_diagnostics(broken)
            self.assertTrue(any("repaired:" in (r.fallback_reason or "") for r in results))
            return
        self.skipTest("no scene could be cut off")


if __name__ == "__main__":
    unittest.main()
