import unittest

from courier.common import Action, Edge, Mission, Scene
from courier.solver.diagnostics import fallback_details


class FallbackDiagnosticTests(unittest.TestCase):
    def scene(self, edges, via=None):
        return Scene(
            scene_id="synthetic", grid_rows=1, grid_cols=3,
            nodes=frozenset(((0, 0), (0, 1), (0, 2))), edges=tuple(edges),
            landmarks=(("library", (0, 2)), ("canteen", (0, 2))),
            robot_rc=(0, 0), robot_heading=Action.RIGHT, weather="dry",
            mission=Mission("synthetic", "library", None, via, None, False, False),
        )

    def test_direction_blocked_route_is_distinct_from_disconnect(self):
        scene = self.scene([
            Edge((0, 0), (0, 1), "normal", False, None),
            Edge((0, 1), (0, 2), "normal", False, (0, 1)),
        ])
        report = fallback_details(scene, 0, 3, "min() iterable argument is empty")
        self.assertEqual(report["category"], "goal_unreachable")
        self.assertTrue(report["weak_undirected_route_exists"])
        self.assertEqual(report["fallback_mode"], "heading")

    def test_disconnected_via(self):
        scene = self.scene([Edge((0, 0), (0, 1), "normal", False, None)], via="canteen")
        report = fallback_details(scene, 0, 3, "empty scores")
        self.assertEqual(report["category"], "via_unreachable")
        self.assertFalse(report["weak_undirected_route_exists"])

    def test_other_error_with_route_is_not_called_unreachable(self):
        scene = self.scene([
            Edge((0, 0), (0, 1), "normal", False, None),
            Edge((0, 1), (0, 2), "normal", False, None),
        ])
        report = fallback_details(scene, 0, 3, "other feature exception")
        self.assertEqual(report["category"], "other_feature_error")
        self.assertEqual(report["unit_route_actions"], [3])
