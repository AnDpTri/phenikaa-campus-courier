"""Compare predicted SceneGraphs with annotations, per stage and per scene.

Everything is compared in (row, col) space, which is what the solver consumes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .types import SceneGraph, edge_key


@dataclass(slots=True)
class CVReport:
    hits: Counter = field(default_factory=Counter)
    totals: Counter = field(default_factory=Counter)
    node_errors_px: list[float] = field(default_factory=list)

    def _count(self, name: str, ok: bool, weight: int = 1) -> None:
        self.hits[name] += int(ok) * weight
        self.totals[name] += weight

    def add(self, predicted: SceneGraph, truth: SceneGraph) -> None:
        # Grid and nodes.
        grid_ok = (predicted.grid.rows, predicted.grid.cols) == (truth.grid.rows, truth.grid.cols)
        nodes_ok = predicted.grid.nodes == truth.grid.nodes
        self._count("grid_shape", grid_ok)
        self._count("node_set", grid_ok and nodes_ok)
        for rc, (x, y) in truth.grid.node_xy.items():
            found = predicted.grid.node_xy.get(rc)
            self._count("node_recall", found is not None)
            if found is not None:
                self.node_errors_px.append(float(np.hypot(found[0] - x, found[1] - y)))
        for rc in predicted.grid.node_xy:
            self._count("node_precision", rc in truth.grid.node_xy)

        # Edges: existence over every candidate pair either side considers, attributes on shared edges.
        true_edges = {edge_key(edge): edge for edge in truth.edges}
        pred_edges = {edge_key(edge): edge for edge in predicted.edges}
        candidates = set(truth.grid.adjacent_pairs()) | set(true_edges) | set(pred_edges)
        for key in candidates:
            self._count("edge_exists", (key in true_edges) == (key in pred_edges))
        for key in true_edges.keys() & pred_edges.keys():
            t, p = true_edges[key], pred_edges[key]
            self._count("edge_status", t.status == p.status)
            self._count("edge_stairs", t.stairs == p.stairs)
            self._count("edge_oneway", t.oneway_to == p.oneway_to)
            if t.stairs:
                self._count("edge_stairs_recall", p.stairs)
            if t.oneway_to is not None:
                self._count("edge_oneway_recall", t.oneway_to == p.oneway_to)
        edges_ok = true_edges == pred_edges
        self._count("edge_set", edges_ok)

        # Landmarks, robot, weather.
        true_landmarks, pred_landmarks = set(truth.landmarks), set(predicted.landmarks)
        for landmark in true_landmarks:
            self._count("landmark_recall", landmark in pred_landmarks)
        for landmark in pred_landmarks:
            self._count("landmark_precision", landmark in true_landmarks)
        landmarks_ok = true_landmarks == pred_landmarks
        self._count("landmark_set", landmarks_ok)
        robot_ok = predicted.robot_rc == truth.robot_rc
        self._count("robot_rc", robot_ok)
        heading_ok = predicted.robot_heading == truth.robot_heading
        self._count("robot_heading", robot_ok and heading_ok)
        weather_ok = predicted.weather == truth.weather
        self._count("weather", weather_ok)

        self._count(
            "scene_exact",
            grid_ok and nodes_ok and edges_ok and landmarks_ok and robot_ok and heading_ok and weather_ok,
        )

    def summary(self) -> dict[str, float]:
        result = {name: self.hits[name] / self.totals[name] for name in sorted(self.totals)}
        if self.node_errors_px:
            errors = np.asarray(self.node_errors_px)
            result["node_err_px_mean"] = float(errors.mean())
            result["node_err_px_p95"] = float(np.percentile(errors, 95))
        return result
