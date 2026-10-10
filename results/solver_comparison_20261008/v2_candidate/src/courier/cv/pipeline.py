"""Compose the five stages into image -> SceneGraph."""

from __future__ import annotations

from dataclasses import dataclass

from .stages import EdgeClassifier, GridDetector, LegendReader, NodeClassifier, WeatherClassifier
from .types import CVInput, SceneGraph, edge_key


@dataclass(slots=True)
class CVPipeline:
    legend: LegendReader
    weather: WeatherClassifier
    grid: GridDetector
    edges: EdgeClassifier
    nodes: NodeClassifier

    def extract(self, cv_input: CVInput) -> SceneGraph:
        legend = self.legend.read(cv_input)
        weather = self.weather.classify(cv_input, legend)
        grid = self.grid.detect(cv_input, legend)
        edges = self.edges.classify(cv_input, grid, legend)
        contents = self.nodes.classify(cv_input, grid, legend)
        return SceneGraph(
            grid=grid,
            edges=edges,
            landmarks=contents.landmarks,
            robot_rc=contents.robot_rc,
            robot_heading=contents.robot_heading,
            weather=weather,
        )


def validate_graph(graph: SceneGraph) -> list[str]:
    """Structural problems that make a graph unusable by the solver. Empty means OK."""
    problems: list[str] = []
    nodes = graph.grid.nodes
    if not (5 <= graph.grid.rows <= 9 and 5 <= graph.grid.cols <= 9):
        problems.append(f"grid {graph.grid.rows}x{graph.grid.cols} outside 5..9")
    for r, c in nodes:
        if not (0 <= r < graph.grid.rows and 0 <= c < graph.grid.cols):
            problems.append(f"node {(r, c)} outside grid")
    if graph.robot_rc not in nodes:
        problems.append(f"robot {graph.robot_rc} is not a node")
    seen = set()
    for edge in graph.edges:
        key = edge_key(edge)
        if key in seen:
            problems.append(f"duplicate edge {key}")
        seen.add(key)
        if edge.a not in nodes or edge.b not in nodes:
            problems.append(f"edge {key} touches a missing node")
        if abs(edge.a[0] - edge.b[0]) + abs(edge.a[1] - edge.b[1]) != 1:
            problems.append(f"edge {key} is not between adjacent nodes")
        if edge.oneway_to is not None and edge.oneway_to not in (edge.a, edge.b):
            problems.append(f"edge {key} one-way target {edge.oneway_to} is not an endpoint")
    for kind, rc in graph.landmarks:
        if rc not in nodes:
            problems.append(f"landmark {kind} at {rc} is not a node")
        if rc == graph.robot_rc:
            problems.append(f"landmark {kind} shares the robot node {rc}")
    return problems
