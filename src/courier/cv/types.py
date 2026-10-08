"""Intermediate CV results passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from courier.common.domain import Action, Edge, Mission, NodeRC, Scene

XY = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels

ROAD_KINDS = ("normal", "crowded", "covered")
LEGEND_ROAD_ORDER = ("normal", "crowded", "covered", "closed", "stairs", "oneway", "robot")
LANDMARK_TYPES = (
    "library",
    "dorm",
    "sports",
    "clinic",
    "canteen",
    "parking",
    "lecture",
    "lab",
    "office",
    "gate",
)
WEATHERS = ("dry", "rain")


@dataclass(frozen=True, slots=True)
class CVInput:
    """Everything a stage may look at. Labels never travel through this object."""

    scene_id: str
    image: np.ndarray  # H x W x 3, RGB, uint8


@dataclass(frozen=True, slots=True)
class LegendEntry:
    kind: str  # normal/crowded/.../robot, weather, or place:<type>
    text: str
    swatch: Box
    label: Box


@dataclass(frozen=True, slots=True)
class LegendReading:
    entries: tuple[LegendEntry, ...]
    weather_box: Box | None

    def entry(self, kind: str) -> LegendEntry | None:
        return next((entry for entry in self.entries if entry.kind == kind), None)

    @property
    def place_types(self) -> frozenset[str]:
        return frozenset(entry.kind.removeprefix("place:") for entry in self.entries if entry.kind.startswith("place:"))


@dataclass(frozen=True, slots=True)
class GridLayout:
    rows: int
    cols: int
    node_xy: dict[NodeRC, XY]

    @property
    def nodes(self) -> frozenset[NodeRC]:
        return frozenset(self.node_xy)

    def adjacent_pairs(self) -> tuple[tuple[NodeRC, NodeRC], ...]:
        """Every pair of existing nodes one step apart, ordered (top/left, bottom/right)."""
        pairs = []
        for r, c in sorted(self.node_xy):
            for nxt in ((r, c + 1), (r + 1, c)):
                if nxt in self.node_xy:
                    pairs.append(((r, c), nxt))
        return tuple(pairs)

    def spacing(self) -> float:
        """Median pixel distance between adjacent nodes; the natural crop scale."""
        distances = [
            float(np.hypot(self.node_xy[b][0] - self.node_xy[a][0], self.node_xy[b][1] - self.node_xy[a][1]))
            for a, b in self.adjacent_pairs()
        ]
        return float(np.median(distances)) if distances else 0.0


@dataclass(frozen=True, slots=True)
class NodeContents:
    robot_rc: NodeRC
    robot_heading: Action
    landmarks: tuple[tuple[str, NodeRC], ...]


@dataclass(frozen=True, slots=True)
class SceneGraph:
    """The image half of `courier.common.Scene`; NLP supplies the mission half."""

    grid: GridLayout
    edges: tuple[Edge, ...]
    landmarks: tuple[tuple[str, NodeRC], ...]
    robot_rc: NodeRC
    robot_heading: Action
    weather: str

    def to_scene(self, scene_id: str, mission: Mission) -> Scene:
        return Scene(
            scene_id=scene_id,
            grid_rows=self.grid.rows,
            grid_cols=self.grid.cols,
            nodes=self.grid.nodes,
            edges=self.edges,
            landmarks=self.landmarks,
            robot_rc=self.robot_rc,
            robot_heading=self.robot_heading,
            weather=self.weather,
            mission=mission,
        )


def edge_key(edge: Edge) -> tuple[NodeRC, NodeRC]:
    a, b = sorted((edge.a, edge.b))
    return a, b
