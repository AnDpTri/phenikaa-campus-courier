"""Typed domain objects shared by the oracle solver and future CV/NLP modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DELTA_TO_ACTION = {delta: Action(i) for i, delta in enumerate(ACTION_DELTAS)}
HEADING_TO_ACTION = {name: Action(i) for i, name in enumerate(ACTION_NAMES)}

NodeRC = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Edge:
    a: NodeRC
    b: NodeRC
    status: str
    stairs: bool
    oneway_to: NodeRC | None


@dataclass(frozen=True, slots=True)
class SpatialRef:
    kind: str
    rc: NodeRC
    anchor: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> SpatialRef | None:
        if value is None:
            return None
        return cls(kind=value["kind"], rc=tuple(value["rc"]), anchor=value.get("anchor"))


@dataclass(frozen=True, slots=True)
class Mission:
    text: str
    goal: str
    goal_ref: SpatialRef | None
    via: str | None
    via_ref: SpatialRef | None
    urgent: bool
    fragile: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Mission:
        return cls(
            text=value["text"],
            goal=value["goal"],
            goal_ref=SpatialRef.from_dict(value.get("goal_ref")),
            via=value.get("via"),
            via_ref=SpatialRef.from_dict(value.get("via_ref")),
            urgent=bool(value["urgent"]),
            fragile=bool(value["fragile"]),
        )


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    grid_rows: int
    grid_cols: int
    nodes: frozenset[NodeRC]
    edges: tuple[Edge, ...]
    landmarks: tuple[tuple[str, NodeRC], ...]
    robot_rc: NodeRC
    robot_heading: Action
    weather: str
    mission: Mission

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Scene:
        return cls(
            scene_id=value["scene_id"],
            grid_rows=value["grid"]["rows"],
            grid_cols=value["grid"]["cols"],
            nodes=frozenset(tuple(node["rc"]) for node in value["nodes"]),
            edges=tuple(
                Edge(
                    a=tuple(edge["a"]),
                    b=tuple(edge["b"]),
                    status=edge["status"],
                    stairs=bool(edge["stairs"]),
                    oneway_to=tuple(edge["oneway_to"]) if edge["oneway_to"] is not None else None,
                )
                for edge in value["edges"]
            ),
            landmarks=tuple((landmark["type"], tuple(landmark["rc"])) for landmark in value["landmarks"]),
            robot_rc=tuple(value["robot"]["rc"]),
            robot_heading=HEADING_TO_ACTION[value["robot"]["heading"]],
            weather=value["weather"],
            mission=Mission.from_dict(value["mission"]),
        )

    def landmark_candidates(self, kind: str, ref: SpatialRef | None) -> frozenset[NodeRC]:
        if ref is not None:
            return frozenset((ref.rc,))
        return frozenset(rc for landmark_kind, rc in self.landmarks if landmark_kind == kind)


@dataclass(frozen=True, slots=True)
class CostProfile:
    """Additive path utility. Negative bonuses are allowed if every arc stays positive."""

    step: float = 1.0
    crowded: float = 0.0
    covered: float = 0.0
    stairs: float = 0.0
    turn: float = 0.0
    left_turn: float = 0.0
    right_turn: float = 0.0
    u_turn: float = 0.0

    def validate(self) -> None:
        minimum_edge = self.step + min(0.0, self.crowded, self.covered) + min(
            0.0, self.stairs
        ) + min(0.0, self.turn + self.left_turn, self.turn + self.right_turn, self.u_turn)
        if minimum_edge <= 0:
            raise ValueError("CostProfile must keep every possible traversal cost positive")

