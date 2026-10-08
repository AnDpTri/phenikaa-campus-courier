"""Oracle graph solver with legal movement and configurable additive route costs."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable

from courier.common.domain import DELTA_TO_ACTION, Action, CostProfile, Edge, NodeRC, Scene


@dataclass(frozen=True, slots=True)
class Arc:
    source: NodeRC
    target: NodeRC
    action: Action
    status: str
    stairs: bool


@dataclass(frozen=True, slots=True)
class RouteResult:
    action: Action
    cost: float
    tied_actions: tuple[Action, ...]


def _turn_kind(previous: Action, current: Action) -> str:
    if previous == current:
        return "straight"
    if (int(previous), int(current)) in ((0, 1), (1, 0), (2, 3), (3, 2)):
        return "u_turn"
    # Coordinates use y downward. These are geometric left turns on the image.
    if (int(previous), int(current)) in ((0, 2), (2, 1), (1, 3), (3, 0)):
        return "left"
    return "right"


class OracleSolver:
    """Find the best first action when SceneGraph and parsed Mission are known."""

    def __init__(self, profile: CostProfile | None = None, tie_priority: Iterable[int] = (0, 1, 2, 3)):
        self.profile = profile or CostProfile()
        self.profile.validate()
        priority = tuple(Action(value) for value in tie_priority)
        if set(priority) != set(Action):
            raise ValueError("tie_priority must contain each action exactly once")
        self.tie_priority = priority

    @staticmethod
    def adjacency(scene: Scene, robot_id: int) -> dict[NodeRC, tuple[Arc, ...]]:
        mutable: dict[NodeRC, list[Arc]] = {node: [] for node in scene.nodes}
        for edge in scene.edges:
            if edge.status == "closed" or (edge.stairs and robot_id != 4):
                continue
            OracleSolver._append_allowed_arc(mutable, edge, edge.a, edge.b)
            OracleSolver._append_allowed_arc(mutable, edge, edge.b, edge.a)
        return {node: tuple(arcs) for node, arcs in mutable.items()}

    @staticmethod
    def _append_allowed_arc(mutable: dict[NodeRC, list[Arc]], edge: Edge, source: NodeRC, target: NodeRC) -> None:
        if edge.oneway_to is not None and edge.oneway_to != target:
            return
        delta = (target[0] - source[0], target[1] - source[1])
        mutable[source].append(
            Arc(source=source, target=target, action=DELTA_TO_ACTION[delta], status=edge.status, stairs=edge.stairs)
        )

    def _arc_cost(self, arc: Arc, previous_heading: Action) -> float:
        value = self.profile.step
        if arc.status == "crowded":
            value += self.profile.crowded
        elif arc.status == "covered":
            value += self.profile.covered
        if arc.stairs:
            value += self.profile.stairs
        turn_kind = _turn_kind(previous_heading, arc.action)
        if turn_kind == "left":
            value += self.profile.turn + self.profile.left_turn
        elif turn_kind == "right":
            value += self.profile.turn + self.profile.right_turn
        elif turn_kind == "u_turn":
            value += self.profile.u_turn
        return value

    def score_actions(
        self,
        scene: Scene,
        robot_id: int,
        *,
        adjacency: dict[NodeRC, tuple[Arc, ...]] | None = None,
    ) -> dict[Action, float]:
        # Feature extraction evaluates many cost profiles on the same scene.
        # Accepting a prebuilt adjacency avoids rebuilding identical legal arcs.
        adjacency = adjacency if adjacency is not None else self.adjacency(scene, robot_id)
        goal_nodes = scene.landmark_candidates(scene.mission.goal, scene.mission.goal_ref)
        via_nodes = (
            scene.landmark_candidates(scene.mission.via, scene.mission.via_ref)
            if scene.mission.via is not None
            else frozenset()
        )
        if not goal_nodes:
            raise ValueError(f"{scene.scene_id}: no goal candidates")
        if scene.mission.via is not None and not via_nodes:
            raise ValueError(f"{scene.scene_id}: no via candidates")
        if scene.robot_rc not in adjacency:
            raise ValueError(f"{scene.scene_id}: robot position is absent from graph")

        scores: dict[Action, float] = {}
        for first_arc in adjacency[scene.robot_rc]:
            reached_via = not via_nodes or scene.robot_rc in via_nodes or first_arc.target in via_nodes
            initial_cost = self._arc_cost(first_arc, scene.robot_heading)
            tail = self._shortest_tail(
                adjacency,
                first_arc.target,
                first_arc.action,
                reached_via,
                via_nodes,
                goal_nodes,
            )
            if math.isfinite(tail):
                scores[first_arc.action] = initial_cost + tail
        return scores

    def _shortest_tail(
        self,
        adjacency: dict[NodeRC, tuple[Arc, ...]],
        start: NodeRC,
        heading: Action,
        reached_via: bool,
        via_nodes: frozenset[NodeRC],
        goal_nodes: frozenset[NodeRC],
    ) -> float:
        start_state = (start, heading, reached_via)
        queue: list[tuple[float, NodeRC, int, bool]] = [(0.0, start, int(heading), reached_via)]
        best = {start_state: 0.0}
        while queue:
            cost, node, heading_value, phase = heapq.heappop(queue)
            current_heading = Action(heading_value)
            state = (node, current_heading, phase)
            if cost != best.get(state):
                continue
            if phase and node in goal_nodes:
                return cost
            for arc in adjacency[node]:
                next_phase = phase or arc.target in via_nodes
                next_cost = cost + self._arc_cost(arc, current_heading)
                next_state = (arc.target, arc.action, next_phase)
                if next_cost < best.get(next_state, math.inf):
                    best[next_state] = next_cost
                    heapq.heappush(queue, (next_cost, arc.target, int(arc.action), next_phase))
        return math.inf

    def solve(self, scene: Scene, robot_id: int) -> RouteResult:
        scores = self.score_actions(scene, robot_id)
        if not scores:
            raise ValueError(f"{scene.scene_id}/R{robot_id}: no route to mission target")
        minimum = min(scores.values())
        tied = tuple(action for action in self.tie_priority if math.isclose(scores.get(action, math.inf), minimum))
        return RouteResult(action=tied[0], cost=minimum, tied_actions=tied)
