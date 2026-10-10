"""Repair a predicted graph on which the mission cannot be completed.

With a correct graph every goal is reachable, so an unreachable goal means CV
dropped or misread something. Instead of guessing an action, relax the graph in
the smallest step that makes the mission completable again, then let the normal
strategy model rank actions on the repaired graph:

1. ignore one-way restrictions;
2. also add a normal road between every pair of grid-adjacent nodes without one
   (CV most often misses roads and nodes; status and direction are ~99.9% right);
3. also treat closed roads as open;
4. also add every missing grid node (plus robot and landmark nodes).

Simulated on damaged validation graphs (scripts/solver/evaluate_repair.py) this
raises accuracy on unreachable scenes from 35.1% to 53.2%.

If even that fails, `greedy_action` walks toward the nearest goal by grid distance.
"""

from __future__ import annotations

import dataclasses
from collections import deque

from courier.common.domain import Edge, Scene

from .graph import OracleSolver

STEPS = ("drop_oneway", "add_lattice_edges", "reopen_closed", "add_lattice_nodes")


def _targets(scene: Scene) -> tuple[frozenset, frozenset]:
    mission = scene.mission
    goals = scene.landmark_candidates(mission.goal, mission.goal_ref)
    vias = scene.landmark_candidates(mission.via, mission.via_ref) if mission.via else frozenset()
    return frozenset(goals), frozenset(vias)


def _reached(scene: Scene, robot_id: int, start) -> set:
    adjacency = OracleSolver.adjacency(scene, robot_id)
    seen, queue = {start}, deque([start])
    while queue:
        node = queue.popleft()
        for arc in adjacency.get(node, ()):
            if arc.target not in seen:
                seen.add(arc.target)
                queue.append(arc.target)
    return seen


def mission_reachable(scene: Scene, robot_id: int) -> bool:
    """True when some via (if any) and then some goal can be reached from the robot."""
    if scene.robot_rc not in scene.nodes:
        return False
    goals, vias = _targets(scene)
    if not goals:
        return False
    reached = _reached(scene, robot_id, scene.robot_rc)
    if not vias:
        return bool(goals & reached)
    return any(goals & _reached(scene, robot_id, via) for via in vias & reached)


def _lattice_edges(nodes: frozenset, existing: tuple[Edge, ...]) -> tuple[Edge, ...]:
    """Add a normal road between every pair of grid-adjacent nodes that has none."""
    present = {frozenset((edge.a, edge.b)) for edge in existing}
    added = []
    for r, c in sorted(nodes):
        for neighbour in ((r + 1, c), (r, c + 1)):
            if neighbour in nodes and frozenset(((r, c), neighbour)) not in present:
                added.append(Edge((r, c), neighbour, "normal", False, None))
    return existing + tuple(added)


def _relaxed(scene: Scene, step: str) -> Scene:
    edges = tuple(dataclasses.replace(edge, oneway_to=None) for edge in scene.edges)
    if step in ("reopen_closed", "add_lattice_nodes"):
        edges = tuple(
            dataclasses.replace(edge, status="normal") if edge.status == "closed" else edge for edge in edges
        )
    nodes = scene.nodes
    if step == "add_lattice_nodes":
        nodes = frozenset(
            (r, c) for r in range(scene.grid_rows) for c in range(scene.grid_cols)
        ) | {scene.robot_rc} | {rc for _, rc in scene.landmarks}
    if step != "drop_oneway":
        edges = _lattice_edges(nodes, edges)
    return dataclasses.replace(scene, nodes=nodes, edges=edges)


def repair_scene(scene: Scene, robot_id: int) -> tuple[Scene, str] | None:
    """Smallest relaxation that makes the mission reachable, with the step name."""
    for step in STEPS:
        candidate = _relaxed(scene, step)
        if mission_reachable(candidate, robot_id):
            return candidate, step
    return None


def greedy_action(scene: Scene, legal: frozenset[int]) -> int:
    """Legal action that most reduces grid distance to the nearest via/goal; ties keep the heading."""
    goals, vias = _targets(scene)
    targets = vias or goals or {rc for _, rc in scene.landmarks} or {scene.robot_rc}
    r, c = scene.robot_rc
    step = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def distance(action: int) -> int:
        nr, nc = r + step[action][0], c + step[action][1]
        return min(abs(nr - tr) + abs(nc - tc) for tr, tc in targets)

    heading = int(scene.robot_heading)
    return min(sorted(legal), key=lambda action: (distance(action), action != heading))
