"""Coordinate search of one robot's additive route-cost weights, optionally per condition.

For each scene the cost of the best route starting with every legal first action
is computed with one backward Dijkstra over (node, heading, via-phase) states,
which matches OracleSolver.score_actions exactly but is ~4x faster. The objective
is the accuracy expected under a uniformly random tie-break:
mean(1/|argmin| if label in argmin else 0), so weights that only create ties do
not score.

    py -3.12 scripts/solver/fit_cost_weights.py --robot 7 --conditions fragile,not_fragile
"""

from __future__ import annotations

import argparse
import heapq
import math
from multiprocessing import Pool
from pathlib import Path

from courier.common import CostProfile, load_dataset
from courier.solver import OracleSolver
from courier.solver.graph import _turn_kind

GRID = {
    "crowded": (-0.5, -0.25, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10),
    "covered": (-0.9, -0.75, -0.6, -0.5, -0.4, -0.25, -0.1, 0, 0.25, 0.5, 1, 2),
    "turn": (0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3),
    "u_turn": (0, 0.25, 0.5, 1, 2, 3, 5),
    "left_turn": (-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1),
    "right_turn": (-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1),
    "stairs": (-0.5, -0.25, 0, 0.5, 1, 2),
}
CONDITIONS = {
    "all": lambda scene: True,
    "rain": lambda scene: scene.weather == "rain",
    "dry": lambda scene: scene.weather != "rain",
    "urgent": lambda scene: scene.mission.urgent,
    "not_urgent": lambda scene: not scene.mission.urgent,
    "fragile": lambda scene: scene.mission.fragile,
    "not_fragile": lambda scene: not scene.mission.fragile,
}
TOL = 1e-9


class Problem:
    """Legal arcs of one scene for one robot, in plain tuples for speed."""

    def __init__(self, scene, robot: int, label: int) -> None:
        adjacency = OracleSolver.adjacency(scene, robot)
        self.arcs = {n: [(a.target, a.action, a.status, a.stairs) for a in arcs] for n, arcs in adjacency.items()}
        self.pred: dict = {}
        for source, arcs in self.arcs.items():
            for target, action, status, stairs in arcs:
                self.pred.setdefault(target, []).append((source, action, status, stairs))
        self.start, self.heading, self.label = scene.robot_rc, scene.robot_heading, label
        self.goals = scene.landmark_candidates(scene.mission.goal, scene.mission.goal_ref)
        self.vias = (
            scene.landmark_candidates(scene.mission.via, scene.mission.via_ref) if scene.mission.via else frozenset()
        )


def arc_cost(profile: CostProfile, status: str, stairs: bool, previous, action) -> float:
    value = profile.step
    if status == "crowded":
        value += profile.crowded
    elif status == "covered":
        value += profile.covered
    if stairs:
        value += profile.stairs
    kind = _turn_kind(previous, action)
    if kind == "left":
        value += profile.turn + profile.left_turn
    elif kind == "right":
        value += profile.turn + profile.right_turn
    elif kind == "u_turn":
        value += profile.u_turn
    return value


def action_costs(problem: Problem, profile: CostProfile) -> dict:
    """Same values as OracleSolver(profile).score_actions, via cost-to-go from the goals."""
    has_via = bool(problem.vias)
    to_go: dict = {}
    queue = []
    for goal in problem.goals:
        for heading in range(4):
            to_go[(goal, heading, True)] = 0.0
            queue.append((0.0, (goal, heading, True)))
    heapq.heapify(queue)
    headings = tuple(range(4))
    while queue:
        cost, (node, arrived_by, phase) = heapq.heappop(queue)
        if cost > to_go[(node, arrived_by, phase)]:
            continue
        if not has_via:
            phases = (True,)
        elif node in problem.vias:
            if not phase:
                continue
            phases = (False, True)
        else:
            phases = (phase,)
        for source, action, status, stairs in problem.pred.get(node, ()):
            if int(action) != arrived_by:
                continue
            for previous in headings:
                step = cost + arc_cost(profile, status, stairs, previous, action)
                for source_phase in phases:
                    state = (source, previous, source_phase)
                    if step < to_go.get(state, math.inf):
                        to_go[state] = step
                        heapq.heappush(queue, (step, state))
    start_phase = (not has_via) or problem.start in problem.vias
    result = {}
    for target, action, status, stairs in problem.arcs[problem.start]:
        rest = to_go.get((target, int(action), start_phase or target in problem.vias), math.inf)
        if rest < math.inf:
            result[int(action)] = arc_cost(profile, status, stairs, problem.heading, action) + rest
    return result


_PROBLEMS: list[Problem] = []


def _init(problems: list[Problem]) -> None:
    global _PROBLEMS
    _PROBLEMS = problems


def _score(weights: dict) -> float:
    profile = CostProfile(**weights)
    total = 0.0
    for problem in _PROBLEMS:
        costs = action_costs(problem, profile)
        if costs:
            best = min(costs.values())
            winners = [action for action, cost in costs.items() if cost <= best + TOL]
            total += (problem.label in winners) / len(winners)
    return total


def is_valid(weights: dict) -> bool:
    try:
        CostProfile(**weights).validate()
    except ValueError:
        return False
    return True


def search(problems: list[Problem], fields: list[str], workers: int, rounds: int = 3) -> tuple[float, dict]:
    size = math.ceil(len(problems) / workers)
    pools = [Pool(1, _init, (problems[i : i + size],)) for i in range(0, len(problems), size)]
    cache: dict = {}

    def objective(weights: dict) -> float:
        key = tuple(sorted((k, v) for k, v in weights.items() if v))
        if key not in cache:
            jobs = [pool.apply_async(_score, (dict(key),)) for pool in pools]
            cache[key] = sum(job.get() for job in jobs) / len(problems)
        return cache[key]

    try:
        weights = {field: 0.0 for field in fields}
        best = objective(weights)
        for _ in range(rounds):
            improved = False
            for field in fields:
                for value in GRID[field]:
                    candidate = dict(weights, **{field: value})
                    if is_valid(candidate) and objective(candidate) > best + 1e-12:
                        best, weights, improved = objective(candidate), candidate, True
            if not improved:
                break
        return best, {k: v for k, v in weights.items() if v}
    finally:
        for pool in pools:
            pool.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--robot", type=int, required=True, choices=range(10))
    parser.add_argument("--conditions", default="all", help=f"comma list of {sorted(CONDITIONS)}")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    fields = [field for field in GRID if field != "stairs" or args.robot == 4]
    train = load_dataset(args.data, "train")
    validation = load_dataset(args.data, "validation")
    for name in args.conditions.split(","):
        keep = CONDITIONS[name]

        def problems(dataset):
            return [
                Problem(scene, args.robot, dataset.labels[index * 10 + args.robot])
                for index, scene in enumerate(dataset.scenes)
                if keep(scene)
            ]

        train_problems, validation_problems = problems(train), problems(validation)
        score, weights = search(train_problems, fields, args.workers)
        _init(validation_problems)
        held_out = _score(weights) / len(validation_problems)
        unit = _score({}) / len(validation_problems)
        print(
            f"R{args.robot} {name:12s} n={len(train_problems)}/{len(validation_problems)}  train {score:.3f}"
            f"  validation {held_out:.3f} (unit cost {unit:.3f})  {weights}",
            flush=True,
        )


if __name__ == "__main__":
    main()
