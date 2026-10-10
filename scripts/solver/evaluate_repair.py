"""Compare the old unreachable-mission fallback with graph repair on simulated CV failures.

Validation has no unreachable missions with the current CV, so failures are
simulated on ground-truth validation graphs: edges, nodes or directions along the
robot's route are damaged until the mission becomes unreachable for wheeled
robots, mimicking a CV miss. True labels stay the original ones.

    PYTHONPATH=src python scripts/solver/evaluate_repair.py --strategy-artifact artifacts/solver/candidate_strategy.joblib
"""

from __future__ import annotations

import argparse
import dataclasses
import random
from collections import Counter, deque
from pathlib import Path

import numpy as np

from courier.common import load_dataset
from courier.solver import OracleSolver
from courier.solver.candidates import load_strategy
from courier.solver.repair import mission_reachable

KINDS = ("drop_edge", "drop_node", "oneway", "closed")


def _route(scene):
    adjacency = OracleSolver.adjacency(scene, 0)
    goals = scene.landmark_candidates(scene.mission.goal, scene.mission.goal_ref)
    parent, queue = {scene.robot_rc: None}, deque([scene.robot_rc])
    while queue:
        node = queue.popleft()
        if node in goals:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]
        for arc in adjacency.get(node, ()):
            if arc.target not in parent:
                parent[arc.target] = node
                queue.append(arc.target)
    return None


def damage(scene, kind: str, rng: random.Random, limit: int = 12):
    keep = {scene.robot_rc} | {rc for _, rc in scene.landmarks}
    for _ in range(limit):
        if not mission_reachable(scene, 0):
            break
        path = _route(scene)
        if path is None or len(path) < 2:
            return None
        hops = list(zip(path, path[1:]))
        if kind == "drop_node":
            inner = [node for node in path[1:-1] if node not in keep]
            if not inner:
                kind = "drop_edge"
            else:
                node = rng.choice(inner)
                scene = dataclasses.replace(
                    scene,
                    nodes=scene.nodes - {node},
                    edges=tuple(e for e in scene.edges if node not in (e.a, e.b)),
                )
                continue
        a, b = rng.choice(hops)
        pair = frozenset((a, b))
        edges = []
        for edge in scene.edges:
            if frozenset((edge.a, edge.b)) != pair:
                edges.append(edge)
            elif kind == "oneway":
                edges.append(dataclasses.replace(edge, oneway_to=a))
            elif kind == "closed":
                edges.append(dataclasses.replace(edge, status="closed"))
        scene = dataclasses.replace(scene, edges=tuple(edges))
    if mission_reachable(scene, 0):
        return None
    if any(not OracleSolver.adjacency(scene, robot).get(scene.robot_rc) for robot in (0, 4)):
        return None  # the real pipeline treats a robot without moves as an emergency scene
    return scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--strategy-artifact", type=Path, default=Path("artifacts/solver/candidate_strategy.joblib"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset(args.data, args.split)
    labels = np.asarray(dataset.labels).reshape(-1, 10)
    strategy = load_strategy(args.strategy_artifact)
    rng = random.Random(args.seed)

    hits = {mode: Counter() for mode in ("old", "repair")}
    totals, steps = Counter(), Counter()
    for index, scene in enumerate(dataset.scenes):
        kind = KINDS[index % len(KINDS)]
        damaged = damage(scene, kind, rng)
        if damaged is None:
            continue
        totals[kind] += 1
        for mode, flag in (("old", False), ("repair", True)):
            strategy.repair_unreachable = flag
            results = strategy.predict_scene_with_diagnostics(damaged)
            correct = sum(int(r.action) == int(labels[index, robot]) for robot, r in enumerate(results))
            hits[mode][kind] += correct
            if flag:
                for r in results:
                    reason = r.fallback_reason or "no_fallback"
                    steps[reason.split("; ")[-1]] += 1
    strategy.repair_unreachable = True

    print(f"{args.split}: simulated unreachable scenes {sum(totals.values())} {dict(totals)}")
    for mode in ("old", "repair"):
        per_kind = "  ".join(f"{k} {hits[mode][k] / (10 * totals[k]):.4f}" for k in KINDS if totals[k])
        overall = sum(hits[mode].values()) / (10 * sum(totals.values()))
        print(f"  {mode:7s} accuracy {overall:.4f}   {per_kind}")
    print(f"  repair outcome per robot: {dict(steps)}")


if __name__ == "__main__":
    main()
