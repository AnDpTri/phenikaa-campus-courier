"""Benchmark a configurable oracle graph solver on train or validation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from courier.common import CostProfile, load_dataset
from courier.solver import OracleSolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    args = parser.parse_args()

    dataset = load_dataset(args.data, args.split)
    assert dataset.scenes is not None and dataset.labels is not None
    solver = OracleSolver(CostProfile())
    correct = defaultdict(int)
    tied = defaultdict(int)
    for scene_index, scene in enumerate(dataset.scenes):
        for robot_id in range(10):
            result = solver.solve(scene, robot_id)
            label = dataset.labels[scene_index * 10 + robot_id]
            correct[robot_id] += int(result.action == label)
            tied[robot_id] += int(len(result.tied_actions) > 1)

    count = len(dataset.scenes)
    scores = {robot: correct[robot] / count for robot in range(10)}
    print("accuracy_by_robot:", {robot: round(value, 4) for robot, value in scores.items()})
    print("macro_accuracy:", round(sum(scores.values()) / 10, 4))
    print("tie_rate_by_robot:", {robot: round(tied[robot] / count, 4) for robot in range(10)})


if __name__ == "__main__":
    main()
