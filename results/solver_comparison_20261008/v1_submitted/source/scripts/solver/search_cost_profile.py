"""One-dimensional cost-profile search with train-only tie-priority fitting."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

from courier.common import CostProfile, load_dataset
from courier.solver import OracleSolver


def optimal_sets(dataset, robot_id: int, profile: CostProfile):
    solver = OracleSolver(profile)
    sets = []
    labels = []
    assert dataset.scenes is not None and dataset.labels is not None
    for index, scene in enumerate(dataset.scenes):
        scores = solver.score_actions(scene, robot_id)
        minimum = min(scores.values())
        sets.append({int(action) for action, cost in scores.items() if math.isclose(cost, minimum)})
        labels.append(dataset.labels[index * 10 + robot_id])
    return sets, labels


def accuracy(sets, labels, priority) -> float:
    predictions = (next(action for action in priority if action in choices) for choices in sets)
    return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--robot", type=int, required=True, choices=range(10))
    parser.add_argument(
        "--field",
        required=True,
        choices=("crowded", "covered", "stairs", "turn", "left_turn", "right_turn", "u_turn"),
    )
    parser.add_argument("values", nargs="+", type=float)
    args = parser.parse_args()

    train = load_dataset(args.data, "train")
    validation = load_dataset(args.data, "validation")
    priorities = tuple(itertools.permutations(range(4)))
    results = []
    for value in args.values:
        profile = CostProfile(**{args.field: value})
        train_sets, train_labels = optimal_sets(train, args.robot, profile)
        train_score, priority = max(
            (accuracy(train_sets, train_labels, candidate), candidate) for candidate in priorities
        )
        validation_sets, validation_labels = optimal_sets(validation, args.robot, profile)
        validation_score = accuracy(validation_sets, validation_labels, priority)
        results.append((train_score, validation_score, value, priority))
    for result in sorted(results, reverse=True):
        print(result)


if __name__ == "__main__":
    main()
