"""Evaluate a saved oracle strategy model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from courier.common import load_dataset
from courier.solver import load_strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--model", type=Path, default=Path("artifacts/solver/oracle_strategy.joblib"))
    parser.add_argument("--report", type=Path, help="Optional machine-readable evaluation report")
    args = parser.parse_args()

    dataset = load_dataset(args.data, args.split)
    assert dataset.scenes is not None and dataset.labels is not None
    model = load_strategy(args.model)
    started = time.perf_counter()
    predictions = model.predict_scenes(dataset.scenes)
    elapsed = time.perf_counter() - started
    labels = np.asarray(dataset.labels).reshape(-1, 10)
    per_robot = (predictions == labels).mean(axis=0)
    print("model_family:", model.family)
    print("accuracy_by_robot:", {robot: round(float(value), 4) for robot, value in enumerate(per_robot)})
    print("macro_accuracy:", round(float(per_robot.mean()), 4))
    print("seconds:", round(elapsed, 3))
    if args.report is not None:
        masks = {
            "all": np.ones(len(dataset.scenes), dtype=bool),
            "via": np.asarray([scene.mission.via is not None for scene in dataset.scenes]),
            "no_via": np.asarray([scene.mission.via is None for scene in dataset.scenes]),
            "rain": np.asarray([scene.weather == "rain" for scene in dataset.scenes]),
            "dry": np.asarray([scene.weather == "dry" for scene in dataset.scenes]),
        }
        groups = {}
        for name, mask in masks.items():
            count = int(mask.sum())
            groups[name] = {
                "scenes": count,
                "macro_accuracy": float((predictions[mask] == labels[mask]).mean()) if count else None,
                "per_robot": (predictions[mask] == labels[mask]).mean(axis=0).tolist() if count else None,
            }
        report = {
            "split": args.split,
            "model": str(args.model),
            "families": model.families,
            "seconds": elapsed,
            "groups": groups,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("report:", args.report)


if __name__ == "__main__":
    main()
