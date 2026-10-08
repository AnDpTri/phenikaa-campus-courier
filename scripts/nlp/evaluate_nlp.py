"""Score the NLP parser + resolver against scenes.json mission annotations.

Landmarks come from ground truth, so this isolates NLP errors from CV errors.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

from courier.common import CostProfile, Mission, load_dataset
from courier.nlp import MissionParser, resolve


def _target(scene, kind: str | None, ref) -> frozenset:
    return scene.landmark_candidates(kind, ref) if kind is not None else frozenset()


def compare(scene, predicted: Mission) -> dict[str, bool]:
    truth = scene.mission
    goal_ok = _target(scene, truth.goal, truth.goal_ref) == _target(scene, predicted.goal, predicted.goal_ref)
    via_ok = _target(scene, truth.via, truth.via_ref) == _target(scene, predicted.via, predicted.via_ref)
    return {
        "goal_nodes": goal_ok,
        "goal_type": truth.goal == predicted.goal,
        "goal_ref": (truth.goal_ref.kind if truth.goal_ref else None) == (predicted.goal_ref.kind if predicted.goal_ref else None),
        "via_nodes": via_ok,
        "urgent": truth.urgent == predicted.urgent,
        "fragile": truth.fragile == predicted.fragile,
        "all": goal_ok and via_ok and truth.urgent == predicted.urgent and truth.fragile == predicted.fragile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--errors", type=Path, help="write mis-parsed missions to this JSON file")
    parser.add_argument(
        "--solver", action="store_true", help="also compare the uniform-cost oracle solver on true vs parsed missions"
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data, args.split)
    assert dataset.scenes is not None
    mission_parser = MissionParser()
    totals: Counter[str] = Counter()
    errors = []
    predictions: list[Mission] = []
    for scene in dataset.scenes:
        parsed = mission_parser.parse(scene.mission.text)
        predicted = resolve(parsed, scene.landmarks)
        predictions.append(predicted)
        result = compare(scene, predicted)
        totals.update(name for name, ok in result.items() if ok)
        if not result["all"]:
            errors.append(
                {
                    "scene_id": scene.scene_id,
                    "failed": [name for name, ok in result.items() if not ok],
                    "text": scene.mission.text,
                    "truth": _mission_dict(scene.mission),
                    "predicted": _mission_dict(predicted),
                    "parsed": {"goal": _spec_dict(parsed.goal), "via": _spec_dict(parsed.via)},
                }
            )
    count = len(dataset.scenes)
    print(f"{args.split}: {count} scenes")
    for name in ("goal_nodes", "goal_type", "goal_ref", "via_nodes", "urgent", "fragile", "all"):
        print(f"  {name:10s} {totals[name] / count:.4f}")
    if args.errors:
        args.errors.parent.mkdir(parents=True, exist_ok=True)
        args.errors.write_text(json.dumps(errors, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.solver:
        _compare_solver(dataset, predictions)


def _compare_solver(dataset, predictions: list[Mission]) -> None:
    """Macro accuracy of the uniform-cost solver with annotated vs parsed missions."""
    from courier.solver import OracleSolver

    solver = OracleSolver(CostProfile())
    hits = Counter()
    for index, (scene, mission) in enumerate(zip(dataset.scenes, predictions)):
        parsed_scene = dataclasses.replace(scene, mission=mission)
        for robot_id in range(10):
            label = dataset.labels[index * 10 + robot_id]
            hits["annotated"] += int(solver.solve(scene, robot_id).action == label)
            try:
                hits["parsed"] += int(solver.solve(parsed_scene, robot_id).action == label)
            except ValueError:
                pass
    rows = 10 * len(dataset.scenes)
    print(f"  solver accuracy with annotated missions {hits['annotated'] / rows:.4f}")
    print(f"  solver accuracy with parsed missions    {hits['parsed'] / rows:.4f}")


def _spec_dict(spec):
    return None if spec is None else {"type": spec.type, "ref": spec.ref, "anchor": spec.anchor}


def _mission_dict(mission: Mission) -> dict:
    def ref(value):
        return None if value is None else [value.kind, value.anchor, list(value.rc)]

    return {
        "goal": mission.goal,
        "goal_ref": ref(mission.goal_ref),
        "via": mission.via,
        "via_ref": ref(mission.via_ref),
        "urgent": mission.urgent,
        "fragile": mission.fragile,
    }


if __name__ == "__main__":
    main()
