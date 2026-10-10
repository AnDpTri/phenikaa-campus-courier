"""Evaluate the complete learned CV -> NLP -> strategy pipeline on validation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from courier.common import load_dataset
from courier.cv import CVInput, CVPipeline, SklearnWeatherClassifier, load_annotations, load_rgb
from courier.cv.neural import (
    NeuralEdgeClassifier,
    NeuralGridDetector,
    NeuralLegendReader,
    NeuralNodeClassifier,
    SharedDetector,
)
from courier.cv.types import SceneGraph, edge_key
from courier.nlp import MissionParser, resolve
from courier.cv.style import with_print_detector
from courier.nlp.neural import HybridMissionParser
from courier.solver import OracleSolver, load_strategy


def discrete_graph_exact(predicted: SceneGraph, truth: SceneGraph) -> bool:
    return (
        (predicted.grid.rows, predicted.grid.cols) == (truth.grid.rows, truth.grid.cols)
        and predicted.grid.nodes == truth.grid.nodes
        and {edge_key(edge): edge for edge in predicted.edges} == {edge_key(edge): edge for edge in truth.edges}
        and set(predicted.landmarks) == set(truth.landmarks)
        and predicted.robot_rc == truth.robot_rc
        and predicted.robot_heading == truth.robot_heading
        and predicted.weather == truth.weather
    )


def emergency_prediction(scene) -> tuple[int, ...]:
    """Always emit ten actions even if a predicted graph disconnects the robot."""
    result = []
    for robot in range(10):
        try:
            legal = {
                int(arc.action)
                for arc in OracleSolver.adjacency(scene, robot).get(scene.robot_rc, ())
            }
        except (KeyError, ValueError):
            legal = set()
        heading = int(scene.robot_heading)
        result.append(heading if heading in legal or not legal else min(legal))
    return tuple(result)


def predict(strategy, scene, failures: Counter[str]) -> tuple[int, ...]:
    try:
        diagnostics = strategy.predict_scene_with_diagnostics(scene)
        failures["feature_fallbacks"] += sum(item.fallback_reason is not None for item in diagnostics)
        return tuple(int(item.action) for item in diagnostics)
    except (KeyError, ValueError) as error:
        failures[f"hard:{type(error).__name__}"] += 1
        return emergency_prediction(scene)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--strategy-artifact",
        type=Path,
        default=Path("artifacts/solver/candidate_strategy.joblib"),
    )
    parser.add_argument("--detector-artifact", type=Path, help="override artifacts/cv/detector.pt")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path, help="Save metrics to a new JSON file")
    parser.add_argument("--detector-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--print-detector", type=Path, help="detector for images classified as print")
    parser.add_argument("--print-detector-threshold", type=float, default=0.30)
    parser.add_argument("--style-artifact", type=Path, default=Path("artifacts/cv/style_classifier.joblib"))
    parser.add_argument("--nlp-goal-threshold", type=float, help="override the stored hybrid goal threshold")
    parser.add_argument(
        "--nlp-model", type=Path, help="neural parser artifact; enables the rule + neural hybrid NLP"
    )
    parser.add_argument("--detector-threshold", type=float, default=0.4)
    parser.add_argument("--semantic-threshold", type=float, default=0.15)
    parser.add_argument("--disable-map-aware", action="store_true", help="use hybrid NLP without map fallback for ablation")
    parser.add_argument("--disable-repair", action="store_true", help="use the original unreachable fallback for ablation")
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error(f"report already exists: {args.report}; choose a new --report path")

    annotations = load_annotations(args.data, args.split)
    dataset = load_dataset(args.data, args.split)
    if args.limit is not None:
        annotations = annotations[: args.limit]
    assert dataset.scenes is not None and dataset.labels is not None
    truth_scenes = dataset.scenes[: len(annotations)]
    labels = np.asarray(dataset.labels, dtype=np.int8).reshape(-1, 10)[: len(annotations)]

    shared = SharedDetector(
        args.detector_artifact or args.artifacts / "cv" / "detector.pt",
        threshold=args.detector_threshold,
        semantic_threshold=args.semantic_threshold,
        device=args.detector_device,
    )
    cv = CVPipeline(
        legend=NeuralLegendReader(shared),
        weather=SklearnWeatherClassifier.load(args.artifacts / "cv" / "weather_classifier.joblib"),
        grid=NeuralGridDetector(shared),
        edges=NeuralEdgeClassifier(args.artifacts / "cv" / "edge_net.pt"),
        nodes=NeuralNodeClassifier(args.artifacts / "cv" / "node_net.pt"),
    )
    if args.print_detector:
        cv = with_print_detector(
            cv, args.style_artifact, args.print_detector, args.print_detector_threshold, device=args.detector_device
        )
    nlp = HybridMissionParser.load(args.nlp_model) if args.nlp_model else MissionParser()
    if args.nlp_goal_threshold is not None:
        if not args.nlp_model or not 0 <= args.nlp_goal_threshold <= 2:
            parser.error("--nlp-goal-threshold requires --nlp-model and a value in [0, 2]")
        nlp.thresholds = dataclasses.replace(nlp.thresholds, goal=args.nlp_goal_threshold)
        print(f"NLP thresholds overridden: {nlp.thresholds}")
    strategy = load_strategy(args.strategy_artifact)
    if args.disable_repair:
        strategy.repair_unreachable = False
    nlp_sources = {"oracle_map": Counter(), "cv_map": Counter()}

    scenarios = ("oracle", "nlp_only", "cv_only", "full")
    failures = {name: Counter() for name in scenarios}
    scenario_scenes = {name: [] for name in scenarios}
    exact_mask = []
    started = time.perf_counter()

    for index, (annotation, truth) in enumerate(zip(annotations, truth_scenes)):
        cv_input = CVInput(annotation.scene_id, load_rgb(annotation.image_path))
        graph = cv.extract(cv_input)
        if not args.disable_map_aware and hasattr(nlp, "parse_for_map"):
            oracle_parsed_mission = resolve(nlp.parse_for_map(truth.mission.text, truth.landmarks), truth.landmarks)
            nlp_sources["oracle_map"].update(getattr(nlp, "last_sources", {}).values())
            learned_parsed_mission = resolve(nlp.parse_for_map(truth.mission.text, graph.landmarks), graph.landmarks)
            nlp_sources["cv_map"].update(getattr(nlp, "last_sources", {}).values())
        else:
            parsed = nlp.parse(truth.mission.text)
            oracle_parsed_mission = resolve(parsed, truth.landmarks)
            learned_parsed_mission = resolve(parsed, graph.landmarks)
        scenes = {
            "oracle": truth,
            "nlp_only": dataclasses.replace(truth, mission=oracle_parsed_mission),
            "cv_only": graph.to_scene(annotation.scene_id, truth.mission),
            "full": graph.to_scene(annotation.scene_id, learned_parsed_mission),
        }
        for name in scenarios:
            scenario_scenes[name].append(scenes[name])
        exact_mask.append(discrete_graph_exact(graph, annotation.graph))

    predictions = {}
    for name in scenarios:
        try:
            predictions[name] = strategy.predict_scenes(tuple(scenario_scenes[name]))
        except (KeyError, ValueError):
            failures[name]["batch_failed"] += 1
            predictions[name] = np.asarray(
                [predict(strategy, scene, failures[name]) for scene in scenario_scenes[name]],
                dtype=np.int8,
            )

    elapsed = time.perf_counter() - started
    count = len(annotations)
    exact_mask_array = np.asarray(exact_mask, dtype=bool)
    exact_scenes = int(exact_mask_array.sum())
    print(f"{args.split}: {count} scenes; seconds={elapsed:.3f}; sec/scene={elapsed / max(count, 1):.3f}")
    print(f"cv_scene_exact: {exact_scenes / max(count, 1):.4f} ({exact_scenes}/{count})")
    for name in scenarios:
        per_robot = (predictions[name] == labels).mean(axis=0)
        print(f"{name:10s} macro={per_robot.mean():.4f} per_robot={[round(float(x), 4) for x in per_robot]}")
        print(f"{'':10s} fallbacks={dict(failures[name])}")
    if exact_scenes:
        print(f"full_on_cv_exact:   {(predictions['full'][exact_mask_array] == labels[exact_mask_array]).mean():.4f}")
    inexact_scenes = count - exact_scenes
    if inexact_scenes:
        print(f"full_on_cv_inexact: {(predictions['full'][~exact_mask_array] == labels[~exact_mask_array]).mean():.4f}")
    if args.report is not None:
        report = {
            "split": args.split,
            "scenes": count,
            "strategy_artifact": str(args.strategy_artifact),
            "nlp_model": str(args.nlp_model) if args.nlp_model else None,
            "nlp_thresholds": dataclasses.asdict(nlp.thresholds) if args.nlp_model else None,
            "print_detector": str(args.print_detector) if args.print_detector else None,
            "map_aware": bool(args.nlp_model and not args.disable_map_aware),
            "repair_enabled": not args.disable_repair,
            "nlp_sources": {name: dict(counts) for name, counts in nlp_sources.items()},
            "seconds": elapsed,
            "cv_scene_exact": exact_scenes / max(count, 1),
            "scenarios": {
                name: {
                    "macro_accuracy": float((predictions[name] == labels).mean()),
                    "per_robot": (predictions[name] == labels).mean(axis=0).tolist(),
                    "failures": dict(failures[name]),
                }
                for name in scenarios
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
