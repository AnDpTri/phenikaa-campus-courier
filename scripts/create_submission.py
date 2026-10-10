"""Create predictions.json for the competition test split.

The test split contains ten consecutive observations per scene (robot 0..9).
Each scene is decoded once by CV/NLP, then the strategy model emits all ten
actions in observation order.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from dataclasses import asdict
import json
import time
from collections import Counter
from pathlib import Path

from courier.common import load_dataset
from courier.cv import CVInput, CVPipeline, SklearnWeatherClassifier, load_rgb, validate_graph
from courier.cv.neural import (
    NeuralEdgeClassifier,
    NeuralGridDetector,
    NeuralLegendReader,
    NeuralNodeClassifier,
    SharedDetector,
)
from courier.nlp import MissionParser, resolve, resolve_target
from courier.cv.style import with_print_detector
from courier.nlp.neural import HybridMissionParser
from courier.solver import OracleSolver, load_strategy
from courier.solver.diagnostics import fallback_details


def emergency_prediction(scene) -> tuple[int, ...]:
    """Return a legal-or-heading action for every robot if strategy inference fails."""
    actions = []
    for robot_id in range(10):
        try:
            legal = {
                int(arc.action)
                for arc in OracleSolver.adjacency(scene, robot_id).get(scene.robot_rc, ())
            }
        except (KeyError, ValueError):
            legal = set()
        heading = int(scene.robot_heading)
        actions.append(heading if heading in legal or not legal else min(legal))
    return tuple(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"),
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--strategy-artifact",
        type=Path,
        default=Path("artifacts/solver/candidate_strategy.joblib"),
    )
    parser.add_argument("--out", type=Path, help="New output file; existing files are never overwritten")
    parser.add_argument("--detector-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--print-detector", type=Path, help="detector for images classified as print")
    parser.add_argument("--print-detector-threshold", type=float, default=0.30)
    parser.add_argument("--style-artifact", type=Path, default=Path("artifacts/cv/style_classifier.joblib"))
    parser.add_argument("--nlp-goal-threshold", type=float, help="override the stored hybrid goal threshold")
    parser.add_argument(
        "--nlp-model", type=Path, help="neural parser artifact; enables the rule + neural hybrid NLP"
    )
    parser.add_argument("--diagnostics-dir", type=Path, help="New directory for read-only fallback audit")
    parser.add_argument("--disable-map-aware", action="store_true", help="use hybrid NLP without map fallback for ablation")
    parser.add_argument("--disable-repair", action="store_true", help="use the original unreachable fallback for ablation")
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("results") / ("submission_" + datetime.now().strftime("%Y%m%d_%H%M%S")) / "predictions.json"
    if args.out.exists():
        parser.error(f"output already exists: {args.out}; choose a new --out path")
    if args.diagnostics_dir is not None and args.diagnostics_dir.exists():
        parser.error(f"diagnostics directory already exists: {args.diagnostics_dir}")

    dataset = load_dataset(args.data, "test", require_scenes=False)
    rows = dataset.observations
    if len(rows) % 10:
        raise ValueError(f"test: expected a multiple of ten observations, got {len(rows)}")

    shared = SharedDetector(args.artifacts / "cv" / "detector.pt", device=args.detector_device)
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

    predictions: list[int] = []
    diagnostics: Counter[str] = Counter()
    audit = []
    started = time.perf_counter()
    scene_count = len(rows) // 10

    for scene_index in range(scene_count):
        group = rows[scene_index * 10 : (scene_index + 1) * 10]
        robot_ids = tuple(int(row["robot_id"]) for row in group)
        if robot_ids != tuple(range(10)):
            raise ValueError(f"test scene {scene_index}: robot rows are not ordered 0..9")
        if len({row["image"] for row in group}) != 1 or len({row["mission"] for row in group}) != 1:
            raise ValueError(f"test scene {scene_index}: image/mission alignment error")

        scene_id = str(group[0]["id"]).rsplit("-R", 1)[0]
        image_path = args.data / "test" / group[0]["image"]
        cv_input = CVInput(scene_id, load_rgb(image_path))
        graph = cv.extract(cv_input)
        problems = validate_graph(graph)
        if problems:
            diagnostics["structural_graph_warnings"] += 1

        parsed = (
            nlp.parse_for_map(group[0]["mission"], graph.landmarks)
            if not args.disable_map_aware and hasattr(nlp, "parse_for_map")
            else nlp.parse(group[0]["mission"])
        )
        for field, source in getattr(nlp, "last_sources", {}).items():
            diagnostics[f"nlp_{field}_{source}"] += 1
        mission = resolve(parsed, graph.landmarks)
        scene = graph.to_scene(scene_id, mission)
        emergency_reason = None
        try:
            results = strategy.predict_scene_with_diagnostics(scene)
            actions = tuple(int(result.action) for result in results)
            diagnostics["strategy_fallbacks"] += sum(result.fallback_reason is not None for result in results)
        except (KeyError, ValueError) as error:
            emergency_reason = str(error)
            diagnostics["emergency_scenes"] += 1
            actions = emergency_prediction(scene)
            results = None
        predictions.extend(actions)

        if args.diagnostics_dir is not None:
            events = []
            if results is not None:
                events = [fallback_details(scene, robot, item.action, item.fallback_reason)
                          for robot, item in enumerate(results) if item.fallback_reason is not None]
            else:
                events = [fallback_details(scene, robot, action, emergency_reason)
                          for robot, action in enumerate(actions)]
            raw_goal = resolve_target(parsed.goal, graph.landmarks) if parsed.goal is not None else None
            goal_replaced = raw_goal is None or not any(kind == raw_goal[0] for kind, _ in graph.landmarks)
            audit.append({
                "scene_id": scene_id,
                "grid": [graph.grid.rows, graph.grid.cols],
                "nodes": len(graph.grid.nodes), "edges": len(graph.edges),
                "landmarks": len(graph.landmarks), "graph_problems": problems,
                "parsed_goal": asdict(parsed.goal) if parsed.goal else None,
                "parsed_via": asdict(parsed.via) if parsed.via else None,
                "resolver_goal_replaced": goal_replaced,
                "resolver_via_dropped": parsed.via is not None and mission.via is None,
                "fallbacks": events,
            })

        completed = scene_index + 1
        if completed % 50 == 0 or completed == scene_count:
            elapsed = time.perf_counter() - started
            print(f"processed {completed}/{scene_count} scenes ({elapsed:.1f}s)", flush=True)

    if len(predictions) != len(rows):
        raise AssertionError(f"expected {len(rows)} predictions, got {len(predictions)}")
    if any(action not in range(4) for action in predictions):
        raise AssertionError("predictions must contain only integers 0..3")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample = json.loads((args.data / "test" / "sample_submission.json").read_text(encoding="utf-8"))
    if len(sample) != len(predictions):
        raise AssertionError("prediction length does not match sample_submission.json")
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(predictions, separators=(",", ":")))

    elapsed = time.perf_counter() - started
    if args.diagnostics_dir is not None:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=False)
        events = [event for row in audit for event in row["fallbacks"]]
        summary = {
            "scenes": scene_count, "observations": len(rows), "seconds": elapsed,
            "strategy_artifact": str(args.strategy_artifact),
            "original_counters": dict(diagnostics),
            "fallback_scenes": sum(bool(row["fallbacks"]) for row in audit),
            "fallback_robot_observations": len(events),
            "categories": dict(Counter(e["category"] for e in events)),
            "by_robot": dict(Counter(e["robot_id"] for e in events)),
            "fallback_modes": dict(Counter(e["fallback_mode"] for e in events)),
            "fallback_actions": dict(Counter(e["action"] for e in events)),
            "fallbacks_per_scene": dict(Counter(len(row["fallbacks"]) for row in audit)),
            "weak_undirected_route_exists": dict(Counter(str(e["weak_undirected_route_exists"]) for e in events)),
            "resolver_goal_replaced_scenes": sum(row["resolver_goal_replaced"] for row in audit),
            "resolver_via_dropped_scenes": sum(row["resolver_via_dropped"] for row in audit),
            "structural_graph_warning_scenes": sum(bool(row["graph_problems"]) for row in audit),
        }
        for name, data in (("summary.json", summary), ("scene_diagnostics.json", audit)):
            with (args.diagnostics_dir / name).open("x", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
        print("fallback_summary:", json.dumps(summary, ensure_ascii=False), flush=True)
    counts = Counter(predictions)
    print(f"wrote {len(predictions)} predictions to {args.out.resolve()}")
    print(f"action counts: {dict(sorted(counts.items()))}")
    print(f"diagnostics: {dict(diagnostics)}")
    print(f"total seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()
