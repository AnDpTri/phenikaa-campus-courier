"""Create predictions.json for the competition test split.

The test split contains ten consecutive observations per scene (robot 0..9).
Each scene is decoded once by CV/NLP, then the strategy model emits all ten
actions in observation order.
"""

from __future__ import annotations

import argparse
from datetime import datetime
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
from courier.nlp import MissionParser, resolve
from courier.solver import OracleSolver, load_strategy


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
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("results") / ("submission_" + datetime.now().strftime("%Y%m%d_%H%M%S")) / "predictions.json"
    if args.out.exists():
        parser.error(f"output already exists: {args.out}; choose a new --out path")

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
    nlp = MissionParser()
    strategy = load_strategy(args.strategy_artifact)

    predictions: list[int] = []
    diagnostics: Counter[str] = Counter()
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

        parsed = nlp.parse(group[0]["mission"])
        mission = resolve(parsed, graph.landmarks)
        scene = graph.to_scene(scene_id, mission)
        try:
            results = strategy.predict_scene_with_diagnostics(scene)
            actions = tuple(int(result.action) for result in results)
            diagnostics["strategy_fallbacks"] += sum(result.fallback_reason is not None for result in results)
        except (KeyError, ValueError):
            diagnostics["emergency_scenes"] += 1
            actions = emergency_prediction(scene)
        predictions.extend(actions)

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
    counts = Counter(predictions)
    print(f"wrote {len(predictions)} predictions to {args.out.resolve()}")
    print(f"action counts: {dict(sorted(counts.items()))}")
    print(f"diagnostics: {dict(diagnostics)}")
    print(f"total seconds: {elapsed:.1f}")


if __name__ == "__main__":
    main()


