"""Run the CV pipeline on train/validation and report per-stage accuracy.

Each stage is chosen independently, so one learned stage can be measured with
the others held at ground truth, e.g. `--grid learned` once it exists.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from courier.cv import (
    CVInput,
    CVPipeline,
    CVReport,
    OracleEdgeClassifier,
    OracleGridDetector,
    OracleLegendReader,
    OracleNodeClassifier,
    OracleWeatherClassifier,
    SklearnWeatherClassifier,
    load_annotations,
    load_rgb,
    validate_graph,
)

# stage name -> implementation name -> factory(annotations)
STAGES = {
    "legend": {"oracle": OracleLegendReader},
    "weather": {"oracle": OracleWeatherClassifier, "learned": SklearnWeatherClassifier},
    "grid": {"oracle": OracleGridDetector},
    "edges": {"oracle": OracleEdgeClassifier},
    "nodes": {"oracle": OracleNodeClassifier},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--weather-artifact",
        type=Path,
        default=Path("artifacts/cv/weather_classifier.joblib"),
    )
    for stage, options in STAGES.items():
        parser.add_argument(f"--{stage}", choices=tuple(options), default="oracle")
    args = parser.parse_args()

    annotations = load_annotations(args.data, args.split)[: args.limit]
    implementations = {}
    for stage in STAGES:
        choice = getattr(args, stage)
        if stage == "weather" and choice == "learned":
            implementations[stage] = SklearnWeatherClassifier.load(args.weather_artifact)
        else:
            implementations[stage] = STAGES[stage][choice](annotations)
    pipeline = CVPipeline(**implementations)

    report = CVReport()
    by_style: dict[str, CVReport] = {}
    invalid = Counter()
    started = time.perf_counter()
    for annotation in annotations:
        cv_input = CVInput(scene_id=annotation.scene_id, image=load_rgb(annotation.image_path))
        graph = pipeline.extract(cv_input)
        invalid[bool(validate_graph(graph))] += 1
        report.add(graph, annotation.graph)
        by_style.setdefault(annotation.style, CVReport()).add(graph, annotation.graph)
    elapsed = time.perf_counter() - started

    print("stages:", {stage: getattr(args, stage) for stage in STAGES})
    print(f"scenes: {len(annotations)}  invalid_graphs: {invalid[True]}  sec/scene: {elapsed / max(len(annotations), 1):.3f}")
    for name, value in report.summary().items():
        print(f"  {name:22s} {value:.4f}")
    print("scene_exact_by_style:", {style: round(r.summary()["scene_exact"], 4) for style, r in sorted(by_style.items())})


if __name__ == "__main__":
    main()
