"""Run the CV pipeline on train/validation and report per-stage accuracy.

Each stage is chosen independently, so one learned stage can be measured with
the others held at ground truth, e.g. `--grid learned` once it exists.
"""

from __future__ import annotations

import argparse
import json
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
from courier.cv.neural import (
    NeuralEdgeClassifier,
    NeuralGridDetector,
    NeuralLegendReader,
    NeuralNodeClassifier,
    SharedDetector,
)

# stage name -> implementation name -> factory(annotations)
STAGES = {
    "legend": {"oracle": OracleLegendReader, "learned": NeuralLegendReader},
    "weather": {"oracle": OracleWeatherClassifier, "learned": SklearnWeatherClassifier},
    "grid": {"oracle": OracleGridDetector, "learned": NeuralGridDetector},
    "edges": {"oracle": OracleEdgeClassifier, "learned": NeuralEdgeClassifier},
    "nodes": {"oracle": OracleNodeClassifier, "learned": NeuralNodeClassifier},
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
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/cv"), help="detector/edge/node nets")
    parser.add_argument("--detector-artifact", type=Path, help="override artifacts/cv/detector.pt")
    parser.add_argument("--detector-threshold", type=float, default=0.4, help="node heatmap threshold")
    parser.add_argument("--semantic-threshold", type=float, default=0.15, help="legend/weather heatmap threshold")
    parser.add_argument("--detector-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report", type=Path, help="write a new machine-readable JSON report")
    parser.add_argument("--all", choices=("oracle", "learned"), help="set every stage at once")
    for stage, options in STAGES.items():
        parser.add_argument(f"--{stage}", choices=tuple(options), default="oracle")
    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error(f"report already exists: {args.report}")
    if args.all:
        for stage in STAGES:
            setattr(args, stage, args.all)

    annotations = load_annotations(args.data, args.split)[: args.limit]
    shared = None
    implementations = {}
    for stage in STAGES:
        choice = getattr(args, stage)
        if choice == "oracle":
            implementations[stage] = STAGES[stage][choice](annotations)
        elif stage == "weather":
            implementations[stage] = SklearnWeatherClassifier.load(args.weather_artifact)
        elif stage in ("legend", "grid"):
            shared = shared or SharedDetector(
                args.detector_artifact or args.artifacts / "detector.pt",
                threshold=args.detector_threshold,
                semantic_threshold=args.semantic_threshold,
                device=args.detector_device,
            )
            implementations[stage] = STAGES[stage][choice](shared)
        else:
            implementations[stage] = STAGES[stage][choice](args.artifacts / f"{stage[:-1]}_net.pt")
    pipeline = CVPipeline(**implementations)

    report = CVReport()
    by_style: dict[str, CVReport] = {}
    invalid = Counter()
    invalid_details = []
    started = time.perf_counter()
    for annotation in annotations:
        cv_input = CVInput(scene_id=annotation.scene_id, image=load_rgb(annotation.image_path))
        graph = pipeline.extract(cv_input)
        problems = validate_graph(graph)
        invalid[bool(problems)] += 1
        if problems:
            invalid_details.append((annotation.scene_id, problems))
        report.add(graph, annotation.graph)
        by_style.setdefault(annotation.style, CVReport()).add(graph, annotation.graph)
    elapsed = time.perf_counter() - started

    print("stages:", {stage: getattr(args, stage) for stage in STAGES})
    print(f"scenes: {len(annotations)}  invalid_graphs: {invalid[True]}  sec/scene: {elapsed / max(len(annotations), 1):.3f}")
    if invalid_details:
        print("invalid_details:", invalid_details)
    summary = report.summary()
    style_summaries = {style: style_report.summary() for style, style_report in sorted(by_style.items())}
    for name, value in summary.items():
        print(f"  {name:22s} {value:.4f}")
    print("scene_exact_by_style:", {style: round(values["scene_exact"], 4) for style, values in style_summaries.items()})
    print("metrics_by_style:")
    for style, values in style_summaries.items():
        print(f"  {style}: { {name: round(value, 4) for name, value in values.items()} }")
    if args.report is not None:
        payload = {
            "split": args.split,
            "scenes": len(annotations),
            "stages": {stage: getattr(args, stage) for stage in STAGES},
            "detector_artifact": str(args.detector_artifact or args.artifacts / "detector.pt"),
            "detector_threshold": args.detector_threshold,
            "semantic_threshold": args.semantic_threshold,
            "invalid_graphs": invalid[True],
            "seconds": elapsed,
            "metrics": summary,
            "metrics_by_style": style_summaries,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print("report:", args.report)


if __name__ == "__main__":
    main()
