"""Evaluate detector grid reconstruction over several node thresholds in one pass."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from courier.cv import load_annotations, load_rgb
from courier.cv.detector import STRIDE, KeypointDetector, find_peaks, refine
from courier.cv.grid_fit import fit_grid
from courier.cv.nets import load_net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--detector", type=Path, default=Path("artifacts/cv/detector.pt"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error(f"output already exists: {args.out}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    net = load_net(args.detector, device=device)
    input_size = int(getattr(net, "artifact_meta", {}).get("input_size", 512))
    detector = KeypointDetector(net, input_size=input_size)
    annotations = load_annotations(args.data, args.split)
    totals: dict[float, Counter] = {threshold: Counter() for threshold in args.thresholds}
    styles: dict[float, dict[str, Counter]] = {
        threshold: defaultdict(Counter) for threshold in args.thresholds
    }
    started = time.perf_counter()

    for annotation in annotations:
        heat, _, scale = detector.predict_maps(load_rgb(annotation.image_path))
        node_heat = heat[0]
        for threshold in args.thresholds:
            points = []
            for row, col, score in find_peaks(node_heat, threshold):
                refined_row, refined_col = refine(node_heat, row, col)
                points.append(
                    (
                        (refined_col * STRIDE + 1.5) / scale,
                        (refined_row * STRIDE + 1.5) / scale,
                        score,
                    )
                )
            try:
                grid = fit_grid(points)
                shape = (grid.rows, grid.cols) == (annotation.graph.grid.rows, annotation.graph.grid.cols)
                nodes = shape and grid.nodes == annotation.graph.grid.nodes
            except ValueError:
                shape = nodes = False
            totals[threshold]["scenes"] += 1
            totals[threshold]["grid_shape"] += shape
            totals[threshold]["node_set"] += nodes
            styles[threshold][annotation.style]["scenes"] += 1
            styles[threshold][annotation.style]["grid_shape"] += shape
            styles[threshold][annotation.style]["node_set"] += nodes

    report = {
        "split": args.split,
        "detector": str(args.detector),
        "input_size": input_size,
        "seconds": time.perf_counter() - started,
        "thresholds": {},
    }
    for threshold in args.thresholds:
        count = totals[threshold]["scenes"]
        entry = {
            "grid_shape": totals[threshold]["grid_shape"] / count,
            "node_set": totals[threshold]["node_set"] / count,
            "by_style": {},
        }
        for style, values in sorted(styles[threshold].items()):
            style_count = values["scenes"]
            entry["by_style"][style] = {
                "count": style_count,
                "grid_shape": values["grid_shape"] / style_count,
                "node_set": values["node_set"] / style_count,
            }
        report["thresholds"][str(threshold)] = entry
        print(threshold, json.dumps(entry, ensure_ascii=False), flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("report:", args.out)


if __name__ == "__main__":
    main()
