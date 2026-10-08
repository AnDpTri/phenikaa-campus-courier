"""Cut labelled node, edge and legend-swatch crops from annotated images.

Output: data/cv/crops_<split>.npz with uint8 crops and integer labels. Train
crops get a small random jitter of centre and scale, so the classifiers tolerate
the imprecision of detected node centres.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from courier.cv import load_annotations, load_rgb
from courier.cv.annotations import SceneAnnotation
from courier.cv.crops import edge_crop, edge_labels, node_crop, node_label, swatch_crop
from courier.cv.types import ROAD_KINDS, edge_key


def scene_crops(job: tuple[SceneAnnotation, int, float]):
    annotation, seed, jitter = job
    rng = np.random.default_rng(seed)
    image = load_rgb(annotation.image_path)
    graph = annotation.graph
    spacing = graph.grid.spacing()
    xy = graph.grid.node_xy

    def moved(point):
        if not jitter:
            return point
        return (point[0] + rng.normal(0, jitter), point[1] + rng.normal(0, jitter))

    def scaled():
        return spacing * (1 + rng.uniform(-0.08, 0.08)) if jitter else spacing

    landmarks = {rc: kind for kind, rc in graph.landmarks}
    nodes, node_y = [], []
    for rc in sorted(xy):
        nodes.append(node_crop(image, moved(xy[rc]), scaled()))
        node_y.append(node_label(rc, landmarks, graph.robot_rc, graph.robot_heading))

    edges_by_key = {edge_key(edge): edge for edge in graph.edges}
    edges, edge_y = [], []
    for a, b in graph.grid.adjacent_pairs():
        if jitter and rng.random() < 0.5:
            a, b = b, a  # both orientations, so one-way labels see both directions
        edge = edges_by_key.get(tuple(sorted((a, b))))
        edges.append(edge_crop(image, moved(xy[a]), moved(xy[b]), scaled()))
        edge_y.append(edge_labels(edge, a, b, annotation.road_look))

    swatches, swatch_y = [], []
    for kind in ROAD_KINDS:
        entry = annotation.legend.entry(kind)
        if entry is not None:
            swatches.append(swatch_crop(image, entry.swatch, scaled()))
            swatch_y.append(("none", "normal", "crowded", "covered", "closed").index(annotation.road_look[kind]))
    return nodes, node_y, edges, edge_y, swatches, swatch_y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--out", type=Path, default=Path("data/cv"))
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--jitter", type=float, default=None, help="centre jitter in px (default: 2 for train, 0 otherwise)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jitter = args.jitter if args.jitter is not None else (2.0 if args.split == "train" else 0.0)
    annotations = load_annotations(args.data, args.split)
    jobs = [(annotation, index, jitter) for index, annotation in enumerate(annotations)]
    parts: dict[str, list] = {name: [] for name in ("nodes", "node_y", "edges", "edge_y", "swatches", "swatch_y", "scene")}
    with ProcessPoolExecutor(args.workers) as pool:
        for index, result in enumerate(pool.map(scene_crops, jobs, chunksize=8)):
            nodes, node_y, edges, edge_y, swatches, swatch_y = result
            parts["nodes"].extend(nodes)
            parts["node_y"].extend(node_y)
            parts["edges"].extend(edges)
            parts["edge_y"].extend(edge_y)
            parts["swatches"].extend(swatches)
            parts["swatch_y"].extend(swatch_y)
            parts["scene"].extend([index] * len(nodes))
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"crops_{args.split}.npz"
    np.savez(
        out,
        nodes=np.stack(parts["nodes"]),
        node_y=np.asarray(parts["node_y"], dtype=np.int64),
        node_scene=np.asarray(parts["scene"], dtype=np.int64),
        edges=np.stack(parts["edges"]),
        edge_y=np.asarray(parts["edge_y"], dtype=np.int64),
        swatches=np.stack(parts["swatches"]),
        swatch_y=np.asarray(parts["swatch_y"], dtype=np.int64),
    )
    print(f"{args.split}: {len(parts['nodes'])} nodes, {len(parts['edges'])} edges, {len(parts['swatches'])} swatches -> {out}")


if __name__ == "__main__":
    main()
