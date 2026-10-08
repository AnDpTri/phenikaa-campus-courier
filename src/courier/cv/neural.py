"""CNN-backed CV stages: detector for legend/grid, crop classifiers for edges and nodes.

One forward pass of the detector serves both the legend reader and the grid
detector; `SharedDetector` caches it per scene.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from courier.common.domain import Action, Edge

from .crops import (
    LOOK_CLASSES,
    NODE_CLASSES,
    ROBOT_CLASS_OFFSET,
    edge_crop,
    node_crop,
    swatch_crop,
)
from .detector import KeypointDetector, Peak
from .grid_fit import fit_grid
from .nets import load_net, to_tensor
from .types import LANDMARK_TYPES, LEGEND_ROAD_ORDER, ROAD_KINDS, CVInput, GridLayout, LegendEntry, LegendReading, NodeContents


class SharedDetector:
    def __init__(
        self,
        path: str | Path,
        threshold: float = 0.4,
        semantic_threshold: float = 0.15,
        device: str = "auto",
    ) -> None:
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.node_threshold = threshold
        channel_thresholds = {
            **{f"swatch_{kind}": semantic_threshold for kind in LEGEND_ROAD_ORDER},
            "weather": semantic_threshold,
        }
        net = load_net(path, device="cpu" if resolved == "auto" else resolved)
        self.detector = KeypointDetector(
            net,
            threshold,
            channel_thresholds,
            input_size=int(getattr(net, "artifact_meta", {}).get("input_size", 512)),
        )
        self._key: str | None = None
        self._peaks: dict[str, list[Peak]] = {}

    def peaks(self, cv_input: CVInput) -> dict[str, list[Peak]]:
        if self._key != cv_input.scene_id:
            self._peaks = self.detector.detect(cv_input.image)
            self._key = cv_input.scene_id
        return self._peaks


class NeuralLegendReader:
    """Read class-specific legend swatches and the weather icon."""

    def __init__(self, shared: SharedDetector) -> None:
        self.shared = shared

    def read(self, cv_input: CVInput) -> LegendReading:
        peaks = self.shared.peaks(cv_input)
        entries = []
        for kind in LEGEND_ROAD_ORDER:
            candidates = peaks[f"swatch_{kind}"]
            if candidates:
                peak = candidates[0]
                box = (peak.x - peak.w / 2, peak.y - peak.h / 2, peak.x + peak.w / 2, peak.y + peak.h / 2)
                entries.append(LegendEntry(kind=kind, text="", swatch=box, label=box))
        weather_box = None
        if peaks["weather"]:
            p = peaks["weather"][0]
            # The detector's box regression is slightly conservative. This
            # expansion is robust to blur/JPEG and better matches the weather
            # classifier's training crops.
            scale = 1.3
            weather_box = (
                p.x - p.w * scale / 2,
                p.y - p.h * scale / 2,
                p.x + p.w * scale / 2,
                p.y + p.h * scale / 2,
            )
        return LegendReading(entries=tuple(entries), weather_box=weather_box)


class NeuralGridDetector:
    def __init__(self, shared: SharedDetector) -> None:
        self.shared = shared

    def detect(self, cv_input: CVInput, legend: LegendReading) -> GridLayout:
        nodes = self.shared.peaks(cv_input)["node"]
        grid = fit_grid([(p.x, p.y, p.score) for p in nodes])
        if 5 <= grid.rows <= 9 and 5 <= grid.cols <= 9:
            return grid
        # A single low-confidence false peak can create an extra row/column.
        # Tighten only invalid grids, preserving normal scenes unchanged.
        for threshold in np.arange(self.shared.node_threshold + 0.05, 0.96, 0.05):
            filtered = [(p.x, p.y, p.score) for p in nodes if p.score >= threshold]
            if len(filtered) < 4:
                break
            candidate = fit_grid(filtered)
            if 5 <= candidate.rows <= 9 and 5 <= candidate.cols <= 9:
                return candidate
        return grid


@torch.no_grad()
def _run(net, crops: list[np.ndarray]):
    return net(to_tensor(np.stack(crops)))


class NeuralEdgeClassifier:
    def __init__(self, path: str | Path) -> None:
        self.net = load_net(path)

    def look_to_status(self, cv_input: CVInput, legend: LegendReading, spacing: float) -> dict[str, str]:
        """Which status each road look means in this image, read from the legend swatches.

        The three swatches are assigned to the three looks jointly (best permutation),
        falling back to the style's default when the legend was not found.
        """
        entries = [legend.entry(kind) for kind in ROAD_KINDS]
        if any(entry is None for entry in entries):
            return {kind: kind for kind in ROAD_KINDS}
        logits = _run(self.net, [swatch_crop(cv_input.image, entry.swatch, spacing) for entry in entries])[0]
        logp = torch.log_softmax(logits[:, [LOOK_CLASSES.index(k) for k in ROAD_KINDS]], 1).numpy()
        best = max(itertools.permutations(range(3)), key=lambda perm: sum(logp[i, perm[i]] for i in range(3)))
        # Status ROAD_KINDS[i] is drawn with look ROAD_KINDS[best[i]].
        return {ROAD_KINDS[best[i]]: ROAD_KINDS[i] for i in range(3)}

    def classify(self, cv_input: CVInput, grid: GridLayout, legend: LegendReading) -> tuple[Edge, ...]:
        spacing = grid.spacing()
        pairs = grid.adjacent_pairs()
        if not pairs:
            return ()
        mapping = self.look_to_status(cv_input, legend, spacing)
        crops = [edge_crop(cv_input.image, grid.node_xy[a], grid.node_xy[b], spacing) for a, b in pairs]
        look, stairs, oneway = (t.argmax(1).numpy() for t in _run(self.net, crops))
        edges = []
        for (a, b), lk, st, ow in zip(pairs, look, stairs, oneway):
            look_name = LOOK_CLASSES[lk]
            if look_name == "none":
                continue
            status = "closed" if look_name == "closed" else mapping[look_name]
            oneway_to = None if ow == 0 else (b if ow == 1 else a)
            edges.append(Edge(a=a, b=b, status=status, stairs=bool(st), oneway_to=oneway_to))
        return tuple(edges)


class NeuralNodeClassifier:
    def __init__(self, path: str | Path) -> None:
        self.net = load_net(path)

    def classify(self, cv_input: CVInput, grid: GridLayout, legend: LegendReading) -> NodeContents:
        spacing = grid.spacing()
        nodes = sorted(grid.node_xy)
        crops = [node_crop(cv_input.image, grid.node_xy[rc], spacing) for rc in nodes]
        probs = torch.softmax(_run(self.net, crops), 1).numpy()
        robot_scores = probs[:, ROBOT_CLASS_OFFSET:].sum(1)
        robot_index = int(np.argmax(robot_scores))
        heading = Action(int(np.argmax(probs[robot_index, ROBOT_CLASS_OFFSET:])))
        landmarks = []
        for index, rc in enumerate(nodes):
            if index == robot_index:
                continue
            label = int(np.argmax(probs[index, :ROBOT_CLASS_OFFSET]))
            if label > 0:
                landmarks.append((LANDMARK_TYPES[label - 1], rc))
        return NodeContents(robot_rc=nodes[robot_index], robot_heading=heading, landmarks=tuple(landmarks))


assert NODE_CLASSES[ROBOT_CLASS_OFFSET] == "robot_UP"
