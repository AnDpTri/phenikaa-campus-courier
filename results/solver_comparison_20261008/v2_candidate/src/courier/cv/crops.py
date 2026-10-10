"""Crop geometry shared by training and inference of the node, edge and swatch classifiers.

Every crop is scaled by the grid spacing, so one model serves every image size.
Labels are defined here too, so the training script and the stages agree.
"""

from __future__ import annotations

import numpy as np

from courier.common.domain import Action, Edge, NodeRC

from .imaging import crop_edge_strip, crop_square
from .types import LANDMARK_TYPES, XY, Box

NODE_SIZE = 48
NODE_SIDE = 0.9  # crop side as a fraction of the grid spacing
EDGE_SIZE = (96, 24)  # width (along the road) x height
EDGE_THICKNESS = 0.32

# Node classes: empty, ten landmark types, robot with each heading.
NODE_CLASSES = ("empty", *LANDMARK_TYPES, *(f"robot_{action.name}" for action in Action))
ROBOT_CLASS_OFFSET = 1 + len(LANDMARK_TYPES)

# Edge "look": how the road is drawn, before the legend says which status that look means.
LOOK_CLASSES = ("none", "normal", "crowded", "covered", "closed")
ONEWAY_CLASSES = ("none", "to_b", "to_a")


def node_crop(image: np.ndarray, center: XY, spacing: float) -> np.ndarray:
    return crop_square(image, center, NODE_SIDE * spacing, NODE_SIZE)


def edge_crop(image: np.ndarray, xy_a: XY, xy_b: XY, spacing: float) -> np.ndarray:
    return crop_edge_strip(image, xy_a, xy_b, EDGE_THICKNESS * spacing, EDGE_SIZE)


def swatch_crop(image: np.ndarray, box: Box, spacing: float) -> np.ndarray:
    """A legend road swatch, cut like an edge running left to right through its middle."""
    x1, y1, x2, y2 = box
    cy = (y1 + y2) / 2
    return crop_edge_strip(image, (x1, cy), (x2, cy), EDGE_THICKNESS * spacing, EDGE_SIZE)


def node_label(rc: NodeRC, landmarks: dict[NodeRC, str], robot_rc: NodeRC, heading: Action) -> int:
    if rc == robot_rc:
        return ROBOT_CLASS_OFFSET + int(heading)
    if rc in landmarks:
        return 1 + LANDMARK_TYPES.index(landmarks[rc])
    return 0


def edge_labels(edge: Edge | None, a: NodeRC, b: NodeRC, road_look: dict[str, str]) -> tuple[int, int, int]:
    """(look, stairs, oneway) for the ordered pair a -> b."""
    if edge is None:
        return 0, 0, 0
    look = "closed" if edge.status == "closed" else road_look.get(edge.status, edge.status)
    oneway = 0 if edge.oneway_to is None else (1 if edge.oneway_to == b else 2)
    return LOOK_CLASSES.index(look), int(edge.stairs), oneway
