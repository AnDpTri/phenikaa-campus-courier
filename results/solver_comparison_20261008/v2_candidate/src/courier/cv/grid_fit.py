"""Turn detected node centres into a (row, col) grid.

The map may be rotated by a few degrees and intersections are jittered, but
every row and column of the grid holds at least one node (checked on all of
train and validation), so rows and columns are the clusters of the de-rotated
coordinates.
"""

from __future__ import annotations

import math

import numpy as np

from .types import GridLayout


def _dominant_angle(points: np.ndarray) -> float:
    """Grid rotation in radians, from the directions to each point's nearest neighbours."""
    diff = points[:, None, :] - points[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    angles = []
    for i in range(len(points)):
        for j in np.argsort(dist[i])[:2]:
            dx, dy = diff[j, i]
            angle = math.atan2(dy, dx)
            angle = (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4  # fold to [-45, 45) degrees
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def _clusters(values: np.ndarray, gap: float) -> np.ndarray:
    """Index of the 1-D cluster each value falls in; a new cluster starts after a gap."""
    order = np.argsort(values)
    labels = np.empty(len(values), dtype=int)
    current = 0
    for rank, index in enumerate(order):
        if rank and values[index] - values[order[rank - 1]] > gap:
            current += 1
        labels[index] = current
    return labels


def fit_grid(points: list[tuple[float, float, float]]) -> GridLayout:
    """points: (x, y, score) in image pixels."""
    if len(points) < 4:
        raise ValueError(f"only {len(points)} node candidates")
    xy = np.asarray([(x, y) for x, y, _ in points], dtype=np.float64)
    scores = np.asarray([s for _, _, s in points])
    theta = _dominant_angle(xy)
    cos, sin = math.cos(-theta), math.sin(-theta)
    rotated = xy @ np.array([[cos, sin], [-sin, cos]])
    dist = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(dist, np.inf)
    spacing = float(np.median(dist.min(1)))
    rows = _clusters(rotated[:, 1], 0.5 * spacing)
    cols = _clusters(rotated[:, 0], 0.5 * spacing)
    node_xy: dict[tuple[int, int], tuple[float, float]] = {}
    best: dict[tuple[int, int], float] = {}
    for i in range(len(points)):
        rc = (int(rows[i]), int(cols[i]))
        if scores[i] > best.get(rc, -1.0):
            best[rc] = scores[i]
            node_xy[rc] = (float(xy[i, 0]), float(xy[i, 1]))
    return GridLayout(rows=int(rows.max()) + 1, cols=int(cols.max()) + 1, node_xy=node_xy)
