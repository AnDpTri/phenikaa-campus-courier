"""Image loading and geometry-aware crops shared by every CV stage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .types import XY, Box


def load_rgb(path: str | Path) -> np.ndarray:
    """Palette PNGs and JPEGs alike become H x W x 3 uint8 RGB."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def crop_box(image: np.ndarray, box: Box, pad: float = 0.0, fill: int = 0) -> np.ndarray:
    """Axis-aligned crop; regions outside the image are filled instead of shrinking the crop."""
    x1, y1, x2, y2 = box
    left, top = int(round(x1 - pad)), int(round(y1 - pad))
    right, bottom = int(round(x2 + pad)), int(round(y2 + pad))
    height, width = image.shape[:2]
    out = np.full((max(bottom - top, 1), max(right - left, 1), image.shape[2]), fill, dtype=image.dtype)
    src_x1, src_y1 = max(left, 0), max(top, 0)
    src_x2, src_y2 = min(right, width), min(bottom, height)
    if src_x2 > src_x1 and src_y2 > src_y1:
        out[src_y1 - top : src_y2 - top, src_x1 - left : src_x2 - left] = image[src_y1:src_y2, src_x1:src_x2]
    return out


def crop_square(image: np.ndarray, center: XY, side: float, out_size: int) -> np.ndarray:
    """Square of `side` pixels around `center`, resampled to `out_size` x `out_size`."""
    scale = out_size / side
    cx, cy = center
    matrix = np.array(
        [[scale, 0.0, out_size / 2 - scale * cx], [0.0, scale, out_size / 2 - scale * cy]],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        image, matrix, (out_size, out_size), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_edge_strip(
    image: np.ndarray,
    xy_a: XY,
    xy_b: XY,
    thickness: float,
    out_size: tuple[int, int],
    margin: float = 0.0,
) -> np.ndarray:
    """Strip along segment a->b, rotated so a is on the left and b on the right.

    `thickness` is the strip height in source pixels; `margin` trims (positive)
    or extends (negative) each end, in source pixels. Output is `out_size` = (width, height).
    Because a always maps to the left, the direction of a one-way arrow is
    expressed relative to (a, b) and survives image rotation.
    """
    ax, ay = xy_a
    bx, by = xy_b
    length = float(np.hypot(bx - ax, by - ay))
    if length == 0:
        raise ValueError("edge endpoints coincide")
    ux, uy = (bx - ax) / length, (by - ay) / length
    span = length - 2 * margin
    out_w, out_h = out_size
    sx, sy = out_w / span, out_h / thickness
    # Source point p maps to (sx * <p - start, u>, sy * <p - start, n> + out_h / 2), n = u rotated +90 deg.
    start_x, start_y = ax + ux * margin, ay + uy * margin
    nx, ny = -uy, ux
    matrix = np.array(
        [
            [sx * ux, sx * uy, -sx * (ux * start_x + uy * start_y)],
            [sy * nx, sy * ny, -sy * (nx * start_x + ny * start_y) + out_h / 2],
        ],
        dtype=np.float64,
    )
    return cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
