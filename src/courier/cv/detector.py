"""Keypoint detector for the whole map: node centres, legend road swatches, weather icon.

A small encoder-decoder predicts CenterNet-style heatmaps at 1/4 of a letterboxed
512 x 512 input, plus box sizes for swatches and the weather icon. Decoding turns
peaks back into original image coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

INPUT = 512
STRIDE = 4
OUT = INPUT // STRIDE
LEGEND_KINDS = ("normal", "crowded", "covered", "closed", "stairs", "oneway", "robot")
SWATCH_HEATMAPS = tuple(f"swatch_{kind}" for kind in LEGEND_KINDS)
HEATMAPS = ("node", *SWATCH_HEATMAPS, "weather")
SIZE_CHANNELS = 2  # box width, height in output cells, read at swatch / weather peaks


def _conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DetectorNet(nn.Module):
    def __init__(self, width: int = 24) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(_conv(3, w, 2), _conv(w, w))  # 1/2
        self.down1 = nn.Sequential(_conv(w, 2 * w, 2), _conv(2 * w, 2 * w))  # 1/4
        self.down2 = nn.Sequential(_conv(2 * w, 4 * w, 2), _conv(4 * w, 4 * w))  # 1/8
        self.down3 = nn.Sequential(_conv(4 * w, 4 * w, 2), _conv(4 * w, 4 * w), _conv(4 * w, 4 * w))  # 1/16
        self.up2 = _conv(8 * w, 4 * w)
        self.up1 = _conv(6 * w, 2 * w)
        self.heat = nn.Conv2d(2 * w, len(HEATMAPS), 1)
        self.size = nn.Conv2d(2 * w, SIZE_CHANNELS, 1)
        self.heat.bias.data.fill_(-2.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s4 = self.down1(self.stem(x))
        s8 = self.down2(s4)
        s16 = self.down3(s8)
        u8 = self.up2(torch.cat([s8, F.interpolate(s16, scale_factor=2, mode="nearest")], 1))
        u4 = self.up1(torch.cat([s4, F.interpolate(u8, scale_factor=2, mode="nearest")], 1))
        return self.heat(u4), self.size(u4)


def letterbox(image: np.ndarray, input_size: int = INPUT) -> tuple[np.ndarray, float]:
    """Resize to a square canvas, padding bottom/right with the median edge colour."""
    height, width = image.shape[:2]
    scale = input_size / max(height, width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    out = np.empty((input_size, input_size, 3), dtype=np.uint8)
    out[:] = np.median(resized.reshape(-1, 3), axis=0).astype(np.uint8)
    out[: resized.shape[0], : resized.shape[1]] = resized
    return out, scale


@dataclass(frozen=True, slots=True)
class Peak:
    x: float  # original image pixels
    y: float
    score: float
    w: float = 0.0
    h: float = 0.0


def find_peaks(heat: np.ndarray, threshold: float, radius: int = 1) -> list[tuple[int, int, float]]:
    """Local maxima above threshold as (row, col, score) in output cells."""
    t = torch.from_numpy(heat)[None, None]
    pooled = F.max_pool2d(t, 2 * radius + 1, stride=1, padding=radius)[0, 0].numpy()
    rows, cols = np.nonzero((heat >= pooled) & (heat >= threshold))
    return [(int(r), int(c), float(heat[r, c])) for r, c in zip(rows, cols)]


def refine(heat: np.ndarray, r: int, c: int) -> tuple[float, float]:
    """Sub-cell peak position from the 3x3 neighbourhood centroid."""
    r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, heat.shape[0]), max(c - 1, 0), min(c + 2, heat.shape[1])
    patch = heat[r0:r1, c0:c1]
    total = float(patch.sum()) or 1.0
    rr, cc = np.mgrid[r0:r1, c0:c1]
    return float((patch * rr).sum() / total), float((patch * cc).sum() / total)


class KeypointDetector:
    def __init__(
        self,
        net: DetectorNet,
        threshold: float = 0.3,
        channel_thresholds: dict[str, float] | None = None,
        input_size: int = INPUT,
    ) -> None:
        if input_size <= 0 or input_size % 16:
            raise ValueError("detector input_size must be a positive multiple of 16")
        self.net = net.eval()
        self.threshold = threshold
        self.channel_thresholds = channel_thresholds or {}
        self.input_size = input_size

    @torch.no_grad()
    def predict_maps(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        boxed, scale = letterbox(image, self.input_size)
        device = next(self.net.parameters()).device
        x = torch.from_numpy(boxed).permute(2, 0, 1)[None].float().div_(255.0).to(device)
        heat, size = self.net(x)
        return torch.sigmoid(heat)[0].cpu().numpy(), size[0].cpu().numpy(), scale

    def detect(self, image: np.ndarray) -> dict[str, list[Peak]]:
        heat, size, scale = self.predict_maps(image)
        to_px = STRIDE / scale
        result: dict[str, list[Peak]] = {}
        for channel, name in enumerate(HEATMAPS):
            peaks = []
            threshold = self.channel_thresholds.get(name, self.threshold)
            for r, c, score in find_peaks(heat[channel], threshold):
                rr, cc = refine(heat[channel], r, c)
                # Cell (i, j) covers input pixels [4j, 4j + 4); its centre is 4j + 1.5.
                x, y = (cc * STRIDE + 1.5) / scale, (rr * STRIDE + 1.5) / scale
                peaks.append(Peak(x, y, score, float(size[0, r, c]) * to_px, float(size[1, r, c]) * to_px))
            result[name] = sorted(peaks, key=lambda p: -p.score)
        return result
