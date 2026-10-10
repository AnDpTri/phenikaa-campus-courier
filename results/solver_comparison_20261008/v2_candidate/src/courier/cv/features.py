"""Fixed image features used by the lightweight learned CV stages."""

from __future__ import annotations

import cv2
import numpy as np


WEATHER_FEATURE_VERSION = 1
WEATHER_INPUT_SIZE = 48
WEATHER_THUMBNAIL_SIZE = 12

_WEATHER_HOG = cv2.HOGDescriptor(
    (WEATHER_INPUT_SIZE, WEATHER_INPUT_SIZE),
    (16, 16),
    (8, 8),
    (8, 8),
    9,
)


def extract_weather_features(image: np.ndarray) -> np.ndarray:
    """Return HOG, HSV histogram and coarse RGB pixels for a weather crop."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected H x W x 3 RGB image, got {image.shape}")
    resized = cv2.resize(image, (WEATHER_INPUT_SIZE, WEATHER_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)

    hog = _WEATHER_HOG.compute(gray).ravel().astype(np.float32) / 255.0
    histogram_parts = []
    for channel, upper in ((0, 180), (1, 256), (2, 256)):
        values = cv2.calcHist([hsv], [channel], None, [16], [0, upper]).ravel()
        histogram_parts.append(values)
    histogram = np.concatenate(histogram_parts).astype(np.float32)
    histogram /= float(histogram.sum()) + 1e-8

    thumbnail = cv2.resize(
        resized,
        (WEATHER_THUMBNAIL_SIZE, WEATHER_THUMBNAIL_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32).ravel() / 255.0
    return np.concatenate((hog, histogram, thumbnail)).astype(np.float32)
