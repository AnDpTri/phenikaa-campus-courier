from __future__ import annotations

import unittest

import numpy as np

from courier.cv.style import StyleRoutedPipeline, style_features
from courier.cv.types import CVInput


class _Fixed:
    def __init__(self, style: str) -> None:
        self.style = style

    def predict(self, image) -> str:
        return self.style


class _Pipeline:
    def __init__(self, name: str) -> None:
        self.name = name

    def extract(self, cv_input) -> str:
        return self.name


class StyleTests(unittest.TestCase):
    def test_features_have_a_fixed_length(self) -> None:
        small = np.zeros((40, 60, 3), dtype=np.uint8)
        large = np.full((900, 1200, 3), 200, dtype=np.uint8)
        self.assertEqual(style_features(small).shape, style_features(large).shape)

    def test_routing(self) -> None:
        cv_input = CVInput("scene", np.zeros((8, 8, 3), dtype=np.uint8))
        routes = {"print": _Pipeline("print")}
        printed = StyleRoutedPipeline(_Fixed("print"), _Pipeline("default"), routes)
        classic = StyleRoutedPipeline(_Fixed("classic"), _Pipeline("default"), routes)
        self.assertEqual(printed.extract(cv_input), "print")
        self.assertEqual(classic.extract(cv_input), "default")
        self.assertEqual(classic.last_style, "classic")


if __name__ == "__main__":
    unittest.main()
