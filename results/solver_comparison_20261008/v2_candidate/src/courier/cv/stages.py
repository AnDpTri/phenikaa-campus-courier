"""Stage interfaces. Each stage can be swapped independently (oracle, heuristic, learned)."""

from __future__ import annotations

from typing import Protocol

from courier.common.domain import Edge

from .types import CVInput, GridLayout, LegendReading, NodeContents


class LegendReader(Protocol):
    """Locate legend rows (swatch + label boxes) and the weather icon."""

    def read(self, cv_input: CVInput) -> LegendReading: ...


class WeatherClassifier(Protocol):
    """Return "dry" or "rain"."""

    def classify(self, cv_input: CVInput, legend: LegendReading) -> str: ...


class GridDetector(Protocol):
    """Find intersection centres and assign each one a (row, col)."""

    def detect(self, cv_input: CVInput, legend: LegendReading) -> GridLayout: ...


class EdgeClassifier(Protocol):
    """Decide which adjacent pairs are roads, with status, stairs and one-way direction."""

    def classify(self, cv_input: CVInput, grid: GridLayout, legend: LegendReading) -> tuple[Edge, ...]: ...


class NodeClassifier(Protocol):
    """Find the robot (position + heading) and every landmark (type + position)."""

    def classify(self, cv_input: CVInput, grid: GridLayout, legend: LegendReading) -> NodeContents: ...
