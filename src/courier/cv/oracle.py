"""Ground-truth stages for ablations: replace any subset of learned stages with annotations.

Only usable on train/validation, where scenes.json exists.
"""

from __future__ import annotations

from collections.abc import Iterable

from courier.common.domain import Edge

from .annotations import SceneAnnotation
from .types import CVInput, GridLayout, LegendReading, NodeContents


class _OracleBase:
    def __init__(self, annotations: Iterable[SceneAnnotation]):
        self._by_id = {annotation.scene_id: annotation for annotation in annotations}

    def _get(self, cv_input: CVInput) -> SceneAnnotation:
        try:
            return self._by_id[cv_input.scene_id]
        except KeyError:
            raise KeyError(f"{cv_input.scene_id}: no annotation for oracle stage") from None


class OracleLegendReader(_OracleBase):
    def read(self, cv_input: CVInput) -> LegendReading:
        return self._get(cv_input).legend


class OracleWeatherClassifier(_OracleBase):
    def classify(self, cv_input: CVInput, legend: LegendReading) -> str:
        return self._get(cv_input).graph.weather


class OracleGridDetector(_OracleBase):
    def detect(self, cv_input: CVInput, legend: LegendReading) -> GridLayout:
        return self._get(cv_input).graph.grid


class OracleEdgeClassifier(_OracleBase):
    def classify(self, cv_input: CVInput, grid: GridLayout, legend: LegendReading) -> tuple[Edge, ...]:
        return self._get(cv_input).graph.edges


class OracleNodeClassifier(_OracleBase):
    def classify(self, cv_input: CVInput, grid: GridLayout, legend: LegendReading) -> NodeContents:
        graph = self._get(cv_input).graph
        return NodeContents(robot_rc=graph.robot_rc, robot_heading=graph.robot_heading, landmarks=graph.landmarks)
