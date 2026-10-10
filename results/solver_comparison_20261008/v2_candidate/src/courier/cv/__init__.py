"""Computer-vision workstream: map image to SceneGraph.

Layout:
  types.py        intermediate results (LegendReading, GridLayout, NodeContents, SceneGraph)
  annotations.py  scenes.json with image-side fields (node xy, legend boxes, road_look, degradation)
  imaging.py      image loading and geometry-aware crops (node squares, rotated edge strips)
  features.py     fixed feature extractors for lightweight learned stages
  learned.py      artifact-backed learned stage implementations
  stages.py       one Protocol per stage: legend, weather, grid, edges, nodes
  oracle.py       ground-truth stage implementations, for ablating one stage at a time
  pipeline.py     CVPipeline composing the stages, plus structural validation
  metrics.py      per-stage and per-scene accuracy against annotations
"""

from .annotations import Degradation, SceneAnnotation, load_annotations, scene_image_paths
from .imaging import crop_box, crop_edge_strip, crop_square, load_rgb
from .learned import SklearnWeatherClassifier
from .metrics import CVReport
from .oracle import (
    OracleEdgeClassifier,
    OracleGridDetector,
    OracleLegendReader,
    OracleNodeClassifier,
    OracleWeatherClassifier,
)
from .pipeline import CVPipeline, validate_graph
from .types import (
    LANDMARK_TYPES,
    LEGEND_ROAD_ORDER,
    ROAD_KINDS,
    WEATHERS,
    CVInput,
    GridLayout,
    LegendEntry,
    LegendReading,
    NodeContents,
    SceneGraph,
    edge_key,
)

__all__ = [
    "LANDMARK_TYPES",
    "LEGEND_ROAD_ORDER",
    "ROAD_KINDS",
    "WEATHERS",
    "CVInput",
    "CVPipeline",
    "CVReport",
    "Degradation",
    "GridLayout",
    "LegendEntry",
    "LegendReading",
    "NodeContents",
    "OracleEdgeClassifier",
    "OracleGridDetector",
    "OracleLegendReader",
    "OracleNodeClassifier",
    "OracleWeatherClassifier",
    "SceneAnnotation",
    "SceneGraph",
    "SklearnWeatherClassifier",
    "crop_box",
    "crop_edge_strip",
    "crop_square",
    "edge_key",
    "load_annotations",
    "load_rgb",
    "scene_image_paths",
    "validate_graph",
]
