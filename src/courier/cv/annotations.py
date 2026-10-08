"""Load `scenes.json` with the image-side fields that `courier.common.Scene` drops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from courier.common.domain import HEADING_TO_ACTION, Edge

from .types import GridLayout, LegendEntry, LegendReading, SceneGraph


@dataclass(frozen=True, slots=True)
class Degradation:
    rotation_deg: float
    blur: float
    jpeg_quality: int | None


@dataclass(frozen=True, slots=True)
class SceneAnnotation:
    scene_id: str
    image_path: Path
    width: int
    height: int
    style: str
    graph: SceneGraph
    legend: LegendReading
    road_look: dict[str, str]  # true status -> status whose default look is drawn
    degradation: Degradation

    @property
    def legend_swapped(self) -> bool:
        return any(status != look for status, look in self.road_look.items())

    @classmethod
    def from_dict(cls, value: dict[str, Any], split_dir: Path) -> SceneAnnotation:
        grid = GridLayout(
            rows=value["grid"]["rows"],
            cols=value["grid"]["cols"],
            node_xy={tuple(node["rc"]): tuple(node["xy"]) for node in value["nodes"]},
        )
        graph = SceneGraph(
            grid=grid,
            edges=tuple(
                Edge(
                    a=tuple(edge["a"]),
                    b=tuple(edge["b"]),
                    status=edge["status"],
                    stairs=bool(edge["stairs"]),
                    oneway_to=tuple(edge["oneway_to"]) if edge["oneway_to"] is not None else None,
                )
                for edge in value["edges"]
            ),
            landmarks=tuple((landmark["type"], tuple(landmark["rc"])) for landmark in value["landmarks"]),
            robot_rc=tuple(value["robot"]["rc"]),
            robot_heading=HEADING_TO_ACTION[value["robot"]["heading"]],
            weather=value["weather"],
        )
        legend = LegendReading(
            entries=tuple(
                LegendEntry(kind=item["kind"], text=item["text"], swatch=tuple(item["swatch"]), label=tuple(item["label"]))
                for item in value["legend"]
            ),
            weather_box=tuple(value["weather_box"]) if value.get("weather_box") else None,
        )
        degradation = value["degradation"]
        return cls(
            scene_id=value["scene_id"],
            image_path=split_dir / value["image"],
            width=value["width"],
            height=value["height"],
            style=value["style"],
            graph=graph,
            legend=legend,
            road_look=dict(value["road_look"]),
            degradation=Degradation(
                rotation_deg=float(degradation["rotation_deg"]),
                blur=float(degradation["blur"]),
                jpeg_quality=degradation["jpeg_quality"],
            ),
        )


def load_annotations(data_root: str | Path, split: str) -> tuple[SceneAnnotation, ...]:
    split_dir = Path(data_root) / split
    raw = json.loads((split_dir / "scenes.json").read_text(encoding="utf-8"))
    return tuple(SceneAnnotation.from_dict(value, split_dir) for value in raw)


def scene_image_paths(data_root: str | Path, split: str) -> tuple[tuple[str, Path], ...]:
    """(scene_id, image path) per scene from observations.json; works for test, which has no scenes.json."""
    split_dir = Path(data_root) / split
    rows = json.loads((split_dir / "observations.json").read_text(encoding="utf-8"))
    result = []
    for index in range(0, len(rows), 10):
        scene_id = rows[index]["id"].rsplit("-R", 1)[0]
        result.append((scene_id, split_dir / rows[index]["image"]))
    return tuple(result)
