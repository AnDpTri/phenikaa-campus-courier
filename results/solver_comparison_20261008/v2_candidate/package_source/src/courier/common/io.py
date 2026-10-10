"""Dataset loading with alignment checks between observations, labels, and scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .domain import Scene


@dataclass(frozen=True, slots=True)
class Dataset:
    split: str
    observations: tuple[dict, ...]
    labels: tuple[int, ...] | None
    scenes: tuple[Scene, ...] | None


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(data_root: str | Path, split: str, *, require_scenes: bool = True) -> Dataset:
    split_dir = Path(data_root) / split
    observations = tuple(_read_json(split_dir / "observations.json"))
    labels_path = split_dir / "labels.json"
    scenes_path = split_dir / "scenes.json"
    labels = tuple(_read_json(labels_path)) if labels_path.exists() else None
    scenes = tuple(Scene.from_dict(raw) for raw in _read_json(scenes_path)) if scenes_path.exists() else None

    if labels is not None and len(labels) != len(observations):
        raise ValueError(f"{split}: labels and observations have different lengths")
    if require_scenes and scenes is None:
        raise ValueError(f"{split}: scenes.json is required for oracle mode")
    if scenes is not None:
        if len(observations) != 10 * len(scenes):
            raise ValueError(f"{split}: expected exactly ten observations per scene")
        for index, scene in enumerate(scenes):
            rows = observations[index * 10 : index * 10 + 10]
            if tuple(row["robot_id"] for row in rows) != tuple(range(10)):
                raise ValueError(f"{split}/{scene.scene_id}: robot rows are not ordered 0..9")
            if any(row["mission"] != scene.mission.text for row in rows):
                raise ValueError(f"{split}/{scene.scene_id}: mission alignment error")
    return Dataset(split=split, observations=observations, labels=labels, scenes=scenes)

