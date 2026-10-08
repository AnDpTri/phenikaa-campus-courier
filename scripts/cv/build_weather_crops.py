"""Export weather-icon crops and metadata in the layout train_weather.py reads.

Writes data/cv/weather/<split>/<scene_id>.png and data/cv/weather/metadata_<split>.json,
cut exactly as SklearnWeatherClassifier cuts at inference (weather box + 4 px).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from courier.cv import crop_box, load_annotations, load_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--out", type=Path, default=Path("data/cv"))
    parser.add_argument("--pad", type=float, default=4.0)
    args = parser.parse_args()
    for split in ("train", "validation"):
        rows = []
        folder = args.out / "weather" / split
        folder.mkdir(parents=True, exist_ok=True)
        for annotation in load_annotations(args.data, split):
            patch = crop_box(load_rgb(annotation.image_path), annotation.legend.weather_box, pad=args.pad)
            name = Path("weather") / split / f"{annotation.scene_id}.png"
            Image.fromarray(patch).save(args.out / name)
            rows.append({"scene_id": annotation.scene_id, "image": name.as_posix(), "label": annotation.graph.weather})
        (args.out / "weather" / f"metadata_{split}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"{split}: {len(rows)} weather crops")


if __name__ == "__main__":
    main()
