"""Parse every mission of a split (test included) into map-independent specs.

The output is one entry per scene, in observation order, holding the goal/via
TargetSpecs plus flags. Resolving them needs the landmarks CV reads from the
image, so that step happens downstream with `courier.nlp.resolve`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from courier.nlp import MissionParser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    observations = json.loads((args.data / args.split / "observations.json").read_text(encoding="utf-8"))
    mission_parser = MissionParser()
    scenes = []
    for row in observations[::10]:
        parsed = mission_parser.parse(row["mission"])
        record = dataclasses.asdict(parsed)
        record["id"] = row["id"].rsplit("-R", 1)[0]
        scenes.append(record)
    out = args.out or Path("data/nlp") / f"{args.split}_parsed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scenes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.split}: parsed {len(scenes)} missions -> {out}")


if __name__ == "__main__":
    main()
