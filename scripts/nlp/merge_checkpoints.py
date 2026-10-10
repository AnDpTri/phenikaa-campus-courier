"""Combine per-seed checkpoints (--checkpoint-dir) into one ensemble artifact.

Use when a multi-seed run is stopped early: the finished seeds still make a usable model.

    py -3.12 scripts/nlp/merge_checkpoints.py results/nlp_v3_ckpt/seed_*_best.pt --out artifacts/nlp/neural_parser_v3_partial.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from courier.nlp.neural import NeuralMissionParser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    nets = [net for path in args.checkpoints for net in NeuralMissionParser.load(path).nets]
    NeuralMissionParser(nets).save(
        args.out,
        thresholds={"goal": 2.0, "via": 2.0, "flags": 2.0},
        merged_from=[str(path) for path in args.checkpoints],
    )
    print(f"saved {len(nets)} nets to {args.out}")


if __name__ == "__main__":
    main()
