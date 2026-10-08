"""Artificial stress tests for the NLP parser on train/validation.

Each variant rewrites the mission text, keeps the annotated mission, and scores
parse + resolve against ground-truth landmarks (as evaluate_nlp.py does).
Variants reuse labels; removing punctuation can make the text ambiguous. These
tests probe sensitivity, not test accuracy or proven label preservation.
"""

from __future__ import annotations

import argparse
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from courier.common import load_dataset
from courier.nlp import MissionParser, resolve

if __package__:
    from .evaluate_nlp import compare
else:
    from evaluate_nlp import compare


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def strip_punctuation(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.!?;()\[\]]", " ", text)).strip()


def typos(rate: float):
    def apply(text: str, rng: random.Random) -> str:
        words = []
        for word in text.split(" "):
            if len(word) >= 4 and rng.random() < rate:
                i = rng.randrange(1, len(word) - 1)
                word = word[:i] + word[i + 1 :] if rng.random() < 0.5 else word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]
            words.append(word)
        return " ".join(words)

    return apply


VARIANTS = {
    "original": lambda text, rng: text,
    "no_accents": lambda text, rng: strip_accents(text),
    "no_punctuation": lambda text, rng: strip_punctuation(text),
    "lowercase_no_punct": lambda text, rng: strip_punctuation(text).lower(),
    "typos_5pct": typos(0.05),
    "typos_15pct": typos(0.15),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--splits", default="train,validation")
    args = parser.parse_args()
    splits = args.splits.split(",")
    if not splits or any(split not in {"train", "validation"} for split in splits):
        parser.error("--splits accepts only train and validation; test must not be used for stress evaluation")
    mission_parser = MissionParser()
    for split in splits:
        dataset = load_dataset(args.data, split)
        for name, variant in VARIANTS.items():
            rng = random.Random(f"{args.seed}:{split}:{name}")
            totals: Counter[str] = Counter()
            for scene in dataset.scenes:
                text = variant(scene.mission.text, rng)
                parsed = mission_parser.parse(text)
                result = compare(scene, resolve(parsed, scene.landmarks))
                totals.update(key for key, ok in result.items() if ok)
                totals["no_goal"] += parsed.goal is None
            n = len(dataset.scenes)
            print(
                f"{split:10s} {name:18s} all {totals['all'] / n:.4f}  goal {totals['goal_nodes'] / n:.4f}"
                f"  via {totals['via_nodes'] / n:.4f}  urgent {totals['urgent'] / n:.4f}"
                f"  fragile {totals['fragile'] / n:.4f}  no-goal {totals['no_goal'] / n:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
