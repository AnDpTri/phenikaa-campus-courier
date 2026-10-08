"""Read-only NLP benchmark and controlled robustness checks on labelled data."""

import dataclasses
import json
import random
import re
from collections import Counter
from pathlib import Path

from courier.common import load_dataset
from courier.nlp import MissionParser, resolve, resolve_target
from courier.nlp.text import fold
from scripts.nlp.evaluate_nlp import compare


OUT = Path("results/nlp_check_20261008")
DATA = Path("Phenikaa_Campus_Courier_2026_v3/delivery_public")
parser = MissionParser()


def damage(text, seed):
    rng = random.Random(seed)
    def word(match):
        value = match.group()
        if len(value) < 4 or rng.random() >= 0.05:
            return value
        index = rng.randrange(1, len(value) - 1)
        if rng.random() < 0.5:
            return value[:index] + value[index + 1:]
        return value[:index] + value[index + 1] + value[index] + value[index + 2:]
    return re.sub(r"[a-z]+", word, fold(text))


report = {"notes": [
    "Official metrics use oracle landmarks and the unmodified parser/resolver.",
    "Stress checks transform train/validation only, retain intended labels, and are not estimates of test accuracy.",
    "Test audit is used only as aggregate diagnostic evidence; no test text is inspected or used to add rules.",
]}
for split in ("train", "validation"):
    dataset = load_dataset(DATA, split)
    entries = {}
    for variant in ("original", "accent_folded", "no_sentence_punctuation", "extra_typos_5pct"):
        correct, flags = Counter(), Counter()
        for index, scene in enumerate(dataset.scenes):
            text = scene.mission.text
            if variant == "accent_folded":
                text = fold(text)
            elif variant == "no_sentence_punctuation":
                text = re.sub(r"[.!?;()\[\]]", " ", text)
            elif variant == "extra_typos_5pct":
                text = damage(text, 20261008 + index)
            parsed = parser.parse(text)
            predicted = resolve(parsed, scene.landmarks)
            correct.update(k for k, hit in compare(scene, predicted).items() if hit)
            raw = resolve_target(parsed.goal, scene.landmarks) if parsed.goal else None
            flags["no_parsed_goal"] += parsed.goal is None
            flags["goal_replaced"] += raw is None or not any(kind == raw[0] for kind, _ in scene.landmarks)
            flags["via_dropped"] += parsed.via is not None and predicted.via is None
        entries[variant] = {
            "scenes": len(dataset.scenes), "correct_counts": dict(correct),
            "accuracy": {k: v / len(dataset.scenes) for k, v in correct.items()},
            "warnings": dict(flags),
        }
        print(split, variant, json.dumps(entries[variant]), flush=True)
    report[split] = entries

original = json.loads(Path("src/courier/nlp/vocab.json").read_text(encoding="utf-8"))
rebuilt = json.loads((OUT / "vocab_rebuilt_from_train.json").read_text(encoding="utf-8"))
report["vocabulary_equals_rebuilt_train_content"] = original == rebuilt

with (OUT / "metrics.json").open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
