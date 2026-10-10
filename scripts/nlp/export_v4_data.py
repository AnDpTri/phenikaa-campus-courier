"""Export synthetic data and reproducible random samples; never train a model."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from courier.nlp.synth import Generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=350000)
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "train.jsonl"
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}")
    generator = Generator(args.seed, holdout=False)
    sample_seed = 20261009
    indices = set(random.Random(sample_seed).sample(range(args.count), min(20, args.count)))
    seen = set()
    counts = {key: Counter() for key in ("goal_type", "goal_ref", "via_type", "via_ref", "urgent", "fragile")}
    samples = []
    rejected = 0
    digest = hashlib.sha256()
    with target.open("xb") as handle:
        for index in range(args.count):
            while True:
                example = generator.example()
                signature = hashlib.sha256(example.text.lower().encode()).digest()
                if signature not in seen:
                    seen.add(signature)
                    break
                rejected += 1
            row = {"id": f"v4_{index:07d}", **asdict(example)}
            assert row["text"].strip() and "{" not in row["text"] and "}" not in row["text"]
            assert row["goal"] is not None
            for role in ("goal", "via"):
                spec = row[role]
                counts[role + "_type"][str(spec["type"] if spec else None)] += 1
                counts[role + "_ref"][str(spec["ref"] if spec else None)] += 1
            for flag in ("urgent", "fragile"):
                counts[flag][str(row[flag])] += 1
            payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
            if index in indices:
                samples.append(row)
            if (index + 1) % 50000 == 0:
                print(f"Exported {index + 1}/{args.count}", flush=True)
    report = {
        "count": args.count, "seed": args.seed, "holdout": False,
        "sampling_seed": sample_seed, "sample_count": len(samples),
        "unique_case_insensitive_texts": len(seen), "duplicates_skipped": rejected,
        "sha256": digest.hexdigest(), "bytes": target.stat().st_size,
        "generator_sha256": hashlib.sha256((ROOT / "src/courier/nlp/synth.py").read_bytes()).hexdigest(),
        "distribution": counts,
        "note": "Labels come from generator construction; schema checks do not prove semantic correctness. No training performed.",
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "random_samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Random samples from the exported training data", "", f"Sampling seed: {sample_seed}. Text is unchanged; intentional typos are preserved.", ""]
    for row in samples:
        lines += [f"## {row['id']}", "", row["text"], "", f"goal={row['goal']}; via={row['via']}; urgent={row['urgent']}; fragile={row['fragile']}", ""]
    (args.out / "random_samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
