"""Validate predictions and compare changes, without reading test labels.

Score bounds require the number of scored scenes. Public scoring uses an
unknown subset and unscored scenes, so full-test bounds do not apply to it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> tuple[list[int], str]:
    raw = path.read_bytes()
    values = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(values, list):
        raise ValueError(f"{path}: expected a JSON list")
    if not values or len(values) % 10:
        raise ValueError(f"{path}: length must be positive and divisible by 10")
    bad = [i for i, v in enumerate(values) if type(v) is not int or v not in range(4)]
    if bad:
        raise ValueError(f"{path}: invalid integer action at index {bad[0]}")
    return values, hashlib.sha256(raw).hexdigest().upper()


def score_bound(candidate: list[int], reference: list[int], scored_scenes: int) -> float:
    """Worst possible macro change on an unknown subset of complete scenes."""
    if len(candidate) != len(reference):
        raise ValueError("lengths differ")
    if not 1 <= scored_scenes <= len(candidate) // 10:
        raise ValueError("invalid scored scene count")
    changes = [sum(a != b for a, b in zip(candidate[i:i + 10], reference[i:i + 10]))
               for i in range(0, len(candidate), 10)]
    return 100 * sum(sorted(changes, reverse=True)[:scored_scenes]) / (10 * scored_scenes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-score", type=float, help="reference accuracy in percent")
    parser.add_argument("--scored-scenes", type=int, help="known number of scored scenes; do not guess")
    parser.add_argument("--full-test-score", action="store_true", help="reference score covers every scene")
    parser.add_argument("--sample", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public/test/sample_submission.json"))
    args = parser.parse_args()
    try:
        candidate, candidate_sha = load(args.candidate)
        reference, reference_sha = load(args.reference)
        if len(candidate) != len(reference):
            raise ValueError("lengths differ; not comparable")
        if args.sample.exists():
            sample, _ = load(args.sample)
            if len(sample) != len(candidate):
                raise ValueError("candidate length differs from sample submission")
        if args.reference_score is not None and not 0 <= args.reference_score <= 100:
            raise ValueError("reference score must be in [0, 100]")
        if args.full_test_score and args.scored_scenes is not None:
            raise ValueError("choose full-test-score or scored-scenes")
        print(f"candidate {args.candidate}\nsha256 {candidate_sha} n={len(candidate)}")
        print(f"reference {args.reference}\nsha256 {reference_sha} n={len(reference)}")
        changed = [i for i, (a, b) in enumerate(zip(candidate, reference)) if a != b]
        by_robot = Counter(i % 10 for i in changed)
        print(f"changed predictions: {len(changed)}/{len(candidate)} ({100 * len(changed) / len(candidate):.4f}%)")
        print("changed per robot: " + "  ".join(f"R{r} {by_robot[r]}" for r in range(10)))
        print(f"changed scenes: {len({i // 10 for i in changed})}")
        print(f"action counts: {dict(sorted(Counter(candidate).items()))}")
        if args.reference_score is not None:
            count = len(candidate) // 10 if args.full_test_score else args.scored_scenes
            if count is None:
                print("PUBLIC SCORE BOUND UNKNOWN: public scoring uses a hidden subset; full-test disagreement is not a public bound.")
            else:
                bound = score_bound(candidate, reference, count)
                low, high = max(0, args.reference_score - bound), min(100, args.reference_score + bound)
                print(f"For {count} scored complete scenes on the same labels: [{low:.2f}%, {high:.2f}%]")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
