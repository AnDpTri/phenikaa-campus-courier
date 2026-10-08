"""Score the parser on hand-written missions with phrasing absent from train/validation.

The set (tests/nlp/challenge_missions.json) was written from general Vietnamese,
not from test. Expected goal/via are [type, ref, anchor]; urgent/fragile are only
checked when given. Prints per-mission failures and the accuracy per field.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from courier.nlp import MissionParser, TargetSpec


def spec(value):
    return None if value is None else TargetSpec(*value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("tests/nlp/challenge_missions.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    items = json.loads(args.file.read_text(encoding="utf-8"))
    mission_parser = MissionParser()
    hits, totals = Counter(), Counter()
    for item in items:
        parsed = mission_parser.parse(item["text"])
        checks = {"goal": parsed.goal == spec(item["goal"]), "via": parsed.via == spec(item.get("via"))}
        for flag in ("urgent", "fragile"):
            if flag in item:
                checks[flag] = getattr(parsed, flag) == item[flag]
        checks["all"] = all(checks.values())
        for name, ok in checks.items():
            hits[name] += ok
            totals[name] += 1
        if not checks["all"] and not args.quiet:
            print(f"FAIL {item['text']}\n     got goal={parsed.goal} via={parsed.via} urgent={parsed.urgent} fragile={parsed.fragile}")
    print("  ".join(f"{name} {hits[name]}/{totals[name]}" for name in ("goal", "via", "urgent", "fragile", "all")))


if __name__ == "__main__":
    main()
