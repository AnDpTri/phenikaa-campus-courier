"""Validate and render NLP v4 phrase expansions without editing synth.py.

The external agent produces JSON. This tool rejects unsafe or malformed phrases,
checks growth limits and duplicates, then renders an append-only Python block for
human review. It never writes to ``src/courier/nlp/synth.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from courier.nlp import lexicon, synth  # noqa: E402


TUPLE_BANKS = (
    "VIA_FRAMES_BEFORE",
    "VIA_FRAMES_AFTER",
    "ANCHOR_NEAR_FRAMES",
    "GOAL_FRAMES",
    "DISTRACTOR_FRAMES",
    "REDIRECT_PREFIX",
)
DICT_BANKS = ("MOST_PHRASES", "EXTRA_PLACES", "EXTRA_PERSONS")
ALL_BANKS = TUPLE_BANKS + DICT_BANKS

ALLOWED_FIELDS = {
    "VIA_FRAMES_BEFORE": {"W", "I2"},
    "VIA_FRAMES_AFTER": {"W", "I2"},
    "ANCHOR_NEAR_FRAMES": {"G", "A"},
    "GOAL_FRAMES": {"V", "I", "T", "L", "tail"},
    "DISTRACTOR_FRAMES": {"X"},
    "REDIRECT_PREFIX": {"X"},
}
REQUIRED_FIELDS = {
    "VIA_FRAMES_BEFORE": {"W"},
    "VIA_FRAMES_AFTER": {"W"},
    "ANCHOR_NEAR_FRAMES": {"G", "A"},
    "GOAL_FRAMES": {"L"},
    "DISTRACTOR_FRAMES": {"X"},
    "REDIRECT_PREFIX": {"X"},
}

SEMANTIC_CUES = {
    "VIA_FRAMES_BEFORE": re.compile(r"\b(truoc|dau tien|truoc het|ghe|qua|lay|nhan|buoc dau|xong|roi)\b"),
    "VIA_FRAMES_AFTER": re.compile(r"\b(truoc|ghe|qua|tat|dung|lay|tren duong|phai)\b"),
    "ANCHOR_NEAR_FRAMES": re.compile(
        r"\b(gan|sat|canh|ke|lien ke|it buoc|ngan nhat|hang xom|"
        r"khoang cach|cu ly|lo trinh|duong di|toi thieu|nho nhat)\b"
    ),
    "DISTRACTOR_FRAMES": re.compile(r"\b(khong|dung|nham|bo qua|tranh|dia chi cu|dong cua)\b"),
    "REDIRECT_PREFIX": re.compile(r"\b(huy|thay vao|khong phai|doi|bo|sua|nham|cap nhat|dinh chinh|quen)\b"),
}

DIRECTION_CUES = {
    "north_most": re.compile(r"\b(bac|tren|cao|dinh|hang dau)\b"),
    "south_most": re.compile(r"\b(nam|duoi|thap|day|hang cuoi)\b"),
    "west_most": re.compile(r"\b(tay|trai|cot dau|le trai|mat troi lan|kinh do nho)\b"),
    "east_most": re.compile(r"\b(dong|phai|cot cuoi|le phai|mat troi moc|kinh do lon)\b"),
}

ASCII_LOWER = re.compile(r"^[a-z0-9 (),.:;!?#\[\]'\"/-]+$")


@dataclass(frozen=True)
class Problem:
    path: str
    message: str


def normalized(value: str) -> str:
    return " ".join(value.strip().split())


def placeholders(value: str) -> set[str]:
    found = set()
    try:
        for _, field, _, _ in string.Formatter().parse(value):
            if field:
                found.add(field)
    except ValueError as error:
        raise ValueError(f"invalid format string: {error}") from error
    return found


def current_banks() -> dict[str, object]:
    result: dict[str, object] = {}
    base_sizes = getattr(synth, "V4_BASE_SIZES", {})
    for name in ALL_BANKS:
        value = getattr(synth, name)
        size = base_sizes.get(name)
        if isinstance(value, dict) and isinstance(size, dict):
            result[name] = {key: items[: size[key]] for key, items in value.items()}
        elif isinstance(size, int):
            result[name] = value[:size]
        else:
            result[name] = value
    return result


def existing_phrases(bank: str, key: str | None, current: Iterable[str]) -> set[str]:
    """Return every phrase already understood by the parser for this semantic slot."""
    values = {normalized(item) for item in current}
    if key is None:
        return values
    if bank == "EXTRA_PLACES":
        values.update(normalized(item) for item in lexicon.PLACE_ALIASES[key])
    elif bank == "EXTRA_PERSONS":
        values.update(normalized(item) for item in lexicon.PERSON_ALIASES[key])
    elif bank == "MOST_PHRASES":
        values.update(normalized(item) for item in lexicon.EXTREME_PHRASES[key])
    return values


def inventory() -> dict:
    result: dict[str, object] = {}
    for name, value in current_banks().items():
        if isinstance(value, dict):
            result[name] = {key: len(items) for key, items in value.items()}
        else:
            result[name] = len(value)
    return {"version": 1, "growth_range": [0.30, 0.50], "banks": result}


def validate_phrase(bank: str, phrase: object, path: str) -> list[Problem]:
    problems: list[Problem] = []
    if not isinstance(phrase, str):
        return [Problem(path, "phrase must be a string")]
    if phrase != normalized(phrase):
        problems.append(Problem(path, "phrase has leading, trailing or repeated whitespace"))
    if not phrase:
        problems.append(Problem(path, "phrase is empty"))
        return problems
    prose = re.sub(r"\{[A-Za-z0-9_]+\}", "", phrase)
    if prose.lower() != prose or not ASCII_LOWER.fullmatch(prose):
        problems.append(Problem(path, "use lowercase ASCII Vietnamese without diacritics"))
    if len(phrase) > 180:
        problems.append(Problem(path, "phrase exceeds 180 characters"))
    try:
        fields = placeholders(phrase)
    except ValueError as error:
        problems.append(Problem(path, str(error)))
        return problems
    if bank in ALLOWED_FIELDS:
        unexpected = fields - ALLOWED_FIELDS[bank]
        missing = REQUIRED_FIELDS[bank] - fields
        if unexpected:
            problems.append(Problem(path, f"unexpected placeholders: {sorted(unexpected)}"))
        if missing:
            problems.append(Problem(path, f"missing placeholders: {sorted(missing)}"))
    elif fields:
        problems.append(Problem(path, "dictionary phrases cannot contain placeholders"))
    cue = SEMANTIC_CUES.get(bank)
    if cue and not cue.search(phrase):
        problems.append(Problem(path, f"no required semantic cue for {bank}"))
    return problems


def validate_document(document: object, *, partial: bool = False) -> tuple[list[Problem], dict]:
    problems: list[Problem] = []
    if not isinstance(document, dict):
        return [Problem("$", "top level must be an object")], {}
    if document.get("version") != 1:
        problems.append(Problem("$.version", "must equal 1"))
    banks = document.get("banks")
    if not isinstance(banks, dict):
        return problems + [Problem("$.banks", "must be an object")], {}
    unknown = set(banks) - set(ALL_BANKS)
    for name in sorted(unknown):
        problems.append(Problem(f"$.banks.{name}", "bank is not allowed"))
    if not partial:
        for name in ALL_BANKS:
            if name not in banks:
                problems.append(Problem(f"$.banks.{name}", "bank is required in final mode"))

    existing = current_banks()
    candidate_seen: dict[str, str] = {}
    counts: dict[str, object] = {}

    for bank, additions in banks.items():
        if bank not in existing:
            continue
        current = existing[bank]
        if isinstance(current, dict):
            if not isinstance(additions, dict):
                problems.append(Problem(f"$.banks.{bank}", "must be an object keyed like the current bank"))
                continue
            bad_keys = set(additions) - set(current)
            for key in sorted(bad_keys):
                problems.append(Problem(f"$.banks.{bank}.{key}", "unknown key"))
            if not partial:
                for key in current:
                    if key not in additions:
                        problems.append(Problem(f"$.banks.{bank}.{key}", "key is required in final mode"))
            counts[bank] = {}
            for key, phrases in additions.items():
                path = f"$.banks.{bank}.{key}"
                if key not in current or not isinstance(phrases, list):
                    if key in current:
                        problems.append(Problem(path, "must be an array"))
                    continue
                counts[bank][key] = len(phrases)
                problems += validate_growth(path, len(current[key]), len(phrases), partial)
                problems += validate_list(
                    bank,
                    phrases,
                    path,
                    current[key],
                    candidate_seen,
                    existing_override=existing_phrases(bank, key, current[key]),
                )
                if bank == "MOST_PHRASES":
                    cue = DIRECTION_CUES[key]
                    for index, phrase in enumerate(phrases):
                        if isinstance(phrase, str) and not cue.search(phrase):
                            problems.append(Problem(f"{path}[{index}]", f"phrase lacks an unambiguous {key} cue"))
        else:
            path = f"$.banks.{bank}"
            if not isinstance(additions, list):
                problems.append(Problem(path, "must be an array"))
                continue
            counts[bank] = len(additions)
            problems += validate_growth(path, len(current), len(additions), partial)
            problems += validate_list(bank, additions, path, current, candidate_seen)
    return problems, counts


def validate_growth(path: str, current: int, added: int, partial: bool) -> list[Problem]:
    lower = max(1, int(current * 0.30 + 0.999999))
    upper = max(lower, int(current * 0.50))
    if added > upper:
        return [Problem(path, f"adds {added}; maximum is {upper} (50% of {current})")]
    if not partial and added < lower:
        return [Problem(path, f"adds {added}; minimum is {lower} (30% of {current})")]
    return []


def validate_list(
    bank: str,
    phrases: list,
    path: str,
    current: Iterable[str],
    candidate_seen: dict[str, str],
    existing_override: set[str] | None = None,
) -> list[Problem]:
    problems: list[Problem] = []
    existing = existing_override or {normalized(item) for item in current}
    for index, phrase in enumerate(phrases):
        item_path = f"{path}[{index}]"
        problems += validate_phrase(bank, phrase, item_path)
        if not isinstance(phrase, str):
            continue
        norm = normalized(phrase)
        if norm in existing:
            problems.append(Problem(item_path, "duplicates an existing phrase"))
        if norm in candidate_seen:
            problems.append(Problem(item_path, f"duplicates candidate at {candidate_seen[norm]}"))
        else:
            candidate_seen[norm] = item_path
    return problems


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render(document: dict) -> str:
    lines = ["# NLP v4 expansion; generated from validated candidates."]
    banks = document["banks"]
    for bank in TUPLE_BANKS:
        if bank not in banks:
            continue
        lines += [f"{bank} += ("]
        lines += [f"    {quote(item)}," for item in banks[bank]]
        lines += [")", ""]
    for bank in DICT_BANKS:
        for key, items in banks.get(bank, {}).items():
            lines += [f"{bank}[{quote(key)}] += ("]
            lines += [f"    {quote(item)}," for item in items]
            lines += [")", ""]
    return "\n".join(lines).rstrip() + "\n"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="print current bank sizes and growth targets")
    inv.add_argument("--out", type=Path)
    val = sub.add_parser("validate", help="validate an agent-produced candidate JSON")
    val.add_argument("candidate", type=Path)
    val.add_argument("--partial", action="store_true", help="allow incomplete banks and less than 30% growth")
    rnd = sub.add_parser("render", help="render append-only Python after strict validation")
    rnd.add_argument("candidate", type=Path)
    rnd.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.command == "inventory":
        text = json.dumps(inventory(), indent=2, ensure_ascii=True) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return

    document = load_json(args.candidate)
    problems, counts = validate_document(document, partial=getattr(args, "partial", False))
    if problems:
        for problem in problems:
            print(f"ERROR {problem.path}: {problem.message}")
        raise SystemExit(f"rejected with {len(problems)} problem(s)")
    print("VALID " + json.dumps(counts, sort_keys=True))
    if args.command == "render":
        text = render(document)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"rendered {args.out}")
        else:
            print(text, end="")


if __name__ == "__main__":
    main()
