"""Rule-based mission parser: Vietnamese text to goal/via specs plus flags.

The parser never sees the map. Map-only descriptions ("the place furthest
north", "the place next to the library") are returned as specs without a
landmark type; `resolver.resolve` turns every spec into a node on the map.

Pipeline per mission:
  1. fold accents, tokenise, repair one-letter typos;
  2. tag place / person / item mentions by longest lexicon match, plus a
     head-noun + keyword fallback for places described by their function;
  3. per sentence, rewrite mentions as placeholders and read map-only
     descriptions and spatial qualifiers with regexes;
  4. give every remaining mention a role (negated, via, goal) from the cue
     words around it, then keep the last goal and via candidates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from . import lexicon as lx
from .text import Speller, split_sentences, tokenize, typo_sources

MAX_PHRASE = 8


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """What the text says about one destination, before looking at the map.

    `type` is None for map-only descriptions. `ref` is one of north/south/west/
    east/near/far (picks one of two copies of `type`) or north_most/south_most/
    west_most/east_most/anchor_near (picks any landmark). `anchor` is the landmark
    type that near/far/anchor_near measure from.
    """

    type: str | None
    ref: str | None = None
    anchor: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedMission:
    text: str
    goal: TargetSpec | None
    via: TargetSpec | None
    urgent: bool
    fragile: bool


@dataclass(slots=True)
class _Mention:
    kind: str  # place, person, item, or the ref name of a map-only description
    type: str | None
    start: int
    end: int
    role: str = "goal"  # goal, via, negated, anchor
    ref: str | None = None
    anchor: str | None = None


@dataclass(slots=True)
class _Lexicon:
    phrases: dict[tuple[str, ...], tuple[str, str | None]] = field(default_factory=dict)
    keywords: list[tuple[tuple[str, ...], str]] = field(default_factory=list)
    heads: frozenset[tuple[str, ...]] = frozenset()


@cache
def _lexicon() -> _Lexicon:
    lex = _Lexicon()
    for kind, table in (("place", lx.PLACE_ALIASES), ("person", lx.PERSON_ALIASES)):
        for place_type, phrases in table.items():
            for phrase in phrases:
                lex.phrases[tuple(phrase.split())] = (kind, place_type)
    for phrase in lx.ITEMS:
        lex.phrases.setdefault(tuple(phrase.split()), ("item", None))
    lex.keywords = sorted(
        ((tuple(word.split()), place_type) for place_type, words in lx.PLACE_KEYWORDS.items() for word in words),
        key=lambda pair: -len(pair[0]),
    )
    lex.heads = frozenset(tuple(head.split()) for head in lx.HEAD_NOUNS)
    return lex


def _ends_with(prefix: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?:^| )(?:{pattern})$", prefix) for pattern in patterns)


def _starts_with(suffix: str, patterns: tuple[str, ...]) -> bool:
    return any(re.match(rf"(?:{pattern})(?: |$)", suffix) for pattern in patterns)


def _phrase_alternation(phrases) -> str:
    return "|".join(sorted((re.escape(p) for p in phrases), key=len, reverse=True))


_DIRECTION = r"(bac|nam|tay|dong|tren|duoi|trai|phai)"
_EXTREME = {kind: _phrase_alternation(phrases) for kind, phrases in lx.EXTREME_PHRASES.items()}


class MissionParser:
    """Text -> ParsedMission. `vocabulary` holds train unigram/bigram counts for typo repair."""

    def __init__(self, vocabulary: dict | None = None) -> None:
        vocabulary = _bundled_vocabulary() if vocabulary is None else vocabulary
        words = dict(vocabulary.get("unigrams", {}))
        for word in lx.lexicon_words():
            words.setdefault(word, 1)
        bigrams = dict(vocabulary.get("bigrams", {}))
        for phrase in lx.lexicon_phrases():
            parts = phrase.split()
            for pair in zip(parts, parts[1:]):
                key = " ".join(pair)
                bigrams[key] = max(bigrams.get(key, 0), 5)
        self.speller = Speller(words, bigrams)

    def parse(self, text: str) -> ParsedMission:
        tokens = self.speller(tokenize(_mark_capital_breaks(text)))
        sentences = [clause for sentence in split_sentences(tokens) for clause in _split_clauses(sentence)]
        goals: list[_Mention] = []
        vias: list[_Mention] = []
        urgent: bool | None = None
        fragile: bool | None = None
        for sentence in sentences:
            mentions = _tag_mentions(sentence)
            for mention in _analyse_sentence(sentence, mentions):
                if mention.role == "goal":
                    goals.append(mention)
                elif mention.role == "via":
                    vias.append(mention)
            urgent = _flag(sentence, lx.URGENT_FORCE_TRUE, lx.URGENT_FORCE_FALSE, lx.URGENT_WORDS, urgent)
            fragile = _flag(sentence, (), lx.FRAGILE_FORCE_FALSE, lx.FRAGILE_WORDS, fragile)
        goal = _pick(goals)
        via = _pick(vias)
        if via is not None and goal is not None and goal.kind == "place" and (via.type, via.ref) == (goal.type, goal.ref):
            via = None
        return ParsedMission(
            text=text,
            goal=_spec(goal),
            via=_spec(via),
            urgent=bool(urgent),
            fragile=bool(fragile),
        )


_SENTENCE_PUNCTUATION = re.compile(r"[.!?;]")
_CAPITAL_BREAK = re.compile(r"(?<=[^\W\d_])\s+(?=[^\W\d_][^\W\d_])")


def _mark_capital_breaks(text: str) -> str:
    """Without any sentence punctuation, treat "word Capitalised" as a sentence break.

    Only used when the text has no . ! ? ; at all. Preserve names, generic
    descriptions and the links introducing destinations or spatial anchors.
    """
    if _SENTENCE_PUNCTUATION.search(text):
        return text

    tokens = tokenize(text)
    protected = set()
    for mention in _tag_mentions(tokens):
        protected.update(range(mention.start + 1, mention.end))

    def boundary(match: re.Match) -> str:
        following = text[match.end() : match.end() + 2]
        if not (following[0].isupper() and following[1].islower()):
            return match.group(0)
        index = len(tokenize(text[:match.end()]))
        if (
            index in protected
            or _starts_phrase(tokens, index, _lexicon())
            or (index and tokens[index - 1] in {
                "toi", "den", "qua", "ghe", "vao", "sang", "tai", "o", "cho",
                "gan", "xa", "canh", "sat", "voi", "ben", "ke",
            })
        ):
            return match.group(0)
        return ". "

    return _CAPITAL_BREAK.sub(boundary, text)


@cache
def _clause_starts() -> frozenset[tuple[str, ...]]:
    return frozenset(tuple(phrase.split()) for phrase in lx.CLAUSE_STARTS)


def _inside_known_phrase(tokens: list[str], i: int) -> bool:
    """True when a lexicon name crosses position i ("cang tin truoc": "tin truoc" is not a new clause)."""
    phrases = _lexicon().phrases
    for start in range(max(0, i - MAX_PHRASE + 1), i):
        for end in range(i + 1, min(len(tokens), start + MAX_PHRASE) + 1):
            if tuple(tokens[start:end]) in phrases:
                return True
    return False


def _split_clauses(tokens: list[str]) -> list[list[str]]:
    """Split a sentence before every clause-opening phrase (see lexicon.CLAUSE_STARTS)."""
    starts = _clause_starts()
    lengths = sorted({len(phrase) for phrase in starts}, reverse=True)
    clauses: list[list[str]] = [[]]
    for i, token in enumerate(tokens):
        if (
            clauses[-1]
            and any(tuple(tokens[i : i + n]) in starts for n in lengths)
            and not _inside_known_phrase(tokens, i)
        ):
            clauses.append([])
        clauses[-1].append(token)
    return [clause for clause in clauses if clause]


def _spec(mention: _Mention | None) -> TargetSpec | None:
    if mention is None:
        return None
    return TargetSpec(type=mention.type, ref=mention.ref, anchor=mention.anchor)


def _pick(candidates: list[_Mention]) -> _Mention | None:
    """Prefer named places and map descriptions over people; then the last one said."""
    strong = [m for m in candidates if m.kind != "person"]
    pool = strong or candidates
    return pool[-1] if pool else None


def _flag(
    tokens: list[str],
    force_true: tuple[str, ...],
    force_false: tuple[str, ...],
    words: tuple[str, ...],
    current: bool | None,
) -> bool | None:
    """Read one yes/no statement from a sentence; later statements override earlier ones."""
    if any(_contains_flag_phrase(tokens, phrase) for phrase in force_true):
        return True
    if any(_contains_flag_phrase(tokens, phrase) for phrase in force_false):
        return False
    for start in range(len(tokens)):
        for phrase in words:
            size = phrase.count(" ") + 1
            if " ".join(tokens[start : start + size]) == phrase:
                negated = any(token in lx.NEGATORS for token in tokens[max(0, start - 3) : start])
                return not negated
    return current


def _contains_flag_phrase(tokens: list[str], phrase: str) -> bool:
    """Match a flag phrase, tolerating one generator-style typo in a multiword cue.

    Some dropped letters form another valid vocabulary word (``cam`` for
    ``cham``, ``cang`` for ``chang``), so the conservative global speller must
    leave them alone.  The surrounding force phrase makes one fuzzy token
    unambiguous without weakening single-word cues such as ``gap`` or ``vo``.
    """
    expected = phrase.split()
    size = len(expected)
    for start in range(len(tokens) - size + 1):
        window = tokens[start : start + size]
        if window == expected:
            return True
        if size > 1:
            mismatches = [(found, wanted) for found, wanted in zip(window, expected) if found != wanted]
            if len(mismatches) == 1 and _is_generator_typo(*mismatches[0]):
                return True
    return False


def _is_generator_typo(found: str, expected: str) -> bool:
    """Whether `found` can be made by the generator corrupting `expected` once."""
    if len(found) + 1 == len(expected):
        # Restrict contextual repair to a missing internal character.  Dropping
        # the final character can turn unrelated short cues into each other
        # (for example ``vo`` and ``voi`` in fragility/urgency sentences).
        return (
            found[:1] == expected[:1]
            and found[-1:] == expected[-1:]
            and any(found == expected[:index] + expected[index + 1 :] for index in range(1, len(expected) - 1))
        )
    return (
        len(found) == len(expected)
        and found[:1] == expected[:1]
        and found[-1:] == expected[-1:]
        and found in typo_sources(expected)
    )


def _tag_mentions(tokens: list[str]) -> list[_Mention]:
    lex = _lexicon()
    mentions: list[_Mention] = []
    i = 0
    while i < len(tokens):
        hit = None
        for length in range(min(MAX_PHRASE, len(tokens) - i), 0, -1):
            entry = lex.phrases.get(tuple(tokens[i : i + length]))
            if entry is not None:
                hit = (length, entry)
                break
        generic = _generic_place(tokens, i, lex) if hit is None else None
        if generic is not None:
            mentions.append(generic)
            i = generic.end
            continue
        if hit is not None:
            length, (kind, place_type) = hit
            mentions.append(_Mention(kind=kind, type=place_type, start=i, end=i + length))
            i += length
            continue
        i += 1
    return mentions


_GENERIC_STOP = frozenset(
    "nhat hon gan sat canh ke xa dang lay nhan truoc roi giup nhe chu ma khong phia nam o tren duoi va hoac "
    "la cho voi man ria mep ben xong cach".split()
)
_NHAT = r"(?:nhat|nht|nat|naht|nhta|hnat|nhatt)"


def _starts_phrase(tokens: list[str], i: int, lex: _Lexicon) -> bool:
    """A place or person name starts here (items may sit inside a description, "noi nop ho so")."""
    for length in range(1, min(MAX_PHRASE, len(tokens) - i) + 1):
        entry = lex.phrases.get(tuple(tokens[i : i + length]))
        if entry is not None and entry[0] != "item":
            return True
    return False


def _generic_place(tokens: list[str], i: int, lex: _Lexicon) -> _Mention | None:
    """A head noun followed by a function keyword, e.g. "noi kham suc khoe"."""
    for head_len in (2, 1):
        head = tuple(tokens[i : i + head_len])
        if len(head) == head_len and head in lex.heads:
            break
    else:
        return None
    if tokens[i] == "cho" and i > 0 and tokens[i - 1] in {"dang", "van", "con", "doi", "ngoi"}:
        return None
    window_end = i + head_len
    while (
        window_end < len(tokens)
        and window_end - i - head_len < 5
        and tokens[window_end].isalpha()
        and tokens[window_end] not in _GENERIC_STOP
        and not _starts_phrase(tokens, window_end, lex)
    ):
        window_end += 1
    window = tokens[i + head_len : window_end]
    best = None
    for start in range(len(window)):
        for keyword, place_type in lex.keywords:
            if tuple(window[start : start + len(keyword)]) == keyword:
                end = start + len(keyword)
                if best is None or end > best[0]:
                    best = (end, place_type)
                break
        if best is not None:
            break
    if best is None:
        return None
    end, place_type = best
    # Swallow the rest of the descriptive phrase, e.g. "noi phuc vu bua trua" up to the stop word.
    return _Mention(kind="place", type=place_type, start=i, end=i + head_len + max(end, len(window)))


def _analyse_sentence(tokens: list[str], mentions: list[_Mention]) -> list[_Mention]:
    """Assign roles; returns the mentions that still matter, in reading order."""
    pieces: list[str] = []
    cursor = 0
    for index, mention in enumerate(mentions):
        pieces.extend(tokens[cursor : mention.start])
        if mention.kind == "item":
            pieces.append("hang")
        else:
            pieces.append(f"@{index}")
        cursor = mention.end
    pieces.extend(tokens[cursor:])
    text = " ".join(pieces)

    # Map-only descriptions become new mentions written as "#k".
    extra: list[_Mention] = []

    def add_map_ref(ref: str, anchor: _Mention | None) -> str:
        extra.append(_Mention(kind=ref, type=None, start=-1, end=-1, ref=ref, anchor=anchor.type if anchor else None))
        if anchor is not None:
            anchor.role = "anchor"
        return f"#{len(extra) - 1}"

    def anchor_near(match: re.Match) -> str:
        return add_map_ref("anchor_near", mentions[int(match.group(1))])

    text = re.sub(
        rf"\b{lx.GENERIC_PLACE}(?: nam| o| ma)? {lx.NEAR_LINK} @(\d+)(?: \w+){{0,2}} {_NHAT}\b", anchor_near, text
    )
    for ref, alternation in _EXTREME.items():
        text = re.sub(
            rf"\b{lx.GENERIC_PLACE}(?: nam| o| ma)?(?: o)? (?:{alternation})(?: (?:tren|cua|trong) ban do| ban do)?\b",
            lambda match, ref=ref: add_map_ref(ref, None),
            text,
        )

    # Spatial qualifiers on a named place: direction, or nearer/further than an anchor.
    def qualify(match: re.Match) -> str:
        mention = mentions[int(match.group(1))]
        if match.group(2):
            mention.ref = lx.DIRECTIONS[match.group(2)]
        elif match.group(4):
            anchor = mentions[int(match.group(4))]
            anchor.role = "anchor"
            mention.ref = "far" if match.group(3).startswith(("xa", "cach")) else "near"
            mention.anchor = anchor.type
        return match.group(0)

    text = re.sub(
        rf"@(\d+)(?:(?: o| nam| nam o)?(?: (?:phia|phai|pha|ben|mien|huong|khu vuc phia|goc|man|ve phia)) {_DIRECTION}\b"
        rf"|(?:,)?(?: o| nam| nam o)? ((?:gan|xa|canh|sat|cach xa|cach)(?: voi| hon)?) @(\d+)(?!(?: \w+)? nhat))",
        qualify,
        text,
    )

    tokens_out = text.split()
    result: list[_Mention] = []
    for position, token in enumerate(tokens_out):
        if not token.startswith(("@", "#")) or len(token) < 2:
            continue
        mention = mentions[int(token[1:])] if token[0] == "@" else extra[int(token[1:])]
        if mention.role == "anchor" or mention.kind == "item":
            continue
        prefix = " ".join(tokens_out[:position])
        suffix = " ".join(tokens_out[position + 1 :])
        if _ends_with(prefix, lx.NEGATE_BEFORE) or _starts_with(suffix, lx.NEGATE_AFTER):
            mention.role = "negated"
        elif mention.kind in ("place", "person") and (
            _ends_with(prefix, lx.VIA_BEFORE) or _starts_with(suffix.split(",")[0].strip(), lx.VIA_AFTER)
        ):
            mention.role = "via"
        else:
            mention.role = "goal"
        result.append(mention)
    return result


def parse_mission(text: str) -> ParsedMission:
    return _default_parser().parse(text)


@cache
def _bundled_vocabulary() -> dict:
    path = Path(__file__).with_name("vocab.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@cache
def _default_parser() -> MissionParser:
    return MissionParser()
