"""Text normalisation: accent folding, tokenisation, and typo repair."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

_TOKEN = re.compile(r"[a-z0-9]+|[.,!?;:()\[\]]")
SENTENCE_BREAKS = frozenset(".!?;()[]")


def fold(text: str) -> str:
    """Lowercase and drop Vietnamese diacritics so accented and unaccented text match."""
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(fold(text))


def edit_distance(a: str, b: str, limit: int = 2) -> int:
    """Optimal-string-alignment distance (adjacent swaps cost 1), capped at limit + 1."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return min(prev[-1], limit + 1)


def typo_sources(token: str) -> set[str]:
    """Words that `token` could be a typo of: one letter dropped, or two adjacent letters swapped.

    These are the only corruptions the mission generator applies; substitutions never occur.
    """
    sources = {token[:i] + token[i + 1] + token[i] + token[i + 2 :] for i in range(len(token) - 1)}
    sources.discard(token)
    return sources


class Speller:
    """Repair typos by choosing, among words one corruption away, the best supported one.

    Support is the frequency of the bigrams the candidate would form with its
    neighbours, then its unigram frequency. Out-of-vocabulary tokens are always
    repaired when a candidate exists; an in-vocabulary token is replaced only when a
    candidate forms far better attested bigrams with its neighbours, which catches
    drops that land on another real word ("van hong khoa" -> "van phong khoa").
    """

    def __init__(self, unigrams: Mapping[str, int], bigrams: Mapping[str, int] | None = None) -> None:
        self.unigrams = dict(unigrams)
        self.bigrams = dict(bigrams or {})
        self._by_skeleton: dict[str, set[str]] = {}
        for word in self.unigrams:
            # Index every word by itself with one letter removed, so dropped-letter typos are O(1).
            for i in range(len(word)):
                self._by_skeleton.setdefault(word[:i] + word[i + 1 :], set()).add(word)

    def candidates(self, token: str) -> set[str]:
        found = set(self._by_skeleton.get(token, ()))
        found.update(word for word in typo_sources(token) if word in self.unigrams)
        found.discard(token)
        return found

    def _support(self, prev: str | None, word: str, nxt: str | None) -> int:
        return self.bigrams.get(f"{prev} {word}", 0) + self.bigrams.get(f"{word} {nxt}", 0)

    def _fits_better(self, prev: str | None, best: str, token: str, nxt: str | None) -> bool:
        """True when `best` forms a well-attested bigram and beats `token` tenfold on every side.

        Train text contains the generator's typos too, so a real-word typo still forms a
        few bigrams; the margin keeps those from protecting it.
        """
        sides = [(f"{prev} {best}", f"{prev} {token}")] if prev and prev.isalpha() else []
        if nxt and nxt.isalpha():
            sides.append((f"{best} {nxt}", f"{token} {nxt}"))
        return any(self.bigrams.get(good, 0) >= 10 for good, _ in sides) and all(
            self.bigrams.get(good, 0) >= 10 * self.bigrams.get(bad, 0) for good, bad in sides
        )

    def __call__(self, tokens: list[str]) -> list[str]:
        out = list(tokens)
        for i, token in enumerate(tokens):
            if len(token) < 2 or not token.isalpha():
                continue
            prev = out[i - 1] if i > 0 else None
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            options = self.candidates(token)
            if not options:
                continue
            best = max(options, key=lambda word: (self._support(prev, word, nxt), self.unigrams.get(word, 0), word))
            if token not in self.unigrams:
                out[i] = best
            elif len(token) >= 3 and self._fits_better(prev, best, token, nxt):
                out[i] = best
        return out


def split_sentences(tokens: list[str]) -> list[list[str]]:
    sentences: list[list[str]] = [[]]
    for token in tokens:
        if token in SENTENCE_BREAKS:
            if sentences[-1]:
                sentences.append([])
        else:
            sentences[-1].append(token)
    return [sentence for sentence in sentences if sentence]
