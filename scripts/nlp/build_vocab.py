"""Build the speller vocabulary (word and bigram frequencies) from train mission text.

Typos in the generator are one-off, so frequent tokens are real words. A token
that sits one edit away from a much more frequent token is treated as a typo
and dropped, unless the lexicon uses it. Only the train split is read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from courier.nlp.lexicon import lexicon_words
from courier.nlp.text import edit_distance, tokenize

DEFAULT_OUT = Path(__file__).parents[2] / "src" / "courier" / "nlp" / "vocab.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scenes = json.loads((args.data / "train" / "scenes.json").read_text(encoding="utf-8"))
    counts = Counter(token for scene in scenes for token in tokenize(scene["mission"]["text"]) if token.isalpha())
    frequent = sorted((word for word, count in counts.items() if count >= args.min_count), key=counts.__getitem__, reverse=True)
    protected = lexicon_words()
    texts = [tokenize(scene["mission"]["text"]) for scene in scenes]
    kept: dict[str, int] = {}
    for word in frequent:
        if word not in protected and any(counts[other] >= 10 * counts[word] and edit_distance(word, other, 1) <= 1 for other in kept):
            continue
        kept[word] = counts[word]
    bigrams = Counter(
        f"{a} {b}" for tokens in texts for a, b in zip(tokens, tokens[1:]) if a in kept and b in kept
    )
    bigrams = {pair: count for pair, count in bigrams.items() if count >= 3}
    payload = {"unigrams": kept, "bigrams": dict(sorted(bigrams.items()))}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"kept {len(kept)} of {len(counts)} token types, {len(bigrams)} bigrams -> {args.out}")


if __name__ == "__main__":
    main()
