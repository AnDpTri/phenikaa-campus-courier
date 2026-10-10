"""Create a NEW, reproducible 200k training corpus.

Preserves the 38,833 training-compatible specialized examples verbatim.
Uses the original V2 synth generator exclusively for the remaining 161,167.
Creates a separate V2-heldout suite; never uses any actual competition test data.
Will NOT overwrite existing output files.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import random
import time

from courier.nlp.synth import Generator, GOAL_MODES, VIA_MODES, TYPES
from courier.nlp.neural import spec_labels, MAX_TOKENS
from courier.nlp.parser import TargetSpec
from courier.nlp.text import tokenize

SOURCE = Path(r"D:\phenikaa\results\nlp_v4_data_350k\replacement_20261009_review")
DEST = Path(r"D:\phenikaa\results\NLP_V4_R2\hybrid_40k_original_v2_160k_20261009_b")
INCOMING = SOURCE / "incoming_40000.jsonl"
TRAINABLE = SOURCE / "retained_swap_38833_TRAINABLE.jsonl"
TRAIN_FILE = DEST / "train_hybrid_200000.jsonl"
HOLDOUT_FILE = DEST / "holdout_v2_5000.jsonl"
MANIFEST_FILE = DEST / "manifest.json"

TRAIN_N = 200_000
SPECIALIZED_N = 38_833
V2_N = TRAIN_N - SPECIALIZED_N
HOLDOUT_N = 5_000
SEED_V2 = 20261102
SEED_HOLDOUT = 20261103
SEED_SHUFFLE = 20261104

def log(*parts):
    print(*parts, flush=True)

def to_spec(o):
    return None if o is None else TargetSpec(o["type"], o["ref"], o["anchor"])

def validate(row):
    assert isinstance(row["text"], str) and len(row["text"].strip()) >= 8
    assert type(row["urgent"]) is bool and type(row["fragile"]) is bool
    goal, via = to_spec(row["goal"]), to_spec(row["via"])
    assert goal is not None
    spec_labels(goal, via, row["urgent"], row["fragile"])
    assert (goal.ref or "named") in GOAL_MODES
    assert via is None or (via.ref or "named") in VIA_MODES
    for tgt in (goal, via):
        if tgt is None:
            continue
        assert tgt.type is None or tgt.type in TYPES
        assert tgt.anchor is None or tgt.anchor in TYPES
        ref = tgt.ref or "named"
        if ref in ("near", "far"):
            assert tgt.type is not None and tgt.anchor is not None
        elif ref == "anchor_near":
            assert tgt.type is None and tgt.anchor is not None
        elif ref.endswith("_most"):
            assert tgt.type is None and tgt.anchor is None
        elif ref == "named" or ref in ("north", "south", "east", "west"):
            assert tgt.type is not None and tgt.anchor is None
    assert len(tokenize(row["text"])) > 0

def make_row(ex, ident):
    def as_target(t):
        return None if t is None else {
            "type": t.type, "ref": t.ref, "anchor": t.anchor
        }
    return {
        "id": ident,
        "text": ex.text,
        "goal": as_target(ex.goal),
        "via": as_target(ex.via),
        "urgent": ex.urgent,
        "fragile": ex.fragile,
    }

def hash_file(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def main():
    for path in (TRAIN_FILE, HOLDOUT_FILE, MANIFEST_FILE):
        if path.exists():
            raise RuntimeError(f"Refuse overwrite: {path}")
    started = time.perf_counter()
    log("PREPARING", str(DEST))
    incoming = {}
    with INCOMING.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert row["id"].startswith("extra_")
            incoming[row["id"]] = row
    assert len(incoming) == 40_000
    log("SOURCE_40K", len(incoming))

    specialized = []
    seen_id = set()
    with TRAINABLE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row["id"].startswith("extra_"):
                continue
            assert row["id"] in incoming
            assert canonical(row) == canonical(incoming[row["id"]]), (
                "Specialized original content was altered", row["id"]
            )
            assert row["id"] not in seen_id
            seen_id.add(row["id"])
            validate(row)
            specialized.append(row)
    assert len(specialized) == SPECIALIZED_N, len(specialized)
    log("SPECIALIZED_UNCHANGED", len(specialized),
        "QUARANTINED", len(incoming) - len(specialized))

    rng = Generator(seed=SEED_V2, holdout=False)
    rows = list(specialized)
    existing_texts = {row["text"].strip().casefold() for row in specialized}
    duplicates_rejected = 0
    while len(rows) < TRAIN_N:
        e = rng.example()
        row = make_row(e, f"r2v2_{len(rows)-SPECIALIZED_N+1:06d}")
        key = row["text"].strip().casefold()
        if key in existing_texts:
            duplicates_rejected += 1
            continue
        validate(row)
        existing_texts.add(key)
        rows.append(row)
        if len(rows) % 50_000 == 0:
            log("BUILDING", len(rows), "/", TRAIN_N)
    assert len(rows) == TRAIN_N
    random.Random(SEED_SHUFFLE).shuffle(rows)

    stat = Counter()
    for row in rows:
        group = "specialized_original" if row["id"].startswith("extra_") else "v2_synthetic"
        via = row["via"]
        stat["total"] += 1
        stat[f"{group}_n"] += 1
        stat[f"{group}_via_none"] += via is None
        stat[f"{group}_via_named"] += via is not None and via["ref"] is None
        stat[f"{group}_long_over_{MAX_TOKENS}_tokens"] += len(tokenize(row["text"])) > MAX_TOKENS
    log("TRAIN_COUNTS", json.dumps(dict(stat),ensure_ascii=True))

    # Exclusive file creation: it is never safe to overwrite an old training artifact.
    with TRAIN_FILE.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    del rows
    log("TRAIN_WRITTEN", TRAIN_FILE, "bytes", TRAIN_FILE.stat().st_size)

    holdout_rng = Generator(seed=SEED_HOLDOUT, holdout=True)
    holdout_stat = Counter()
    with HOLDOUT_FILE.open("x", encoding="utf-8", newline="\n") as f:
        for i in range(HOLDOUT_N):
            ex = holdout_rng.example()
            row = make_row(ex, f"r2ho_{i+1:06d}")
            validate(row)
            assert row["text"].strip().casefold() not in existing_texts, "Holdout leakage"
            via = row["via"]
            holdout_stat["n"] += 1
            holdout_stat["via_none"] += via is None
            holdout_stat["via_named"] += via is not None and via["ref"] is None
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "status": "generated_untrained",
        "purpose": "New 200k hybrid training corpus plus independent V2 synthetic phrase-bank holdout",
        "quality_limit": "Specialized 38833 preserve original content exactly. Remaining V2 samples inherit original generator, NOT the same style/quality as specialized 40k. No human semantic re-audit performed.",
        "train_path": str(TRAIN_FILE),
        "holdout_path": str(HOLDOUT_FILE),
        "specialized_original": SPECIALIZED_N,
        "specialized_source_original_total": 40_000,
        "specialized_quarantined_unsupported": 40_000 - SPECIALIZED_N,
        "v2_synthetic": V2_N,
        "train_total": TRAIN_N,
        "holdout_v2_new": HOLDOUT_N,
        "holdout_used_for_training": False,
        "train_text_unique_casefolded": True,
        "v2_generated_duplicates_rejected": duplicates_rejected,
        "seeds": {"v2_train": SEED_V2, "v2_holdout": SEED_HOLDOUT, "train_shuffle": SEED_SHUFFLE},
        "source_files": {
            "incoming_40k": str(INCOMING),
            "specialized_extracted_from": str(TRAINABLE),
            "incoming_sha256": hash_file(INCOMING),
            "trainable_swapped_sha256": hash_file(TRAINABLE),
        },
        "generated_sha256": {"train": hash_file(TRAIN_FILE), "holdout": hash_file(HOLDOUT_FILE)},
        "counts": dict(stat),
        "holdout_counts": dict(holdout_stat),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "validation_used_for_generation": False,
        "competition_test_used_for_generation": False,
        "source_modified": False,
    }
    with MANIFEST_FILE.open("x", encoding="utf-8") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)
    log("FINAL", json.dumps({
        "train": TRAIN_N, "specialized": SPECIALIZED_N,
        "v2": V2_N, "holdout": HOLDOUT_N,
        "dupes": duplicates_rejected, "elapsed": manifest["elapsed_seconds"]
    },ensure_ascii=True))

if __name__ == "__main__":
    main()
