"""Train the neural mission parser and pick hybrid thresholds; never reads test.

Training data: synthetic missions (courier.nlp.synth, train phrase banks) plus the
annotated train split. Model selection and hybrid thresholds use the validation
split, synthetic missions built only from held-out phrase banks, and the
hand-written challenge set.

    PYTHONPATH=src python scripts/nlp/train_neural_parser.py --out artifacts/nlp/neural_parser.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn

from courier.common import load_dataset
from courier.nlp import MissionParser, TargetSpec, resolve
from courier.nlp.neural import (
    HEADS,
    HybridMissionParser,
    HybridThresholds,
    NeuralMissionParser,
    NeuralParserNet,
    encode_texts,
    spec_labels,
)
from courier.nlp.synth import augment_real, generate, spec_from_mission

ROOT = Path(__file__).resolve().parents[2]


def labelled_real(dataset):
    rows = []
    for scene in dataset.scenes:
        goal, via = spec_from_mission(scene.mission)
        rows.append((scene.mission.text, goal, via, scene.mission.urgent, scene.mission.fragile))
    return rows


def labelled_synthetic(count: int, seed: int, holdout: bool):
    return [(e.text, e.goal, e.via, e.urgent, e.fragile) for e in generate(count, seed, holdout)]


def labelled_prebuilt(path: Path):
    """Load an already audited JSONL corpus without regenerating or relabelling it."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)

            def spec(value):
                return None if value is None else TargetSpec(value["type"], value["ref"], value["anchor"])

            rows.append((item["text"], spec(item["goal"]), spec(item.get("via")), item["urgent"], item["fragile"]))
    if not rows:
        raise SystemExit(f"prebuilt corpus is empty: {path}")
    return rows


def labelled_challenge(path: Path):
    def spec(value):
        return None if value is None else TargetSpec(*value)

    items = json.loads(path.read_text(encoding="utf-8"))
    return [
        (item["text"], spec(item["goal"]), spec(item.get("via")), item.get("urgent"), item.get("fragile"))
        for item in items
    ]


def tensors(rows):
    labels = [spec_labels(goal, via, bool(urgent), bool(fragile)) for _, goal, via, urgent, fragile in rows]
    return {name: torch.tensor([label[name] for label in labels]) for name in HEADS}


def spec_score(rows, parsed) -> dict[str, float]:
    """Exact parser-level agreement; None urgent/fragile labels are not checked."""
    hits = {"goal": 0, "via": 0, "urgent": 0, "fragile": 0, "all": 0}
    for (_, goal, via, urgent, fragile), p in zip(rows, parsed):
        ok = {
            "goal": p.goal == goal,
            "via": p.via == via,
            "urgent": urgent is None or p.urgent == urgent,
            "fragile": fragile is None or p.fragile == fragile,
        }
        ok["all"] = all(ok.values())
        for name, value in ok.items():
            hits[name] += value
    return {name: value / len(rows) for name, value in hits.items()}


def grounded_score(dataset, parsed) -> dict[str, float]:
    """Same metric as scripts/nlp/evaluate_nlp.py: resolved nodes on ground-truth landmarks."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts" / "nlp"))
    from evaluate_nlp import compare

    hits: dict[str, int] = {}
    for scene, p in zip(dataset.scenes, parsed):
        for name, ok in compare(scene, resolve(p, scene.landmarks)).items():
            hits[name] = hits.get(name, 0) + ok
    return {name: value / len(dataset.scenes) for name, value in hits.items()}


def predict_all(model: NeuralMissionParser, texts, batch: int = 256):
    out = []
    for start in range(0, len(texts), batch):
        out += model.predict_batch(texts[start : start + batch])
    return out


def train_one(args, seed: int, real, validation, holdout, device, init_index: int = 0, prebuilt=None) -> NeuralParserNet:
    """One net; fresh synthetic missions every epoch unless --fixed-synthetic."""
    random.seed(seed)
    torch.manual_seed(seed)
    net = NeuralParserNet(dim=args.dim, hidden=args.hidden)
    if args.init:
        artifact = torch.load(args.init, map_location="cpu", weights_only=False)
        states = artifact.get("state_dicts") or [artifact["state_dict"]]
        if artifact["config"] != net.config:
            raise SystemExit(f"--init config {artifact['config']} differs from --dim/--hidden {net.config}")
        net.load_state_dict(states[init_index % len(states)])
        print(f"[seed {seed}] initialised from {args.init} net {init_index % len(states)}", flush=True)
    net = net.to(device)
    print(f"[seed {seed}] parameters: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    rows_per_epoch = (len(prebuilt) if prebuilt is not None else args.synthetic) + len(real) * (args.real_repeat + args.real_augment)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(rows_per_epoch / args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss()
    fixed = labelled_synthetic(args.synthetic, seed, holdout=False) if args.fixed_synthetic else None

    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        net.train()
        synthetic = prebuilt or fixed or labelled_synthetic(args.synthetic, seed * 1000 + epoch, holdout=False)
        augment_rng = random.Random(seed * 1000 + epoch)
        augmented = [
            (augment_real(text, augment_rng), goal, via, urgent, fragile)
            for _ in range(args.real_augment)
            for text, goal, via, urgent, fragile in real
        ]
        rows = synthetic + real * args.real_repeat + augmented
        random.shuffle(rows)
        started, total = time.time(), 0.0
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            x = encode_texts([row[0] for row in chunk]).to(device)
            y = {name: value.to(device) for name, value in tensors(chunk).items()}
            logits = net(x)
            loss = sum(loss_fn(logits[name], y[name]) for name in HEADS)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += loss.item() * len(chunk)
        model = NeuralMissionParser(net)
        val = spec_score(validation, [p.parsed for p in predict_all(model, [r[0] for r in validation])])
        ho = spec_score(holdout, [p.parsed for p in predict_all(model, [r[0] for r in holdout])])
        score = val["all"] + ho["all"]
        print(
            f"[seed {seed}] epoch {epoch + 1}: loss {total / len(rows):.4f}  validation all {val['all']:.4f}"
            f"  holdout-phrasing all {ho['all']:.4f} goal {ho['goal']:.4f}  {time.time() - started:.0f}s",
            flush=True,
        )
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in net.state_dict().items()}
            if args.checkpoint_dir:
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                NeuralMissionParser(net).save(
                    args.checkpoint_dir / f"seed_{seed}_best.pt",
                    thresholds={"goal": 2.0, "via": 2.0, "flags": 2.0},
                    checkpoint=True, seed=seed, epoch=epoch + 1,
                    validation=val, holdout=ho,
                )
    net.load_state_dict(best_state)
    return net.cpu().eval()


def train(args) -> NeuralMissionParser:
    real = labelled_real(load_dataset(args.data, "train"))
    validation = labelled_real(load_dataset(args.data, "validation"))
    holdout = labelled_synthetic(2000, args.seed + 1, holdout=True)
    prebuilt = labelled_prebuilt(args.prebuilt_data) if args.prebuilt_data else None
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print(
        f"device: {device}; per epoch {len(prebuilt) if prebuilt is not None else args.synthetic} "
        f"{'prebuilt' if prebuilt is not None else 'synthetic'} + {len(real)} x{args.real_repeat} train;"
        f" {args.seeds} seed(s)",
        flush=True,
    )
    nets = [train_one(args, args.seed + 7919 * i, real, validation, holdout, device, i, prebuilt) for i in range(args.seeds)]
    return NeuralMissionParser(nets)


def choose_thresholds(model, sets) -> tuple[HybridThresholds, dict]:
    """Highest combined accuracy on validation + held-out phrasing + challenge, without losing validation."""
    rules = MissionParser()
    cached = []
    for name, rows in sets.items():
        texts = [r[0] for r in rows]
        cached.append((name, rows, [rules.parse(t) for t in texts], predict_all(model, texts)))

    def evaluate(thresholds):
        hybrid = HybridMissionParser(model, thresholds, rules)
        return {
            name: spec_score(rows, [hybrid.combine(r, n) for r, n in zip(rule_out, neural_out)])["all"]
            for name, rows, rule_out, neural_out in cached
        }

    rule_only = evaluate(HybridThresholds(2.0, 2.0, 2.0))
    best, best_scores = None, None
    grid = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 2.0)
    for goal in grid:
        for via in grid:
            for flags in (0.99, 0.999, 2.0):
                thresholds = HybridThresholds(goal, via, flags)
                scores = evaluate(thresholds)
                if scores["validation"] < rule_only["validation"]:
                    continue
                key = (scores["holdout"] + scores["challenge"], goal + via + flags)  # ties: most conservative
                if best is None or key > best[0]:
                    best, best_scores = (key, thresholds), scores
    return best[1], {"rule_only": rule_only, "chosen": best_scores}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/nlp/neural_parser.pt"))
    parser.add_argument("--report", type=Path, help="write metrics JSON here")
    parser.add_argument("--synthetic", type=int, default=60000)
    parser.add_argument("--prebuilt-data", type=Path, help="audited JSONL corpus used as the per-epoch synthetic block")
    parser.add_argument("--real-repeat", type=int, default=5)
    parser.add_argument("--real-augment", type=int, default=0, help="re-phrased copies of each real train mission per epoch")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--seeds", type=int, default=1, help="ensemble size (nets averaged at inference)")
    parser.add_argument("--init", type=Path, help="continue from this neural parser artifact (net i -> seed i)")
    parser.add_argument("--fixed-synthetic", action="store_true", help="reuse one synthetic set for all epochs")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--checkpoint-dir", type=Path, help="save the best model of each seed during training")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    model = train(args)

    validation_data = load_dataset(args.data, "validation")
    sets = {
        "validation": labelled_real(validation_data),
        "holdout": labelled_synthetic(3000, args.seed + 2, holdout=True),
        "challenge": labelled_challenge(ROOT / "tests" / "nlp" / "challenge_missions.json"),
    }
    thresholds, selection = choose_thresholds(model, sets)
    print(f"hybrid thresholds: {thresholds}  selection: {selection}", flush=True)

    rules = MissionParser()
    hybrid = HybridMissionParser(model, thresholds, rules)
    stored = {"goal": thresholds.goal, "via": thresholds.via, "flags": thresholds.flags}
    report = {"thresholds": stored, "metrics": {}}
    texts_val = [row[0] for row in sets["validation"]]
    report["metrics"]["validation_grounded"] = {
        "rules": grounded_score(validation_data, [rules.parse(t) for t in texts_val]),
        "neural": grounded_score(validation_data, [p.parsed for p in predict_all(model, texts_val)]),
        "hybrid": grounded_score(validation_data, [hybrid.parse(t) for t in texts_val]),
    }
    for name, rows in sets.items():
        texts = [row[0] for row in rows]
        report["metrics"][name] = {
            "rules": spec_score(rows, [rules.parse(t) for t in texts]),
            "neural": spec_score(rows, [p.parsed for p in predict_all(model, texts)]),
            "hybrid": spec_score(rows, [hybrid.parse(t) for t in texts]),
        }
    for name, table in report["metrics"].items():
        for source, scores in table.items():
            print(f"{name:20s} {source:7s} " + "  ".join(f"{k} {v:.4f}" for k, v in scores.items()))

    model.save(
        args.out,
        thresholds=stored,
        training={k: str(v) for k, v in vars(args).items()},
    )
    print(f"saved {args.out}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
