"""Standalone NLP V4 R2 research pilot. Creates only new files in its output dir."""
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
from courier.nlp.neural import (
    HEADS, HybridMissionParser, HybridThresholds, NeuralMissionParser,
    NeuralParserNet, encode_texts, spec_labels,
)
from courier.nlp.parser import MissionParser, TargetSpec
from courier.nlp.synth import generate, spec_from_mission

ROOT = Path(r"D:\phenikaa")
OUT = ROOT / "results" / "NLP_V4_R2" / "pilot_distill_20261009_a"
DATA = ROOT / "Phenikaa_Campus_Courier_2026_v3" / "delivery_public"
CORPUS = ROOT / "results" / "nlp_v4_data_350k" / "replacement_20261009_review" / "nlp_v4_curated_200000_quality_v2_reindexed.jsonl"
V2 = ROOT / "artifacts" / "nlp" / "neural_parser_v2.pt"
V4 = ROOT / "results" / "NLP_V4_Curated200K_Finetune_R1" / "resume_E1_B64_LR5e5" / "model.pt"
TEMPERATURE = 2.0
BATCH = 64
LR = 5e-5
KD_VIA = 0.6
KD_GOAL = 0.3
SEED = 20261009
teacher_heads = ("via_mode", "via_type", "via_anchor", "goal_mode", "goal_type", "goal_anchor")


def announce(*parts):
    print(*parts, flush=True)


def mk_rows(dataset):
    return [(s.mission.text, *spec_from_mission(s.mission), s.mission.urgent, s.mission.fragile) for s in dataset.scenes]


def from_json(line):
    item = json.loads(line)
    def spec(val):
        return None if val is None else TargetSpec(val["type"], val["ref"], val["anchor"])
    return item["text"], spec(item["goal"]), spec(item["via"]), item["urgent"], item["fragile"]


def labels(chunk, device):
    specs = [spec_labels(z[1], z[2], z[3], z[4]) for z in chunk]
    return {name: torch.tensor([z[name] for z in specs], dtype=torch.long, device=device) for name in HEADS}


def predictions(model, rows):
    out = []
    for start in range(0, len(rows), 32):
        out.extend(model.predict_batch([r[0] for r in rows[start:start + 32]]))
    return out


def evaluate(model, rows, rules_out=None):
    pred = predictions(model, rows)
    result = {}
    for variant in ("neural", "hybrid") if rules_out is not None else ("neural",):
        counts = {x: 0 for x in ("goal", "via", "urgent", "fragile", "all", "named_total", "named_correct", "named_none", "via_wrong_confident")}
        hybrid = HybridMissionParser(model, HybridThresholds(.8, .95, 2.0)) if rules_out is not None else None
        for i, (r, p) in enumerate(zip(rows, pred)):
            decoded = hybrid.combine(rules_out[i], p) if variant == "hybrid" else p.parsed
            ok = {"goal": decoded.goal == r[1], "via": decoded.via == r[2], "urgent": decoded.urgent == r[3], "fragile": decoded.fragile == r[4]}
            for k, flag in ok.items():
                counts[k] += int(flag)
            counts["all"] += int(all(ok.values()))
            if r[2] is not None and r[2].ref is None:
                counts["named_total"] += 1
                counts["named_correct"] += int(ok["via"])
                counts["named_none"] += int(decoded.via is None)
            counts["via_wrong_confident"] += int(not ok["via"] and p.via_confidence >= .95)
        n = len(rows)
        result[variant] = {key: round(counts[key] / n, 5) for key in ("goal", "via", "urgent", "fragile", "all")}
        result[variant].update({key: counts[key] for key in ("named_total", "named_correct", "named_none", "via_wrong_confident")})
    return result


def teacher_probabilities(artifact_path, texts, names, selected):
    """Prepare teacher targets sequentially so student/teachers never share GPU allocations."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    states = artifact.get("state_dicts") or [artifact["state_dict"]]
    collected = {name: torch.zeros((len(texts), HEADS[name]), dtype=torch.float32) for name in selected}
    for seed_index, state in enumerate(states):
        net = NeuralParserNet(**artifact["config"])
        net.load_state_dict(state)
        net.to("cuda").eval()
        start_time = time.time()
        with torch.no_grad():
            for start in range(0, len(texts), 48):
                encoded = encode_texts(texts[start:start + 48]).to("cuda")
                logits = net(encoded)
                for name in selected:
                    collected[name][start:start + len(encoded)] += torch.softmax(logits[name], -1).cpu() / len(states)
        announce("TEACHER", names, "seed", seed_index + 1, "/", len(states), "seconds", round(time.time() - start_time, 1))
        del net
        torch.cuda.empty_cache()
    del artifact
    return collected


def distill(head, logits, y, probabilities):
    q = probabilities.clamp_min(1e-8)
    q = q.pow(1 / TEMPERATURE)
    q = q / q.sum(-1, keepdim=True)
    mask = probabilities.argmax(-1) == y[head]
    if head == "via_type":
        mask = mask & (y["via_mode"] != 0)
    if head == "via_anchor":
        mask = mask & ((y["via_mode"] == 6) | (y["via_mode"] == 7))
    if head == "goal_type":
        mask = mask & (y["goal_type"] != 0)
    if head == "goal_anchor":
        mask = mask & (y["goal_anchor"] != 0)
    if not mask.any():
        return logits[head].sum() * 0.0
    loss = -(q * torch.log_softmax(logits[head] / TEMPERATURE, -1)).sum(-1) * (TEMPERATURE ** 2)
    return loss[mask].mean()


def pilot(name, examples, indices, truth, teacher_via, teacher_goal, init_state, config, val, holdout, rules):
    if (OUT / (name + ".pt")).exists():
        raise SystemExit("Refusing to overwrite existing checkpoint " + name)
    torch.manual_seed(SEED)
    random.seed(SEED)
    net = NeuralParserNet(**config)
    net.load_state_dict(init_state)
    net = net.cuda().train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = math.ceil(len(indices) / BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=.1)
    lossfn = nn.CrossEntropyLoss()
    started = time.time()
    total_loss = 0.0
    announce("START", name, "rows", len(indices), "steps", steps)
    for iteration, pos in enumerate(range(0, len(indices), BATCH)):
        local_indices = indices[pos:pos + BATCH]
        batch = [examples[k] for k in local_indices]
        x = encode_texts([r[0] for r in batch]).cuda()
        y = {key: val[local_indices].cuda() for key, val in truth.items()}
        out = net(x)
        loss = sum(lossfn(out[head], y[head]) for head in HEADS)
        if name != "R2_A_rehearsal":
            for head in ("via_mode", "via_type", "via_anchor"):
                p = teacher_via[head][local_indices].cuda()
                loss = loss + KD_VIA * distill(head, out, y, p)
        if name == "R2_C_via_goal_KD":
            for head in ("goal_mode", "goal_type", "goal_anchor"):
                p = teacher_goal[head][local_indices].cuda()
                loss = loss + KD_GOAL * distill(head, out, y, p)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        schedule.step()
        total_loss += float(loss.item()) * len(batch)
        if (iteration + 1) % 100 == 0:
            announce("PROGRESS", name, iteration + 1, "/", steps, "seconds", round(time.time() - started, 1))
    model = NeuralMissionParser(net)
    result = {
        "seconds": round(time.time() - started, 1),
        "training_loss_mean": round(total_loss / len(indices), 5),
        "validation": evaluate(model, val, rules),
        "holdout": evaluate(model, holdout),
        "checkpoint": name + ".pt",
    }
    checkpoint = OUT / result["checkpoint"]
    if checkpoint.exists():
        raise SystemExit("Refusing to overwrite " + str(checkpoint))
    model.save(checkpoint,
               thresholds={"goal": .8, "via": .95, "flags": 2.0},
               training={"experiment": name, "source": "NLP_V4_R2", "data": "curated 12000 + dynamic 12000 + real 2000",
                         "epochs": 1, "batch": BATCH, "lr": LR, "kd_via_weight": KD_VIA if name != "R2_A_rehearsal" else 0,
                         "kd_goal_weight": KD_GOAL if name == "R2_C_via_goal_KD" else 0,
                         "temperature": TEMPERATURE, "seed": SEED})
    announce("RESULT", name, json.dumps(result, ensure_ascii=False))
    del net, model, opt, schedule
    torch.cuda.empty_cache()
    return result


def main():
    torch.set_num_threads(4)
    if any((OUT / name).exists() for name in
           ("report.json", "R2_A_rehearsal.pt", "R2_B_via_KD.pt", "R2_C_via_goal_KD.pt")):
        raise SystemExit("Refusing to modify existing output")
    original = torch.load(V2, map_location="cpu", weights_only=False)
    init_state = original["state_dicts"][0]
    config = original["config"]
    train = mk_rows(load_dataset(DATA, "train"))
    val = mk_rows(load_dataset(DATA, "validation"))
    holdout = [(s.text, s.goal, s.via, s.urgent, s.fragile) for s in generate(2000, 20261010, holdout=True)]
    with CORPUS.open(encoding="utf-8") as handle:
        selected = random.Random(61009).sample([from_json(line) for line in handle], 12000)
    dynamic = [(s.text, s.goal, s.via, s.urgent, s.fragile) for s in generate(12000, 20261030, holdout=False)]
    examples = selected + dynamic + train
    texts = [r[0] for r in examples]
    truth = {name: torch.tensor([spec_labels(r[1], r[2], r[3], r[4])[name] for r in examples], dtype=torch.long) for name in HEADS}
    rule_parser = MissionParser()
    rule_out = [rule_parser.parse(row[0]) for row in val]
    announce("READY", "examples", len(examples), "validation", len(val), "holdout", len(holdout))
    teacher_via = teacher_probabilities(V2, texts, "V2 ensemble via", ("via_mode", "via_type", "via_anchor"))
    teacher_goal = teacher_probabilities(V4, texts, "V4 R1 goal", ("goal_mode", "goal_type", "goal_anchor"))
    indices = list(range(len(examples)))
    random.Random(20261009).shuffle(indices)
    results = {}
    for name in ("R2_A_rehearsal", "R2_B_via_KD", "R2_C_via_goal_KD"):
        results[name] = pilot(name, examples, indices, truth, teacher_via, teacher_goal,
                              init_state, config, val, holdout, rule_out)
    report = {
        "objective": "Controlled one-epoch pilot comparing rehearsal vs via KD vs via+goal KD",
        "notice": "Exploratory, small pilot; not an independent test score or official submission",
        "data": {"curated": 12000, "dynamic": 12000, "real": len(train), "validation": len(val),
                 "synthetic_holdout": len(holdout), "test_used": False},
        "settings": {"seed": SEED, "batch": BATCH, "lr": LR, "temperature": TEMPERATURE,
                     "via_KD": KD_VIA, "goal_KD": KD_GOAL,
                     "teacher_agreement_mask": True, "init": str(V2), "via_teacher": str(V2),
                     "goal_teacher": str(V4), "preexisting_files_modified": False},
        "results": results,
    }
    with (OUT / "report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    announce("ALL_DONE", str(OUT / "report.json"))


if __name__ == "__main__":
    main()