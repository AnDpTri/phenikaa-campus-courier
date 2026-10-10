"""Small neural mission parser, trained from scratch, and its hybrid with the rule parser.

Text is accent-folded and tokenised like the rule parser. Each token is embedded
as the sum of hashed buckets for the whole word and its character trigrams, so
unseen words and typos still share most of their features with known ones. A
two-layer bidirectional GRU encodes the sequence; every output field has its own
attention pooling and linear classifier.

The model predicts parser-level fields (TargetSpec for goal and via, urgent,
fragile); the existing resolver grounds them on the map, unchanged.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .parser import MissionParser, ParsedMission, TargetSpec
from .synth import GOAL_MODES, TYPES, VIA_MODES
from .text import tokenize

ARTIFACT_KIND = "courier.nlp.neural_parser"
BUCKETS = 1 << 16
MAX_TOKENS = 160
MAX_PIECES = 24
TYPE_LABELS = (None, *TYPES)
HEADS = {
    "goal_mode": len(GOAL_MODES),
    "goal_type": len(TYPE_LABELS),
    "goal_anchor": len(TYPE_LABELS),
    "via_mode": len(VIA_MODES),
    "via_type": len(TYPE_LABELS),
    "via_anchor": len(TYPE_LABELS),
    "urgent": 2,
    "fragile": 2,
}
DIRECTIONAL = ("north", "south", "west", "east")


# ---------------------------------------------------------------- features

def _bucket(piece: str) -> int:
    return 1 + zlib.crc32(piece.encode()) % (BUCKETS - 1)


def token_pieces(token: str) -> list[int]:
    """Hashed ids of the word and its character trigrams (0 is padding)."""
    padded = f"<{token}>"
    pieces = [_bucket("w:" + token)]
    pieces += [_bucket(padded[i : i + 3]) for i in range(len(padded) - 2)]
    return pieces[:MAX_PIECES]


def encode_texts(texts: list[str]) -> torch.Tensor:
    """(batch, tokens, pieces) int64 tensor of hashed ids."""
    tokenised = [tokenize(text)[:MAX_TOKENS] or ["."] for text in texts]
    length = max(len(tokens) for tokens in tokenised)
    out = torch.zeros((len(texts), length, MAX_PIECES), dtype=torch.long)
    for row, tokens in enumerate(tokenised):
        for column, token in enumerate(tokens):
            pieces = token_pieces(token)
            out[row, column, : len(pieces)] = torch.tensor(pieces)
    return out


# ---------------------------------------------------------------- labels

def spec_labels(goal: TargetSpec | None, via: TargetSpec | None, urgent: bool, fragile: bool) -> dict[str, int]:
    goal = goal or TargetSpec(None, "named")
    labels = {
        "goal_mode": GOAL_MODES.index(goal.ref or "named"),
        "goal_type": TYPE_LABELS.index(goal.type),
        "goal_anchor": TYPE_LABELS.index(goal.anchor),
        "urgent": int(urgent),
        "fragile": int(fragile),
    }
    if via is None:
        labels.update(via_mode=0, via_type=0, via_anchor=0)
    else:
        labels.update(
            via_mode=VIA_MODES.index(via.ref or "named"),
            via_type=TYPE_LABELS.index(via.type),
            via_anchor=TYPE_LABELS.index(via.anchor),
        )
    return labels


# ---------------------------------------------------------------- model

class NeuralParserNet(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.config = {"dim": dim, "hidden": hidden}
        self.embed = nn.Embedding(BUCKETS, dim, padding_idx=0)
        self.encoder = nn.GRU(dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        width = 2 * hidden
        self.attention = nn.ModuleDict({name: nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1)) for name in HEADS})
        self.output = nn.ModuleDict({name: nn.Linear(width, size) for name, size in HEADS.items()})

    def forward(self, pieces: torch.Tensor) -> dict[str, torch.Tensor]:
        counts = (pieces > 0).sum(-1, keepdim=True).clamp(min=1)
        tokens = self.embed(pieces).sum(2) / counts.sqrt()
        mask = pieces[:, :, 0] > 0
        states, _ = self.encoder(self.dropout(tokens))
        states = self.dropout(states)
        logits = {}
        for name in HEADS:
            score = self.attention[name](states).squeeze(-1).masked_fill(~mask, -1e4)
            pooled = (torch.softmax(score, dim=1).unsqueeze(-1) * states).sum(1)
            logits[name] = self.output[name](pooled)
        return logits


@dataclass(frozen=True, slots=True)
class NeuralPrediction:
    parsed: ParsedMission
    goal_confidence: float
    via_confidence: float
    urgent_confidence: float
    fragile_confidence: float
    goal_type_probabilities: tuple[float, ...] = ()  # aligned with TYPE_LABELS


def _decode(kind: str, mode: str, type_index: int, anchor_index: int) -> TargetSpec | None:
    if kind == "via" and mode == "none":
        return None
    type_ = TYPE_LABELS[type_index]
    anchor = TYPE_LABELS[anchor_index]
    if mode == "named":
        return TargetSpec(type_) if type_ else None
    if mode == "anchor_near":
        return TargetSpec(None, mode, anchor) if anchor else None
    if mode.endswith("_most"):
        return TargetSpec(None, mode)
    if type_ is None:
        return None
    if mode in DIRECTIONAL:
        return TargetSpec(type_, mode)
    return TargetSpec(type_, mode, anchor) if anchor and anchor != type_ else TargetSpec(type_)


def _used_heads(kind: str, mode: str) -> tuple[str, ...]:
    if kind == "via" and mode == "none":
        return ("via_mode",)
    heads = [f"{kind}_mode"]
    if mode == "anchor_near":
        heads.append(f"{kind}_anchor")
    elif not mode.endswith("_most"):
        heads.append(f"{kind}_type")
        if mode in ("near", "far"):
            heads.append(f"{kind}_anchor")
    return tuple(heads)


class NeuralMissionParser:
    """One net, or an ensemble whose softmax outputs are averaged."""

    def __init__(self, nets: NeuralParserNet | list[NeuralParserNet]) -> None:
        self.nets = [net.eval() for net in (nets if isinstance(nets, list) else [nets])]
        self.net = self.nets[0]

    @classmethod
    def load(cls, path: str | Path) -> "NeuralMissionParser":
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("kind") != ARTIFACT_KIND:
            raise ValueError(f"{path} is not a neural parser artifact")
        states = artifact.get("state_dicts") or [artifact["state_dict"]]
        nets = []
        for state in states:
            net = NeuralParserNet(**artifact["config"])
            net.load_state_dict(state)
            nets.append(net)
        return cls(nets)

    def save(self, path: str | Path, **metadata) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": ARTIFACT_KIND,
                "config": self.net.config,
                "state_dicts": [{k: v.cpu() for k, v in net.state_dict().items()} for net in self.nets],
                **metadata,
            },
            path,
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[NeuralPrediction]:
        device = next(self.net.parameters()).device
        pieces = encode_texts(texts).to(device)
        probabilities = {}
        for net in self.nets:
            for name, value in net(pieces).items():
                probabilities[name] = probabilities.get(name, 0) + torch.softmax(value, -1).cpu() / len(self.nets)
        best = {name: value.max(-1) for name, value in probabilities.items()}
        results = []
        for row, text in enumerate(texts):
            index = {name: int(best[name].indices[row]) for name in HEADS}
            confidence = {name: float(best[name].values[row]) for name in HEADS}
            specs, scores = {}, {}
            for kind, modes in (("goal", GOAL_MODES), ("via", VIA_MODES)):
                mode = modes[index[f"{kind}_mode"]]
                specs[kind] = _decode(kind, mode, index[f"{kind}_type"], index[f"{kind}_anchor"])
                score = 1.0
                for head in _used_heads(kind, mode):
                    score *= confidence[head]
                scores[kind] = score
            parsed = ParsedMission(text, specs["goal"], specs["via"], bool(index["urgent"]), bool(index["fragile"]))
            results.append(
                NeuralPrediction(
                    parsed, scores["goal"], scores["via"], confidence["urgent"], confidence["fragile"],
                    tuple(float(v) for v in probabilities["goal_type"][row]),
                )
            )
        return results

    def predict(self, text: str) -> NeuralPrediction:
        return self.predict_batch([text])[0]

    def parse(self, text: str) -> ParsedMission:
        return self.predict(text).parsed


# ---------------------------------------------------------------- hybrid

@dataclass(frozen=True, slots=True)
class HybridThresholds:
    """The neural answer replaces the rule answer only when it is at least this confident."""

    goal: float = 0.9
    via: float = 0.9
    flags: float = 0.97


class HybridMissionParser:
    """Rule parser first; the neural parser fills a missing goal and overrides confident disagreements."""

    def __init__(self, neural: NeuralMissionParser, thresholds: HybridThresholds | None = None, rules=None) -> None:
        self.neural = neural
        self.rules = rules or MissionParser()
        self.thresholds = thresholds or HybridThresholds()
        self.last_sources: dict[str, str] = {}

    @classmethod
    def load(cls, path: str | Path, thresholds: HybridThresholds | None = None) -> "HybridMissionParser":
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        stored = artifact.get("thresholds")
        neural = NeuralMissionParser.load(path)
        return cls(neural, thresholds or (HybridThresholds(**stored) if stored else None))

    def combine(self, rule: ParsedMission, prediction: NeuralPrediction) -> ParsedMission:
        neural, t = prediction.parsed, self.thresholds
        sources = {}
        goal = rule.goal
        if goal is None and neural.goal is not None:
            goal, sources["goal"] = neural.goal, "neural_fill"
        elif neural.goal is not None and neural.goal != goal and prediction.goal_confidence >= t.goal:
            goal, sources["goal"] = neural.goal, "neural_override"
        via = rule.via
        if neural.via != via and prediction.via_confidence >= t.via:
            via, sources["via"] = neural.via, "neural_override"
        urgent = rule.urgent
        if neural.urgent != urgent and prediction.urgent_confidence >= t.flags:
            urgent, sources["urgent"] = neural.urgent, "neural_override"
        fragile = rule.fragile
        if neural.fragile != fragile and prediction.fragile_confidence >= t.flags:
            fragile, sources["fragile"] = neural.fragile, "neural_override"
        self.last_sources = sources
        return ParsedMission(rule.text, goal, via, urgent, fragile)

    def parse(self, text: str) -> ParsedMission:
        return self.combine(self.rules.parse(text), self.neural.predict(text))

    def parse_for_map(self, text: str, landmarks) -> ParsedMission:
        """Like parse, but when the goal cannot be found on the map, try the neural reading,
        then the most probable neural goal type that is on the map.

        Without this the resolver falls back to the first landmark in the list.
        """
        from .resolver import resolve_target

        prediction = self.neural.predict(text)
        parsed = self.combine(self.rules.parse(text), prediction)
        present = {kind for kind, _ in landmarks}

        def usable(spec) -> bool:
            target = resolve_target(spec, landmarks) if spec is not None else None
            return target is not None and target[0] in present

        if usable(parsed.goal):
            return parsed
        sources = dict(self.last_sources)
        goal = None
        if usable(prediction.parsed.goal):
            goal, sources["goal"] = prediction.parsed.goal, "neural_map"
        elif prediction.goal_type_probabilities:
            ranked = sorted(
                (p, kind) for p, kind in zip(prediction.goal_type_probabilities, TYPE_LABELS) if kind in present
            )
            if ranked:
                goal, sources["goal"] = TargetSpec(ranked[-1][1]), "neural_map_type"
        self.last_sources = sources
        if goal is None:
            return parsed
        return ParsedMission(parsed.text, goal, parsed.via, parsed.urgent, parsed.fragile)
