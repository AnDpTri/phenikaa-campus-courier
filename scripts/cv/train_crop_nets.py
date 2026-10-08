"""Train the node and edge CNNs on crops from build_crops.py.

Edge training also sees the legend road swatches (look labels only), so the same
network can read which look the legend assigns to each status.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from courier.cv.crops import LOOK_CLASSES, NODE_CLASSES
from courier.cv.nets import EdgeNet, NodeNet, save_net, to_tensor


def batches(n: int, size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, size):
        yield order[start : start + size]


def color_jitter(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    gain = torch.from_numpy(rng.uniform(0.85, 1.15, (x.shape[0], 1, 1, 1))).float()
    bias = torch.from_numpy(rng.uniform(-0.06, 0.06, (x.shape[0], 1, 1, 1))).float()
    return (x * gain + bias).clamp_(0, 1)


def train_nodes(train, val, args) -> None:
    rng = np.random.default_rng(0)
    net = NodeNet(args.width)
    x, y = train["nodes"], torch.from_numpy(train["node_y"])
    weights = torch.ones(len(NODE_CLASSES))
    weights[0] = 0.3  # empty nodes dominate
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(x) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    for epoch in range(args.epochs):
        net.train()
        started, total = time.perf_counter(), 0.0
        for idx in batches(len(x), args.batch, rng):
            xb = color_jitter(to_tensor(x[idx]), rng)
            loss = loss_fn(net(xb), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        pred = predict(net, val["nodes"], lambda out: out)
        acc = (pred.argmax(1) == val["node_y"]).mean()
        occupied = val["node_y"] > 0
        occ_acc = (pred.argmax(1)[occupied] == val["node_y"][occupied]).mean()
        print(
            f"nodes epoch {epoch + 1}: loss {total / len(x):.4f}  val acc {acc:.4f}  occupied acc {occ_acc:.4f}"
            f"  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
    save_net(net, args.out / "node_net.pt", init={"width": args.width}, classes=NODE_CLASSES)


def train_edges(train, val, args) -> None:
    rng = np.random.default_rng(1)
    net = EdgeNet(args.width)
    # Swatches join the look task; their stairs/one-way targets are ignored (-100).
    x = np.concatenate([train["edges"], train["swatches"]])
    swatch_targets = np.stack(
        [train["swatch_y"], np.full(len(train["swatch_y"]), -100), np.full(len(train["swatch_y"]), -100)], 1
    )
    y = torch.from_numpy(np.concatenate([train["edge_y"], swatch_targets]))
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(x) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    for epoch in range(args.epochs):
        net.train()
        started, total = time.perf_counter(), 0.0
        for idx in batches(len(x), args.batch, rng):
            xb = color_jitter(to_tensor(x[idx]), rng)
            yb = y[idx].clone()
            flip = torch.from_numpy(rng.random(len(idx)) < 0.5)
            # Mirroring along the road swaps the one-way direction; across the road changes nothing.
            xb[flip] = xb[flip].flip(3)
            oneway = yb[flip, 2]
            yb[flip, 2] = torch.where(oneway == 1, 2, torch.where(oneway == 2, 1, oneway))
            vflip = torch.from_numpy(rng.random(len(idx)) < 0.5)
            xb[vflip] = xb[vflip].flip(2)
            look, stairs, ow = net(xb)
            loss = ce(look, yb[:, 0]) + ce(stairs, yb[:, 1]) + ce(ow, yb[:, 2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        look, stairs, ow = predict(net, val["edges"], lambda out: torch.cat(out, 1), split=True)
        ey = val["edge_y"]
        sw_look = predict(net, val["swatches"], lambda out: out[0])
        print(
            f"edges epoch {epoch + 1}: loss {total / len(x):.4f}  look {(look.argmax(1) == ey[:, 0]).mean():.4f}"
            f"  stairs {(stairs.argmax(1) == ey[:, 1]).mean():.4f}  oneway {(ow.argmax(1) == ey[:, 2]).mean():.4f}"
            f"  swatch {(sw_look.argmax(1) == val['swatch_y']).mean():.4f}  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
    save_net(net, args.out / "edge_net.pt", init={"width": args.width}, looks=LOOK_CLASSES)


@torch.no_grad()
def predict(net, x: np.ndarray, pack, split: bool = False, size: int = 1024):
    net.eval()
    outs = [pack(net(to_tensor(x[i : i + size]))) for i in range(0, len(x), size)]
    out = torch.cat(outs).numpy()
    if split:
        return out[:, :5], out[:, 5:7], out[:, 7:]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, default=Path("data/cv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cv"))
    parser.add_argument("--which", choices=("nodes", "edges", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(0)
    train = dict(np.load(args.crops / "crops_train.npz"))
    val = dict(np.load(args.crops / "crops_validation.npz"))
    if args.which in ("nodes", "both"):
        train_nodes(train, val, args)
    if args.which in ("edges", "both"):
        train_edges(train, val, args)


if __name__ == "__main__":
    main()
