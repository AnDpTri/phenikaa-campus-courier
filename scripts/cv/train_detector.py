"""Train the keypoint detector (node centres, legend road swatches, weather icon)."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from courier.cv import load_annotations, load_rgb
from courier.cv.annotations import SceneAnnotation
from courier.cv.detector import HEATMAPS, OUT, STRIDE, DetectorNet, letterbox
from courier.cv.types import LEGEND_ROAD_ORDER
from courier.cv.nets import save_net

SIGMA = {"node": 1.0, "swatch": 1.0, "weather": 1.5}


def scene_targets(annotation: SceneAnnotation):
    image, scale = letterbox(load_rgb(annotation.image_path))
    to_cell = scale / STRIDE

    def cell(x: float, y: float) -> tuple[float, float]:
        return (x * scale - 1.5) / STRIDE, (y * scale - 1.5) / STRIDE

    points = {name: [] for name in HEATMAPS}
    for x, y in annotation.graph.grid.node_xy.values():
        points["node"].append((*cell(x, y), 0.0, 0.0))
    for kind in LEGEND_ROAD_ORDER:
        x1, y1, x2, y2 = annotation.legend.entry(kind).swatch
        points[f"swatch_{kind}"].append(
            (*cell((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1) * to_cell, (y2 - y1) * to_cell)
        )
    if annotation.legend.weather_box is not None:
        x1, y1, x2, y2 = annotation.legend.weather_box
        points["weather"].append((*cell((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1) * to_cell, (y2 - y1) * to_cell))
    return image, points


def render(points: dict[str, list]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat = np.zeros((len(HEATMAPS), OUT, OUT), dtype=np.float32)
    size = np.zeros((2, OUT, OUT), dtype=np.float32)
    mask = np.zeros((OUT, OUT), dtype=np.float32)
    yy, xx = np.mgrid[0:OUT, 0:OUT]
    for channel, name in enumerate(HEATMAPS):
        sigma = SIGMA.get(name, 1.0)
        for cx, cy, w, h in points[name]:
            heat[channel] = np.maximum(heat[channel], np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
            r, c = int(round(cy)), int(round(cx))
            if 0 <= r < OUT and 0 <= c < OUT:
                heat[channel, r, c] = 1.0
                if name != "node":
                    size[:, r, c] = (w, h)
                    mask[r, c] = 1.0
    return heat, size, mask


def build_cache(data: Path, split: str, out: Path, workers: int) -> Path:
    path = out / f"detector_{split}.npz"
    if path.exists():
        with np.load(path) as cached:
            if "heatmaps" in cached and tuple(str(name) for name in cached["heatmaps"]) == HEATMAPS:
                return path
        print(f"detector cache schema changed; rebuilding {path}", flush=True)
        path.unlink()
    annotations = load_annotations(data, split)
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(scene_targets, annotations, chunksize=8))
    images = np.stack([image for image, _ in results])
    maps = [render(points) for _, points in results]
    np.savez(
        path,
        heatmaps=np.asarray(HEATMAPS),
        images=images,
        heat=np.stack([m[0] for m in maps]).astype(np.float16),
        size=np.stack([m[1] for m in maps]).astype(np.float16),
        mask=np.stack([m[2] for m in maps]).astype(np.uint8),
    )
    return path


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet focal loss on Gaussian targets, normalised by the number of peaks."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p**2 * (1 - target) ** 4 * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Phenikaa_Campus_Courier_2026_v3/delivery_public"))
    parser.add_argument("--cache", type=Path, default=Path("data/cv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cv/detector.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if resolved_device == "auto" else resolved_device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    train = np.load(build_cache(args.data, "train", args.cache, args.workers))
    val = np.load(build_cache(args.data, "validation", args.cache, args.workers))
    tx, th, ts, tm = train["images"], train["heat"], train["size"], train["mask"]
    vx, vh = val["images"], val["heat"]

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = DetectorNet(args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * ((len(tx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
    best_validation = float("inf")
    for epoch in range(args.epochs):
        net.train()
        started, total = time.perf_counter(), 0.0
        order = rng.permutation(len(tx))
        for start in range(0, len(tx), args.batch):
            idx = np.sort(order[start : start + args.batch])
            x = torch.from_numpy(tx[idx]).permute(0, 3, 1, 2).float().div_(255.0).to(device)
            gain = torch.from_numpy(rng.uniform(0.8, 1.2, (len(idx), 1, 1, 1))).float().to(device)
            bias = torch.from_numpy(rng.uniform(-0.08, 0.08, (len(idx), 3, 1, 1))).float().to(device)
            x = (x * gain + bias).clamp_(0, 1)
            heat = torch.from_numpy(th[idx].astype(np.float32)).to(device)
            size = torch.from_numpy(ts[idx].astype(np.float32)).to(device)
            mask = torch.from_numpy(tm[idx].astype(np.float32)).to(device)[:, None]
            out_heat, out_size = net(x)
            size_loss = (torch.abs(out_size - size) * mask).sum() / mask.sum().clamp(min=1)
            loss = focal_loss(out_heat, heat) + 0.1 * size_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val_loss = 0.0
            for start in range(0, len(vx), 32):
                x = torch.from_numpy(vx[start : start + 32]).permute(0, 3, 1, 2).float().div_(255.0).to(device)
                target = torch.from_numpy(vh[start : start + 32].astype(np.float32)).to(device)
                val_loss += focal_loss(net(x)[0], target).item() * len(x)
        validation = val_loss / len(vx)
        improved = validation < best_validation
        print(
            f"epoch {epoch + 1}: train {total / len(tx):.4f}  val {validation:.4f}"
            + ("  best" if improved else "")
            + f"  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        if improved:
            best_validation = validation
            save_net(net, args.out, init={"width": args.width}, heatmaps=HEATMAPS, validation_loss=validation)


if __name__ == "__main__":
    main()
