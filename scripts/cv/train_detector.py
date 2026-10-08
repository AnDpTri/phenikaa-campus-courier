"""Train the keypoint detector (node centres, legend road swatches, weather icon)."""

from __future__ import annotations

import argparse
import functools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from courier.cv import load_annotations, load_rgb
from courier.cv.annotations import SceneAnnotation
from courier.cv.detector import HEATMAPS, STRIDE, DetectorNet, letterbox
from courier.cv.types import LEGEND_ROAD_ORDER
from courier.cv.nets import save_net

SIGMA = {"node": 1.0, "swatch": 1.0, "weather": 1.5}


def scene_targets(annotation: SceneAnnotation, input_size: int):
    image, scale = letterbox(load_rgb(annotation.image_path), input_size)
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


def render(points: dict[str, list], out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heat = np.zeros((len(HEATMAPS), out_size, out_size), dtype=np.float32)
    size = np.zeros((2, out_size, out_size), dtype=np.float32)
    mask = np.zeros((out_size, out_size), dtype=np.float32)
    yy, xx = np.mgrid[0:out_size, 0:out_size]
    for channel, name in enumerate(HEATMAPS):
        sigma = SIGMA.get(name, 1.0)
        for cx, cy, w, h in points[name]:
            heat[channel] = np.maximum(heat[channel], np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
            r, c = int(round(cy)), int(round(cx))
            if 0 <= r < out_size and 0 <= c < out_size:
                heat[channel, r, c] = 1.0
                if name != "node":
                    size[:, r, c] = (w, h)
                    mask[r, c] = 1.0
    return heat, size, mask


def build_cache(data: Path, split: str, out: Path, workers: int, input_size: int) -> Path:
    path = out / f"detector_{split}_{input_size}.npz"
    if path.exists():
        with np.load(path) as cached:
            if (
                "heatmaps" in cached
                and tuple(str(name) for name in cached["heatmaps"]) == HEATMAPS
                and int(cached["input_size"]) == input_size
            ):
                return path
        print(f"detector cache schema changed; rebuilding {path}", flush=True)
        path.unlink()
    annotations = load_annotations(data, split)
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(functools.partial(scene_targets, input_size=input_size), annotations, chunksize=8))
    images = np.stack([image for image, _ in results])
    maps = [render(points, input_size // STRIDE) for _, points in results]
    np.savez(
        path,
        heatmaps=np.asarray(HEATMAPS),
        input_size=np.asarray(input_size),
        images=images,
        heat=np.stack([m[0] for m in maps]).astype(np.float16),
        size=np.stack([m[1] for m in maps]).astype(np.float16),
        mask=np.stack([m[2] for m in maps]).astype(np.uint8),
    )
    return path


def degrade_batch(
    images: np.ndarray,
    rng: np.random.Generator,
    jpeg_prob: float,
    jpeg_min: int,
    blur_prob: float,
) -> np.ndarray:
    """Photometric print-like degradation; targets remain geometrically unchanged."""
    if jpeg_prob <= 0 and blur_prob <= 0:
        return images
    result = images.copy()
    for index, image in enumerate(result):
        if rng.random() < blur_prob:
            sigma = float(rng.uniform(0.35, 1.35))
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if rng.random() < jpeg_prob:
            quality = int(rng.integers(jpeg_min, 86))
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("OpenCV could not encode detector augmentation")
            image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        result[index] = image
    return result


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
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--jpeg-prob", type=float, default=0.0)
    parser.add_argument("--jpeg-min", type=int, default=30)
    parser.add_argument("--blur-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--early-stop-min-delta", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.input_size <= 0 or args.input_size % 16:
        parser.error("--input-size must be a positive multiple of 16")
    if not 0 <= args.jpeg_prob <= 1 or not 0 <= args.blur_prob <= 1:
        parser.error("augmentation probabilities must be in [0, 1]")
    if not 1 <= args.jpeg_min <= 85:
        parser.error("--jpeg-min must be in [1, 85]")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if resolved_device == "auto" else resolved_device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    train = np.load(build_cache(args.data, "train", args.cache, args.workers, args.input_size))
    val = np.load(build_cache(args.data, "validation", args.cache, args.workers, args.input_size))
    tx, th, ts, tm = train["images"], train["heat"], train["size"], train["mask"]
    vx, vh = val["images"], val["heat"]

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = DetectorNet(args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps = args.epochs * ((len(tx) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
    best_validation = float("inf")
    meaningful_best = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        net.train()
        started, total = time.perf_counter(), 0.0
        order = rng.permutation(len(tx))
        for start in range(0, len(tx), args.batch):
            idx = np.sort(order[start : start + args.batch])
            augmented = degrade_batch(tx[idx], rng, args.jpeg_prob, args.jpeg_min, args.blur_prob)
            x = torch.from_numpy(augmented).permute(0, 3, 1, 2).float().div_(255.0).to(device)
            gain = torch.from_numpy(rng.uniform(0.8, 1.2, (len(idx), 1, 1, 1))).float().to(device)
            bias = torch.from_numpy(rng.uniform(-0.08, 0.08, (len(idx), 3, 1, 1))).float().to(device)
            x = (x * gain + bias).clamp_(0, 1)
            heat = torch.from_numpy(th[idx].astype(np.float32)).to(device)
            size = torch.from_numpy(ts[idx].astype(np.float32)).to(device)
            mask = torch.from_numpy(tm[idx].astype(np.float32)).to(device)[:, None]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out_heat, out_size = net(x)
                size_loss = (torch.abs(out_size - size) * mask).sum() / mask.sum().clamp(min=1)
                loss = focal_loss(out_heat, heat) + 0.1 * size_loss
            opt.zero_grad()
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            # AMP may skip an optimizer update when it detects overflow. Keep
            # OneCycleLR aligned with real parameter updates in that case.
            if not use_amp or scaler.get_scale() >= scale_before:
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
        meaningful = validation < meaningful_best - args.early_stop_min_delta
        if meaningful:
            meaningful_best = validation
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"epoch {epoch + 1}: train {total / len(tx):.4f}  val {validation:.4f}"
            + ("  best" if improved else "")
            + f"  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        if improved:
            best_validation = validation
            save_net(
                net,
                args.out,
                init={"width": args.width},
                heatmaps=HEATMAPS,
                input_size=args.input_size,
                validation_loss=validation,
                best_epoch=epoch + 1,
                training={
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "jpeg_prob": args.jpeg_prob,
                    "jpeg_min": args.jpeg_min,
                    "blur_prob": args.blur_prob,
                },
            )
        if args.early_stop_patience and stale_epochs >= args.early_stop_patience:
            print(
                f"early stop: no validation improvement >= {args.early_stop_min_delta:g} "
                f"for {stale_epochs} epochs",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
