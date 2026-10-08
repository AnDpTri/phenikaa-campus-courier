"""Bộ khởi động Phenikaa Campus Courier v2.

    python starter.py --data ../delivery_public --out predictions.json
    python starter.py --data ../delivery_public --show 0      # vẽ lại chú thích của cảnh train số 0

- Đọc dữ liệu, chấm macro accuracy trên validation giống cách BTC chấm test.
- Baseline rất đơn giản: mỗi robot luôn đoán hướng xuất hiện nhiều nhất trong train (~25%).
- ``--show`` vẽ đồ thị trong scenes.json đè lên ảnh để hiểu định dạng chú thích.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def load(split_dir):
    split_dir = Path(split_dir)
    rows = json.loads((split_dir / "observations.json").read_text(encoding="utf-8"))
    labels_path = split_dir / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else None
    return rows, labels


def macro_accuracy(rows, labels, preds):
    per_robot = defaultdict(list)
    for row, y, p in zip(rows, labels, preds):
        per_robot[row["robot_id"]].append(y == p)
    scores = {r: sum(v) / len(v) for r, v in sorted(per_robot.items())}
    return sum(scores.values()) / len(scores), scores


def show(data, index, out="scene_preview.png"):
    from PIL import Image, ImageDraw
    scenes = json.loads((data / "train" / "scenes.json").read_text(encoding="utf-8"))
    s = scenes[index]
    im = Image.open(data / "train" / s["image"]).convert("RGB")
    d = ImageDraw.Draw(im)
    xy = {tuple(n["rc"]): n["xy"] for n in s["nodes"]}
    colour = {"normal": "grey", "crowded": "orange", "covered": "blue", "closed": "red"}
    for e in s["edges"]:
        a, b = xy[tuple(e["a"])], xy[tuple(e["b"])]
        d.line([tuple(a), tuple(b)], fill=colour[e["status"]], width=3)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        tag = ("S" if e["stairs"] else "") + ("1" if e["oneway_to"] else "")
        if tag:
            d.text((mx + 4, my + 4), tag, fill="black")
    for rc, (x, y) in xy.items():
        d.ellipse((x - 4, y - 4, x + 4, y + 4), outline="magenta", width=2)
    for lm in s["landmarks"]:
        x, y = xy[tuple(lm["rc"])]
        d.text((x + 8, y - 22), lm["type"], fill="magenta")
    rx, ry = xy[tuple(s["robot"]["rc"])]
    d.text((rx + 8, ry + 10), "robot " + s["robot"]["heading"], fill="magenta")
    im.save(out)
    print(json.dumps({k: s[k] for k in ("style", "grid", "robot", "weather", "mission")},
                     ensure_ascii=False, indent=2))
    print(f"đã lưu {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("../delivery_public"))
    ap.add_argument("--out", type=Path, default=Path("predictions.json"))
    ap.add_argument("--show", type=int, help="vẽ chú thích của cảnh train có chỉ số này")
    a = ap.parse_args()
    if a.show is not None:
        show(a.data, a.show)
        return

    train_rows, train_labels = load(a.data / "train")
    majority = {}
    for r in range(10):
        counts = Counter(y for row, y in zip(train_rows, train_labels) if row["robot_id"] == r)
        majority[r] = counts.most_common(1)[0][0]
    print("hướng phổ biến nhất theo robot:", {r: ACTIONS[m] for r, m in majority.items()})

    val_rows, val_labels = load(a.data / "validation")
    val_pred = [majority[row["robot_id"]] for row in val_rows]
    score, per = macro_accuracy(val_rows, val_labels, val_pred)
    print(f"validation macro accuracy: {score:.4f}")
    print("theo robot:", {r: round(s, 3) for r, s in per.items()})

    test_rows, _ = load(a.data / "test")
    test_pred = [majority[row["robot_id"]] for row in test_rows]
    a.out.write_text(json.dumps(test_pred), encoding="utf-8")
    print(f"đã ghi {len(test_pred)} dự đoán vào {a.out}")


if __name__ == "__main__":
    main()
