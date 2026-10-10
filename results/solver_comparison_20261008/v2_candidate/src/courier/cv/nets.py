"""Small CNNs for the learned CV stages. All run on CPU in milliseconds per scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .crops import LOOK_CLASSES, NODE_CLASSES, ONEWAY_CLASSES
from .detector import DetectorNet


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class NodeNet(nn.Module):
    """48x48 node crop -> empty / landmark type / robot heading."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(3, width), nn.MaxPool2d(2),  # 24
            _block(width, 2 * width), nn.MaxPool2d(2),  # 12
            _block(2 * width, 4 * width), nn.MaxPool2d(2),  # 6
            _block(4 * width, 4 * width), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * width, len(NODE_CLASSES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class EdgeNet(nn.Module):
    """96x24 strip along a candidate road -> look, stairs, one-way direction."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(3, width), nn.MaxPool2d(2),  # 48x12
            _block(width, 2 * width), nn.MaxPool2d(2),  # 24x6
            _block(2 * width, 4 * width), nn.MaxPool2d((1, 2)),  # 24x3
            _block(4 * width, 4 * width), nn.AdaptiveAvgPool2d((1, 4)),
        )
        hidden = 4 * width * 4
        self.look = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(hidden, len(LOOK_CLASSES)))
        self.stairs = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(hidden, 2))
        self.oneway = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(hidden, len(ONEWAY_CLASSES)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.features(x)
        return self.look(z), self.stairs(z), self.oneway(z)


def to_tensor(batch: np.ndarray, device: torch.device | str | None = None) -> torch.Tensor:
    """N x H x W x 3 uint8 -> N x 3 x H x W float in [0, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float().div_(255.0)
    return tensor.to(device) if device is not None else tensor


def save_net(net: nn.Module, path: str | Path, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in net.state_dict().items()}
    torch.save({"kind": type(net).__name__, "state": state, "meta": meta}, path)


def load_net(path: str | Path, device: torch.device | str | None = None) -> nn.Module:
    # Artifacts only contain tensors and primitive metadata. Restrict loading to
    # that safe subset instead of allowing arbitrary pickle objects to execute.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    net = {"NodeNet": NodeNet, "EdgeNet": EdgeNet, "DetectorNet": DetectorNet}[payload["kind"]](**payload["meta"].get("init", {}))
    net.load_state_dict(payload["state"])
    if device is not None:
        net = net.to(device)
    return net.eval()
