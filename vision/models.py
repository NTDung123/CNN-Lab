"""Small CNNs; neural layers use PyTorch as expressly permitted in Chapter 0."""

from pathlib import Path
import pickle

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CLASSES = ["circle", "rectangle", "triangle", "ellipse"]
SHAPE_CLASSES = SYNTHETIC_CLASSES + ["background"]


class ModelLoadError(RuntimeError):
    pass


class SmallCNN(nn.Module):
    def __init__(self, classes=10, image_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (image_size // 4) ** 2, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_cnn(task, root=ROOT):
    path = Path(root) / "weights" / f"{task}_cnn.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path.name}. Chạy python scripts/train.py --task {task}.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        classes = checkpoint["metadata"]["classes"]
        if task == "shapes" and classes not in (SYNTHETIC_CLASSES, SHAPE_CLASSES):
            raise ValueError("Unexpected geometry class order")
        model = SmallCNN(10, 28) if task == "digits" else SmallCNN(len(classes), 32)
        model.load_state_dict(checkpoint["state_dict"])
        metadata = checkpoint["metadata"]
    except (
        OSError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise ModelLoadError(
            f"Không nạp được {path.name}. Khôi phục tệp trọng số hoặc chạy lại scripts/train.py."
        ) from exc
    model.eval()
    return model, metadata


@torch.inference_mode()
def predict(model, images):
    a = np.asarray(images, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    tensor = torch.from_numpy(np.ascontiguousarray(a[:, None]))
    return torch.softmax(model(tensor), 1).numpy()


@torch.inference_mode()
def first_feature_maps(model, image, count=8):
    x = torch.from_numpy(np.asarray(image, np.float32)[None, None].copy())
    a = torch.relu(model.features[0](x))[0, :count].numpy()
    return [m / max(float(m.max()), 1e-8) * 255 for m in a]
