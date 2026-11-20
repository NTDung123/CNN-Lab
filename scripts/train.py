"""Train reproducible CNNs. The test split is deliberately not loaded here."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import math
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from vision.models import SHAPE_CLASSES, SmallCNN
from vision.datasets import DEFAULT_COUNTS, load_shape_split, prepare_shape_splits


def setup(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def mnist_training_data(path, seed):
    with np.load(path, allow_pickle=False) as data:
        # Only the original training partition is accessed by this function.
        x, y = data["x_train"], data["y_train"].astype(np.int64)
    if x.shape != (60000, 28, 28) or y.shape != (60000,):
        raise ValueError("Cần bộ MNIST chuẩn gồm 60.000 ảnh train.")
    order = np.random.default_rng(seed).permutation(len(y))
    val, train = order[:5000], order[5000:]
    return (x[train].astype(np.float32) / 255, y[train]), (x[val].astype(np.float32) / 255, y[val])


@torch.inference_mode()
def evaluate(model, x, y):
    model.eval()
    losses, correct = 0.0, 0
    for start in range(0, len(y), 256):
        xb, yb = x[start : start + 256], y[start : start + 256]
        logits = model(xb)
        losses += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        correct += int((logits.argmax(1) == yb).sum())
    return losses / len(y), correct / len(y)


def translated_batch(x, rng):
    """Explicit zero-filled integer translation, not a built-in vision transform."""
    dy, dx = map(int, rng.integers(-2, 3, size=2))
    result = torch.zeros_like(x)
    h, w = x.shape[-2:]
    ya, yb, xa, xb = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    result[:, :, ya:yb, xa:xb] = x[:, :, ya - dy : yb - dy, xa - dx : xb - dx]
    return result


def train_one(task, args):
    if task == "shapes":
        from scripts.train_geometry import train_geometry

        return train_geometry(args)
    setup(args.seed, args.threads)
    if task == "digits":
        (x, y), (vx, vy) = mnist_training_data(ROOT / "data" / "mnist.npz", args.seed)
        model = SmallCNN(10, 28)
        dataset = "MNIST official train split, 55,000 train / 5,000 validation; test untouched"
        augmentation = "zero-filled translations dx,dy in [-2,2] on training batches only"
    if args.limit_train:
        x, y = x[: args.limit_train], y[: args.limit_train]
    tx, ty = torch.from_numpy(x[:, None]), torch.from_numpy(y)
    tvx, tvy = torch.from_numpy(vx[:, None]), torch.from_numpy(vy)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    epochs = args.epochs or (12 if task == "digits" else 10)
    history, best, started = [], -1.0, time.time()
    destination = ROOT / "weights" / f"{task}_cnn.pt"
    destination.parent.mkdir(exist_ok=True)
    print(
        f"{task}: {len(y)} train, {len(vy)} validation, {epochs} epochs, CPU threads={args.threads}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        tick = time.time()
        if task == "digits":
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (0.5 ** ((epoch - 1) // 5))
        model.train()
        order = rng.permutation(len(y))
        loss_sum, correct = 0.0, 0
        for start in range(0, len(y), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb, yb = tx[idx], ty[idx]
            if task == "digits":
                xb = translated_batch(xb, rng)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)
            correct += int((logits.detach().argmax(1) == yb).sum())
        val_loss, val_acc = evaluate(model, tvx, tvy)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(y),
            "train_accuracy": correct / len(y),
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "seconds": time.time() - tick,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_acc > best:
            best = val_acc
            metadata = {
                "task": task,
                "dataset": dataset,
                "seed": args.seed,
                "selected_epoch": epoch,
                "training_samples": len(y),
                "validation_samples": len(vy),
                "validation_accuracy": val_acc,
                "architecture": "Conv3x3(1,16)-ReLU-MaxPool2-Conv3x3(16,32)-ReLU-MaxPool2-FC64-Dropout0.15-FC",
                "classes": [str(i) for i in range(10)] if task == "digits" else SHAPE_CLASSES,
                "augmentation": augmentation,
                "torch_version": str(torch.__version__),
            }
            torch.save({"state_dict": model.state_dict(), "metadata": metadata}, destination)
    result = {"task": task, "epochs_run": epochs, "total_seconds": time.time() - started, "history": history}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"training_{task}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved best validation checkpoint: {destination.name}; validation accuracy={best:.4%}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["digits", "shapes", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--shape-samples", type=int, default=8000)
    parser.add_argument(
        "--limit-train", type=int, default=0, help="Smoke training only; 0 uses the full training split"
    )
    args = parser.parse_args()
    if args.epochs < 0 or min(args.batch_size, args.threads, args.shape_samples) < 1 or args.limit_train < 0:
        parser.error("Số mẫu, batch và threads phải dương; epochs và limit-train không được âm.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate phải hữu hạn và dương.")
    if args.task in ("shapes", "all") and (args.limit_train or args.shape_samples != 8000):
        parser.error(
            "Geometry dùng bộ dữ liệu đã khóa; --limit-train chỉ dành cho digits và --shape-samples phải là 8000."
        )
    if args.task in ("digits", "all"):
        train_one("digits", args)
    if args.task in ("shapes", "all"):
        train_one("shapes", args)


if __name__ == "__main__":
    main()
