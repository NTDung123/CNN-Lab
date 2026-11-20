"""Geometry CNN: synthetic rehearsal + development-photo candidates + background.

Select by real validation macro-F1 with a synthetic validation accuracy guard.
No new real test photo, target label or metric is accessed.
"""

import argparse, json, sys, time, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from vision.datasets import file_sha256, load_shape_split
from vision.metrics import classification_metrics
from vision.models import SHAPE_CLASSES, SmallCNN, predict
from vision.training_data import prepare_real_training
from vision.real_data import read_real_manifest, real_image_path
from vision.metrics import detection_score, summarize_detection
from vision.service import Recognizer
from PIL import Image


def validation_detector(model):
    """Use the fixed training protocol without reading deployed checkpoints.

    Model recovery must work even when the demo's digit/shape weights are
    missing. Demo confidence settings must not change epoch selection.
    """
    detector = object.__new__(Recognizer)
    detector.shapes = model
    detector.shape_classes = list(SHAPE_CLASSES)
    detector.profile = {
        "mode": "repaired_regions",
        "min_confidence": 0.75,
        "complex_min_confidence": 0.75,
        "illumination_border_std": 12,
    }
    return detector


def train_geometry(args):
    from scripts.train import setup

    setup(args.seed, args.threads)
    extra = prepare_real_training(ROOT)
    with np.load(ROOT / "data/real_training.npz", allow_pickle=False) as d:
        x, y = d["x_train"], d["y_train"]
        rx, ry = d["x_real_validation"], d["y_real_validation"]
    vx, vy = load_shape_split("validation", ROOT)
    tx, ty = torch.from_numpy(x[:, None]), torch.from_numpy(y)
    model = SmallCNN(len(SHAPE_CLASSES), 32)
    legacy = ROOT / "reference_v3/weights/shapes_cnn.pt"
    if legacy.exists():
        previous = torch.load(legacy, map_location="cpu", weights_only=True)["state_dict"]
        current = model.state_dict()
        for key in previous:
            if key in current and current[key].shape == previous[key].shape:
                current[key].copy_(previous[key])
        current["classifier.4.weight"][:4].copy_(previous["classifier.4.weight"])
        current["classifier.4.bias"][:4].copy_(previous["classifier.4.bias"])
        current["classifier.4.weight"][4].zero_()
        current["classifier.4.bias"][4] = -3.5
        model.load_state_dict(current)
    lr = min(args.lr, 0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(args.seed)
    history = []
    best = -1.0
    best_guarded = False
    started = time.time()
    epochs = args.epochs or 12
    destination = ROOT / "weights/shapes_cnn.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "train_samples": len(y),
                "real_validation_samples": len(ry),
                "synthetic_validation_samples": len(vy),
                "classes": SHAPE_CLASSES,
                "epochs": epochs,
                "lr": lr,
            }
        ),
        flush=True,
    )

    def infer(a):
        return np.concatenate([predict(model, a[i : i + 256]) for i in range(0, len(a), 256)])

    geometry = json.loads((ROOT / "data/geometry_splits.json").read_text())
    detector = validation_detector(model)
    prepared_validation = []
    for case in read_real_manifest()["cases"]:
        if case["split"] != "validation":
            continue
        image = np.asarray(Image.open(real_image_path(case)).convert("RGB"))
        prepared = detector.prepare_geometry(image, "repaired_regions")
        prepared["trace"].pop("branches", None)
        prepared_validation.append((case, image, prepared))
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(y))
        loss_sum = correct = 0
        tick = time.time()
        for start in range(0, len(y), args.batch_size):
            idx = order[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(tx[idx])
            loss = nn.functional.cross_entropy(logits, ty[idx])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
            correct += int((logits.detach().argmax(1) == ty[idx]).sum())
        model.eval()
        sv = classification_metrics(vy, infer(vx).argmax(1), SHAPE_CLASSES)
        rv = classification_metrics(ry, infer(rx).argmax(1), SHAPE_CLASSES)
        detection_rows = []
        for case, image, prepared in prepared_validation:
            prediction = detector.geometry(image, previews=False, prepared=prepared)
            detection_rows.append(detection_score(case["annotations"], prediction["objects"]))
        vd = summarize_detection(
            detection_rows,
            len(prepared_validation),
            sum(len(c["annotations"]) for c, _, _ in prepared_validation),
        )
        target_macro = sum(v["f1"] for v in vd["per_class"].values()) / 2
        guarded = sv["accuracy"] >= 0.995
        score = target_macro + 0.001 * sv["accuracy"] + 0.0001 * rv["macro_f1"]
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(y),
            "train_accuracy": correct / len(y),
            "synthetic_validation_accuracy": sv["accuracy"],
            "real_validation_macro_f1": rv["macro_f1"],
            "real_validation_accuracy": rv["accuracy"],
            "validation_detection": vd,
            "validation_target_macro_f1": target_macro,
            "selection_score": score,
            "synthetic_guard_passed": guarded,
            "seconds": time.time() - tick,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if (guarded and not best_guarded) or (guarded == best_guarded and score > best):
            best, best_guarded = score, guarded
            metadata = {
                "task": "shapes",
                "dataset": "phase4 hybrid: frozen synthetic training + weakly supervised development-photo candidates + irregular background; test untouched",
                "seed": args.seed,
                "selected_epoch": epoch,
                "training_samples": len(y),
                "validation_samples": len(vy) + len(ry),
                "validation_accuracy": rv["accuracy"],
                "validation_macro_f1": rv["macro_f1"],
                "validation_detection": vd,
                "selection_score": score,
                "selection_rule": "Prefer synthetic validation accuracy >= .995; maximize full-frame validation target macro-F1, then synthetic accuracy, then ROI classification macro-F1. Fixed repaired_regions/.75 used for epoch selection; no test score used.",
                "classes": SHAPE_CLASSES,
                "augmentation": "source-group-preserving rotations, resampling, mild erosion/dilation, irregular boundary/chamfer variants; training only",
                "architecture": "Conv3x3(1,16)-ReLU-MaxPool2-Conv3x3(16,32)-ReLU-MaxPool2-FC64-Dropout0.15-FC5",
                "geometry_dataset_sha256": geometry["archive_sha256"],
                "geometry_protocol_version": geometry["protocol_version"],
                "geometry_train_signature": geometry["splits"]["train"]["split_signature"],
                "geometry_validation_signature": geometry["splits"]["validation"]["split_signature"],
                "real_training_archive_sha256": extra["archive_sha256"],
                "real_manifest_sha256": extra["dependencies"]["manifest_sha256"],
                "real_training_roles": ["development"],
                "real_selection_roles": ["validation"],
                "real_test_used": False,
                "torch_version": str(torch.__version__),
            }
            torch.save({"state_dict": model.state_dict(), "metadata": metadata}, destination)
    (ROOT / "results/training_shapes.json").write_text(
        json.dumps(
            {
                "task": "shapes",
                "epochs_run": epochs,
                "total_seconds": time.time() - started,
                "best_selection_score": best,
                "synthetic_validation_guard_passed": best_guarded,
                "real_validation_class_counts": extra["real_validation_class_counts"],
                "history": history,
            },
            indent=2,
        )
    )
    print("Saved validation-selected geometry checkpoint", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0005)
    a = p.parse_args()
    if min(a.epochs, a.threads, a.batch_size) <= 0 or not math.isfinite(a.lr) or a.lr <= 0:
        p.error("Epochs, threads, batch size and learning rate must be finite and positive.")
    train_geometry(a)
