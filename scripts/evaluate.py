"""Independent tests: MNIST 10k, synthetic shapes, end-to-end detection, failures.

This command never updates weights. Checkpoint selection uses validation only.
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from vision.models import SHAPE_CLASSES, load_cnn, predict
from vision.metrics import classification_metrics, detection_score, summarize_detection
from vision.datasets import (
    DEFAULT_COUNTS,
    audit_splits,
    geometry_fingerprints,
    load_shape_split,
    shape_fingerprint,
)
from vision.processing import (
    closing,
    connected_components,
    foreground,
    normalize_shape,
    preprocess_digit,
    resize_bilinear,
)
from vision.service import Recognizer
from vision.synthetic import SEEDS, make_scene, make_shape_dataset
from vision.report_files import evaluation_directory, write_json_atomic

RESULTS = ROOT / "results"


def predict_batches(model, x):
    return np.concatenate([predict(model, x[i : i + 256]) for i in range(0, len(x), 256)])


def failure_grid(images, labels, probabilities, filename):
    predictions = probabilities.argmax(1)
    wrong = np.flatnonzero(predictions != labels)
    examples = [
        {
            "index": int(i),
            "true": int(labels[i]),
            "predicted": int(predictions[i]),
            "confidence": float(probabilities[i].max()),
        }
        for i in wrong[:24]
    ]
    grid = Image.new("RGB", (8 * 110, max(1, (len(examples) + 7) // 8) * 123), "#f5f7fa")
    draw = ImageDraw.Draw(grid)
    for k, item in enumerate(examples):
        i = item["index"]
        a = np.rint(resize_bilinear(images[i], (84, 84)) * 255).astype(np.uint8)
        x, y = (k % 8) * 110 + 12, (k // 8) * 123 + 5
        grid.paste(Image.fromarray(a).convert("RGB"), (x, y))
        draw.text((x - 5, y + 87), f"True {labels[i]} / Pred {predictions[i]}", fill="#1e293b")
        draw.text((x - 5, y + 102), f"#{i}  {item['confidence']:.1%}", fill="#59677a")
    if not examples:
        draw.text((20, 20), "No misclassified examples in this test split.", fill="black")
    grid.save(RESULTS / filename)
    return {"total_failures": len(wrong), "first_examples": examples, "image": filename}


def _scene_input_hashes(image, recognizer, prepared=None):
    """Hash the ACTUAL candidate tensors before any CNN inference."""
    prepared = prepared or recognizer.prepare_geometry(image)
    return [shape_fingerprint(im) for im in prepared["normalized"]]


def evaluate_scenes(recognizer, count, stress=False, blocked_fingerprints=None):
    # Reuse phase-3 scenes as a fixed regression suite. Do not silently pick
    # easier scenes based on a new detector. The independent 1200-shape
    # classification split remains separate from this reused scene suite.
    history = ROOT / "results/history/v3/scene_predictions.json"
    previous = json.loads(history.read_text()) if history.exists() else {}
    frozen = previous.get("stress" if stress else "supported", [])
    rows, conditions, records = [], {}, []
    seen = set()
    with np.load(ROOT / "data/real_training.npz", allow_pickle=False) as training:
        train_ids = {shape_fingerprint(im) for im in training["x_train"]}
    overlaps_train, overlaps_scenes, repeated = set(), set(), 0
    for i in range(count):
        condition = (
            "touching" if stress else ["clean", "gaussian_and_impulse_noise", "dark_background"][i % 3]
        )
        seed = frozen[i]["seed"] if i < len(frozen) else SEEDS["scenes"] + (10000 if stress else 0) + i
        if i < len(frozen):
            condition = frozen[i]["condition"]
        image, truth = make_scene(
            seed,
            count=3 if stress else 6,
            invert=condition == "dark_background",
            noise=18 if condition == "gaussian_and_impulse_noise" else 0,
            touch=stress,
        )
        prepared = recognizer.prepare_geometry(image)
        hashes = _scene_input_hashes(image, recognizer, prepared)
        identities = set(hashes)
        overlaps_train.update(identities & train_ids)
        overlaps_scenes.update(identities & seen)
        repeated += len(hashes) - len(identities)
        seen.update(identities)
        result = recognizer.geometry(image, previews=False, prepared=prepared)
        score = detection_score(truth, result["objects"])
        rows.append(score)
        conditions.setdefault(condition, []).append(score)
        mistakes = sum(v["fp"] + v["fn"] for v in score[0].values())
        record = {
            "seed": seed,
            "condition": condition,
            "ground_truth": truth,
            "predictions": result["objects"],
            "target_errors": mistakes,
            "actual_unique_cnn_inputs": len(identities),
            "overlap_with_training_inputs": len(identities & train_ids),
        }
        if (mistakes and sum("image" in r for r in records) < 6) or (stress and i == 0):
            detail = recognizer.geometry(image, previews=True, prepared=prepared)
            name = f"scene_{condition}_{i:03d}.png"
            (RESULTS / name).write_bytes(base64.b64decode(detail["annotated"].split(",")[1]))
            record["image"] = name
        records.append(record)
        if (i + 1) % 15 == 0:
            print(f"Scene regression {condition}: {i + 1}/{count}", flush=True)
    summary = summarize_detection(
        rows, count, count * (3 if stress else 6), recognizer.profile["min_confidence"]
    )
    summary["conditions"] = {
        k: summarize_detection(v, len(v), len(v) * (3 if stress else 6)) for k, v in conditions.items()
    }
    summary["normalized_input_deduplication"] = {
        "passed": not overlaps_train,
        "actual_input_resolution": "32x32 quantized uint8",
        "actual_unique_cnn_inputs": len(seen),
        "overlap_with_training_inputs": len(overlaps_train),
        "duplicates_across_regression_scenes": len(overlaps_scenes),
        "repeated_inputs_from_multiple_branches": repeated,
        "checked_before_inference": True,
        "scope": "All actual normalized proposal inputs, including every polarity/color/hull branch. Reused fixed scenes are regression, not a newly unseen scene benchmark. No scene removed by score or post-hoc identity.",
    }
    summary["evaluation_role"] = "fixed_phase3_regression"
    return summary, records


def save_examples(x, y):
    directory = ROOT / "examples"
    directory.mkdir(exist_ok=True)
    sources = {}
    for digit in range(10):
        idx = int(np.flatnonzero(y == digit)[0])
        a = 255 - resize_bilinear(x[idx] * 255, (280, 280)).astype(np.uint8)
        Image.fromarray(a).save(directory / f"digit_{digit}.png")
        sources[str(digit)] = idx
    for name, seed, options in [
        ("shapes_mixed", 1, {}),
        ("shapes_dark", 12, {"invert": True}),
        ("shapes_noise", 17, {"noise": 18}),
        ("shapes_touching", 23, {"touch": True, "count": 3}),
    ]:
        image, truth = make_scene(seed, **options)
        Image.fromarray(image).save(directory / f"{name}.png")
        (directory / f"{name}.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    (directory / "mnist_sample_indices.json").write_text(json.dumps(sources, indent=2))


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-count", type=int, default=90)
    parser.add_argument("--shape-test-count", type=int, default=1200)
    parser.add_argument(
        "--exploratory-real",
        action="store_true",
        help="Mark real-photo results exploratory if model or processing changed",
    )
    args = parser.parse_args()
    if args.scene_count < 1 or args.shape_test_count < 4:
        parser.error("Cần ít nhất 1 ảnh scene và 4 mẫu phân loại hình.")
    partial = args.scene_count != 90 or args.shape_test_count != 1200
    RESULTS = evaluation_directory(ROOT, exploratory=args.exploratory_real, partial=partial)
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    recognizer = Recognizer()
    dataset_path = ROOT / "data" / "mnist.npz"
    with np.load(dataset_path, allow_pickle=False) as d:
        x, y = d["x_test"].astype(np.float32) / 255, d["y_test"].astype(int)
    if x.shape != (10000, 28, 28):
        raise ValueError("Cần MNIST official test 10.000 ảnh.")
    raw_probs = predict_batches(recognizer.digits, x)
    raw_metrics = classification_metrics(y, raw_probs.argmax(1), [str(i) for i in range(10)])
    print(f"MNIST raw test: {raw_metrics['accuracy']:.4%}", flush=True)
    pipeline_inputs = np.stack([preprocess_digit(np.rint(255 - im * 255).astype(np.uint8))[0] for im in x])
    pipe_probs = predict_batches(recognizer.digits, pipeline_inputs)
    pipeline_metrics = classification_metrics(y, pipe_probs.argmax(1), [str(i) for i in range(10)])
    print(f"MNIST through actual image pipeline: {pipeline_metrics['accuracy']:.4%}", flush=True)
    # Existing ANN checkpoint is a baseline only, with its original weights.
    from reference_original.ann import forward_prop, load_model

    params = load_model(ROOT / "reference_original" / "model_infinity.npz")
    ann_predictions = []
    for start in range(0, len(y), 256):
        *_, probabilities = forward_prop(*params, x[start : start + 256].reshape(-1, 784).T)
        ann_predictions.extend(probabilities.argmax(0).tolist())
    ann_metrics = classification_metrics(y, ann_predictions, [str(i) for i in range(10)])
    print(f"Original provided ANN on same raw test set: {ann_metrics['accuracy']:.4%}", flush=True)
    all_splits = {name: load_shape_split(name) for name in DEFAULT_COUNTS}
    geometry_audit = audit_splits(all_splits)
    if not geometry_audit["passed"]:
        raise AssertionError("Geometry splits contain duplicate normalized inputs.")
    geometry_manifest = json.loads((ROOT / "data" / "geometry_splits.json").read_text(encoding="utf-8"))
    if recognizer.shape_metadata.get("geometry_dataset_sha256") != geometry_manifest["archive_sha256"]:
        raise ValueError(
            "Shape checkpoint was not trained using this deduplicated geometry dataset. Retrain shapes first."
        )
    sx, sy = all_splits["test"]
    if args.shape_test_count > len(sy):
        raise ValueError("Requested more test samples than the frozen geometry test split contains.")
    sx, sy = sx[: args.shape_test_count], sy[: args.shape_test_count]
    shape_probs = predict_batches(recognizer.shapes, sx)
    shape_metrics = classification_metrics(sy, shape_probs.argmax(1), recognizer.shape_classes)
    blocked = geometry_fingerprints()
    detection, scenes = evaluate_scenes(recognizer, args.scene_count, blocked_fingerprints=blocked)
    stress, stress_scenes = evaluate_scenes(recognizer, 12, stress=True, blocked_fingerprints=blocked)
    from scripts.evaluate_real import evaluate_real_report

    real = evaluate_real_report(recognizer, ROOT, exploratory=args.exploratory_real, include_test=True)
    if RESULTS == ROOT / "results":
        save_examples(x, y)
    raw_failures = failure_grid(x, y, raw_probs, "digit_failures.png")
    pipeline_failures = failure_grid(pipeline_inputs, y, pipe_probs, "pipeline_failures.png")
    result = {
        "asset_identity": recognizer.asset_identity,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "checkpoint_selection": "Digits: highest MNIST validation accuracy. Shapes: full-frame validation target macro-F1 with synthetic validation accuracy guard; region parameters chosen on validation only. Test never used for training or selection.",
            "mnist_train": 55000,
            "mnist_validation": 5000,
            "mnist_test": 10000,
            "geometry_seeds": SEEDS,
            "shape_training_samples": recognizer.shape_metadata["training_samples"],
            "shape_validation_samples": recognizer.shape_metadata["validation_samples"],
            "geometry_deduplication_protocol_version": geometry_manifest["protocol_version"],
            "geometry_dataset_sha256": geometry_manifest["archive_sha256"],
            "shape_data_scope": "Synthetic classification test is reserved; scene suite reuses fixed phase-3 scenes as regression. Real data roles are separate.",
            "geometry_training_v4": json.loads((ROOT / "data/real_training.json").read_text()),
            "stress_scope": "touching/overlapping objects deliberately excluded from supported-scene metrics",
        },
        "dataset": {
            "name": "MNIST",
            "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        },
        "digits": {
            "cnn_raw": raw_metrics,
            "cnn_pipeline": pipeline_metrics,
            "original_ann_raw": ann_metrics,
            "baseline_note": "provided EMNIST/Infinity ANN; no retraining; comparison is checkpoint behavior, not a controlled architecture ablation",
            "raw_failures": raw_failures,
            "pipeline_failures": pipeline_failures,
        },
        "shapes": {
            "classification": shape_metrics,
            "detection": detection,
            "stress_touching": stress,
            "deduplication": geometry_audit,
        },
        "real_images": {
            "audit": real["audit"],
            "protocol": real["protocol"],
            "detection_all": real["detection_all"],
            "detection_by_role": real["detection_by_role"],
            "detection_by_difficulty": real["detection_by_difficulty"],
            "roi_diagnostic": real["roi_diagnostic"],
            "failure_counts": real["failure_counts"],
            "full_results": ("../" if RESULTS != ROOT / "results" else "") + real["report_file"],
        },
        "elapsed_seconds": time.time() - started,
        "limits": [
            "One digit at a time; no alphabetic or multi-digit OCR.",
            "Synthetic classification, reused scene regression and real-photo roles are reported separately. This pilot is not representative of all photos.",
            "Touching/overlapping objects can merge into one connected component.",
            "Very thin/open contours, complex backgrounds, low contrast, clipped shapes and unknown objects can fail.",
            "Softmax scores are not calibrated probabilities of correctness.",
        ],
    }
    write_json_atomic(RESULTS / "evaluation.json", result)
    write_json_atomic(RESULTS / "scene_predictions.json", {"supported": scenes, "stress": stress_scenes})
    for name, value in [
        ("digits_raw", raw_metrics),
        ("digits_pipeline", pipeline_metrics),
        ("shapes", shape_metrics),
    ]:
        np.savetxt(RESULTS / f"confusion_{name}.csv", value["confusion_matrix"], delimiter=",", fmt="%d")
    print(
        json.dumps(
            {
                "mnist_raw_accuracy": raw_metrics["accuracy"],
                "mnist_pipeline_accuracy": pipeline_metrics["accuracy"],
                "original_ann_accuracy": ann_metrics["accuracy"],
                "shape_classification_accuracy": shape_metrics["accuracy"],
                "detection": detection,
                "stress": stress,
                "seconds": result["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
