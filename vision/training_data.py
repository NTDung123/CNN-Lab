"""Phase-4 CNN input dataset, with source-group and normalized-pixel exclusion.

Real test photos are never proposed/normalized here. Development supplies extra
training examples; validation supplies untouched diagnostic candidates. The
frozen synthetic validation/test image hashes are used only for exclusion.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets import ROOT, file_sha256, load_shape_split, shape_fingerprint
from .models import SHAPE_CLASSES, SYNTHETIC_CLASSES
from .metrics import iou
from .processing import (
    centered_crop,
    closing,
    connected_components,
    dilate,
    erode,
    fill_holes,
    grayscale,
    normalize_shape,
)
from .real_data import audit_real_manifest, read_real_manifest, real_image_path
from .segmentation import oriented_extent, region_proposals
from .synthetic import raster_shape


def _candidate_data(case, root):
    with Image.open(real_image_path(case, root)) as im:
        rgb = np.asarray(im.convert("RGB"))
    candidates, _ = region_proposals(grayscale(rgb), "repaired_regions", {}, rgb=rgb)
    truth = case["annotations"] + [
        a for a in case.get("distractors", []) if a["class"] in ("ellipse", "triangle")
    ]
    output = []
    for component, mask, bank in candidates:
        pairs = sorted(((iou(component.bbox, a["bbox"]), a) for a in truth), key=lambda v: v[0], reverse=True)
        overlap, annotation = pairs[0] if pairs else (0.0, None)
        if overlap >= 0.62 and not (len(pairs) > 1 and pairs[1][0] >= 0.50):
            label = SHAPE_CLASSES.index(annotation["class"])
        elif overlap < 0.15 and oriented_extent(mask) < 0.78:
            label = SHAPE_CLASSES.index("background")
        else:
            continue
        normalized = normalize_shape(mask, fill=False)
        output.append((normalized, label, case["source_group"], case["id"], bank))
    return output


def _rotate(a, angle):
    """Inverse-rotation bilinear sampler with zero outside the CNN canvas."""
    h, w = a.shape
    yy, xx = np.indices(a.shape, dtype=np.float32)
    x = xx - (w - 1) / 2
    y = yy - (h - 1) / 2
    c, s = np.cos(angle), np.sin(angle)
    sx = c * x + s * y + (w - 1) / 2
    sy = -s * x + c * y + (h - 1) / 2
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    out = np.zeros_like(a, np.float32)
    for dx, dy, weight in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        x1, y1 = x0 + dx, y0 + dy
        valid = (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)
        out[valid] += a[y1[valid], x1[valid]] * weight[valid]
    return out


def _augment_real(a, rng):
    a = _rotate(a, float(rng.uniform(-np.pi, np.pi)))
    mask = a > float(rng.uniform(0.20, 0.70))
    mode = int(rng.integers(5))
    if mode == 0:
        mask = erode(mask)
    elif mode == 1:
        mask = dilate(mask)
    return normalize_shape(mask, fill=False)


def _synthetic_variant(label, rng):
    kind = SYNTHETIC_CLASSES[label]
    rx = float(rng.uniform(16, 27))
    ry = rx if kind == "circle" else float(rng.uniform(12, 27))
    if kind == "ellipse":
        ry = rx * rng.uniform(0.40, 0.73)
    angle = float(rng.uniform(-np.pi, np.pi))
    mask = raster_shape(kind, 80, 80, rng.uniform(35, 45), rng.uniform(35, 45), rx, ry, angle)
    # Small boundary defects / chamfers, while keeping the target silhouette.
    if kind == "rectangle" and rng.random() < 0.7:
        yy, xx = np.indices(mask.shape)
        cy, cx = np.nonzero(mask)
        cx0, cy0 = np.mean(cx), np.mean(cy)
        u = (xx - cx0) * np.cos(angle) + (yy - cy0) * np.sin(angle)
        v = -(xx - cx0) * np.sin(angle) + (yy - cy0) * np.cos(angle)
        mask &= np.abs(u) + np.abs(v) <= rx + ry - rng.uniform(1, 5)
    if rng.random() < 0.4:
        yy, xx = np.indices(mask.shape)
        noise = np.sin(xx * rng.uniform(0.3, 0.8) + yy * rng.uniform(0.2, 0.7))
        boundary = mask & ~erode(mask)
        mask &= ~(boundary & (noise > 0.86))
    return normalize_shape(mask, fill=False)


def _background_variant(rng):
    # Irregular connected strokes/blobs; do not label ideal circles or filled
    # rectangles as background, since their normalized inputs are ambiguous.
    mask = np.zeros((64, 64), bool)
    x, y = 32, 32
    for _ in range(int(rng.integers(5, 16))):
        nx, ny = np.clip([x + rng.integers(-20, 21), y + rng.integers(-20, 21)], 5, 58)
        steps = int(max(abs(nx - x), abs(ny - y))) + 1
        xx = np.rint(np.linspace(x, nx, steps)).astype(int)
        yy = np.rint(np.linspace(y, ny, steps)).astype(int)
        mask[yy, xx] = True
        x, y = int(nx), int(ny)
    mask = dilate(mask, int(rng.choice([3, 5, 7])))
    parts = connected_components(mask, min_area=10)
    if not parts:
        return None
    mask = fill_holes(max(parts, key=lambda c: c.area).mask)
    if oriented_extent(mask) >= 0.76:
        return None
    return normalize_shape(mask, fill=False)


def prepare_real_training(root=ROOT, seed=20261115, force=False):
    root = Path(root)
    path = root / "data/real_training.npz"
    mp = root / "data/real_training.json"
    dependencies = {
        "manifest_sha256": file_sha256(root / "data/real/manifest.json"),
        "geometry_sha256": file_sha256(root / "data/geometry_splits.npz"),
        "processing_sha256": file_sha256(root / "vision/processing.py"),
        "segmentation_sha256": file_sha256(root / "vision/segmentation.py"),
        "builder_sha256": file_sha256(Path(__file__)),
    }
    if path.exists() and mp.exists() and not force:
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m["archive_sha256"] != file_sha256(path):
            raise ValueError("Extra CNN dataset changed. Rebuild explicitly with --force, then retrain/evaluate.")
        changed_data = sorted(
            key for key in ("manifest_sha256", "geometry_sha256")
            if m["dependencies"].get(key) != dependencies[key]
        )
        if changed_data:
            raise ValueError(
                "Hybrid training data sources changed (" + ", ".join(changed_data) + "). "
                "Rebuild explicitly with --force, then retrain/evaluate."
            )
        # Tệp dữ liệu nguyên vẹn nhưng được tạo bằng mã cũ hơn (ví dụ dữ liệu giai đoạn 4 dùng
        # tiếp ở giai đoạn 5). Dữ liệu vẫn dùng được; ghi rõ mã nào đã đổi kể từ khi tạo.
        changed = sorted(k for k in dependencies if m["dependencies"].get(k) != dependencies[k])
        if changed:
            m = {**m, "code_changed_since_build": changed,
                 "note_code_changed": "Archive built by earlier code; rebuild with --force to regenerate with current code."}
        return m
    audit_real_manifest(root)
    manifest = read_real_manifest(root)
    rng = np.random.default_rng(seed)
    base_x, base_y = load_shape_split("train", root)
    train_seen = {shape_fingerprint(x) for x in base_x}
    blocked = set()
    # Hash-only exclusion, no reserved labels or performance accessed.
    with np.load(root / "data/geometry_splits.npz", allow_pickle=False) as frozen:
        for key in ("x_validation", "x_test"):
            blocked.update(shape_fingerprint(x) for x in frozen[key])
    val_records, train_records = [], []
    counts = Counter()
    opened = []
    for case in manifest["cases"]:
        if case["split"] == "test":
            continue
        opened.append({"id": case["id"], "role": case["split"], "source_group": case["source_group"]})
        records = _candidate_data(case, root)
        (val_records if case["split"] == "validation" else train_records).extend(records)
    # Validate first, so no augmentation can duplicate a real validation input.
    vx, vy, vg, vc = [], [], [], []
    val_seen = set()
    for a, label, group, case, _ in val_records:
        key = shape_fingerprint(a)
        if key in val_seen or key in train_seen or key in blocked:
            counts["validation_duplicate_exclusions"] += 1
            continue
        val_seen.add(key)
        blocked.add(key)
        vx.append(a)
        vy.append(label)
        vg.append(group)
        vc.append(case)
    extra, labels, groups, cases = [], [], [], []

    def accept(a, label, group, case):
        if a is None:
            return False
        key = shape_fingerprint(a)
        if key in blocked or key in train_seen:
            counts["training_duplicate_exclusions"] += 1
            return False
        train_seen.add(key)
        extra.append(a)
        labels.append(label)
        groups.append(group)
        cases.append(case)
        return True

    positive_real = []
    negative_real = []
    for a, label, group, case, bank in train_records:
        if accept(a, label, group, case):
            (negative_real if label == 4 else positive_real).append((a, label, group, case))
    counts["real_positive_seed_inputs"] = len(positive_real)
    counts["real_background_seed_inputs"] = len(negative_real)
    for _ in range(1600):
        if not positive_real:
            break
        a, label, group, case = positive_real[int(rng.integers(len(positive_real)))]
        accept(_augment_real(a, rng), label, group, case)
    for i in range(4000):
        label = i % 4
        accept(_synthetic_variant(label, rng), label, "synthetic_training_v4", "synthetic_boundary_variant")
    bg_count = sum(label == 4 for label in labels)
    attempts = 0
    while bg_count < 2400 and attempts < 30000:
        attempts += 1
        if negative_real and rng.random() < 0.6:
            a, _, group, case = negative_real[int(rng.integers(len(negative_real)))]
            a = _augment_real(a, rng)
        else:
            a = _background_variant(rng)
            group = "synthetic_training_v4"
            case = "synthetic_background"
        if accept(a, 4, group, case):
            bg_count += 1
    if bg_count < 1200:
        raise ValueError("Not enough distinct background samples.")
    tx = np.concatenate((base_x, np.asarray(extra, np.float32)))
    ty = np.concatenate((base_y, np.asarray(labels, np.int64)))
    canonical_groups = ["synthetic_train_v2"] * len(base_x)
    canonical_cases = ["synthetic_train_v2"] * len(base_x)
    np.savez_compressed(
        path,
        x_train=tx,
        y_train=ty,
        groups_train=np.asarray(canonical_groups + groups),
        cases_train=np.asarray(canonical_cases + cases),
        x_real_validation=np.asarray(vx, np.float32),
        y_real_validation=np.asarray(vy, np.int64),
        groups_real_validation=np.asarray(vg),
        cases_real_validation=np.asarray(vc),
    )
    hashes = [shape_fingerprint(x) for x in tx]
    vh = {shape_fingerprint(x) for x in vx}
    train_groups = set(canonical_groups + groups)
    val_groups = set(vg)
    if len(set(hashes)) != len(hashes) or set(hashes) & vh or train_groups & val_groups:
        raise ValueError("CNN input / group leakage.")
    if not all(np.count_nonzero(np.asarray(vy) == i) > 0 for i in (0, 1, 4)):
        raise ValueError("Real validation must contain circle, rectangle and background inputs.")
    m = {
        "version": 4,
        "seed": seed,
        "classes": SHAPE_CLASSES,
        "dependencies": dependencies,
        "archive_sha256": file_sha256(path),
        "train_samples": len(ty),
        "train_class_counts": dict(Counter(str(int(y)) for y in ty)),
        "real_validation_samples": len(vy),
        "real_validation_class_counts": dict(Counter(str(int(y)) for y in vy)),
        "synthetic_validation_samples": 1600,
        "source_cases_opened": opened,
        "real_test_cases_opened": [],
        "within_training_duplicate_inputs": 0,
        "training_real_validation_overlap": 0,
        "training_reserved_synthetic_overlap": 0,
        "source_groups_crossing_train_validation": 0,
        "generation": dict(counts),
        "policy": "Development candidates only for real training; validation candidates never augmented or trained. New real test never proposed or inferred. Synthetic reserved input hashes exclude exact duplicates; labels not consulted.",
        "limits": "Bbox-overlap candidate labels are weak supervision, manually defined outer-object policy; not pixel-mask ground truth. Near duplicates remain possible.",
    }
    mp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m
