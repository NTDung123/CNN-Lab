"""Synthetic geometry drawn with NumPy equations, without image libraries.

Seeds provide reproducibility. Actual split separation is enforced by hashes
of normalized CNN inputs in vision.datasets, not by seeds alone.
"""
import numpy as np

from .models import SYNTHETIC_CLASSES as SHAPE_CLASSES
from .processing import normalize_shape

SEEDS = {"train": 20261101, "validation": 20261102, "test": 20261103, "scenes": 20261104}


def raster_shape(kind, h, w, cx, cy, rx, ry, angle=0, outline=0):
    yy, xx = np.indices((h, w), dtype=np.float32)
    c, s = np.cos(angle), np.sin(angle)
    u, v = (xx - cx) * c + (yy - cy) * s, -(xx - cx) * s + (yy - cy) * c

    def inside(ax, ay):
        if ax <= 0 or ay <= 0:
            return np.zeros((h, w), bool)
        if kind in ("circle", "ellipse"):
            return (u / ax) ** 2 + (v / ay) ** 2 <= 1
        if kind == "rectangle":
            return (np.abs(u) <= ax) & (np.abs(v) <= ay)
        if kind == "triangle":
            return (v >= -ay) & (v <= ay) & (np.abs(u) <= ax * (v + ay) / (2 * ay))
        raise ValueError(kind)

    outer = inside(rx, ry)
    if outline:
        # Triangle hollowing is unnecessary for negative examples.
        return outer & ~inside(rx - outline, ry - outline) if kind != "triangle" else outer
    return outer


def make_shape_dataset(count, seed, excluded_hashes=None, return_audit=False):
    from .datasets import shape_fingerprint
    rng = np.random.default_rng(seed)
    images, labels = [], []
    external = set() if excluded_hashes is None else set(excluded_hashes)
    selected = set()
    attempts, rejected_external, rejected_internal = 0, 0, 0
    while len(images) < count:
        attempts += 1
        if attempts > max(10000, count * 100):
            raise RuntimeError("Unable to generate enough unique shapes under this configuration.")
        label = len(images) % len(SHAPE_CLASSES)
        kind = SHAPE_CLASSES[label]
        rx = float(rng.uniform(12, 24))
        ry = rx if kind == "circle" else float(rng.uniform(9, 23))
        if kind == "ellipse":
            ry = rx * rng.uniform(.40, .73)
        angle = float(rng.uniform(-np.pi, np.pi))
        mask = raster_shape(kind, 80, 80, rng.uniform(36, 44), rng.uniform(36, 44), rx, ry, angle)
        # Rasterization and subpixel shifts produce independent pixel boundaries.
        normalized = normalize_shape(mask, fill=False)
        fingerprint = shape_fingerprint(normalized)
        if fingerprint in external:
            rejected_external += 1
            continue
        if fingerprint in selected:
            rejected_internal += 1
            continue
        selected.add(fingerprint)
        images.append(normalized)
        labels.append(label)
    result = np.asarray(images, np.float32), np.asarray(labels, np.int64)
    if return_audit:
        return (*result, {"attempts": attempts, "rejected_against_previous_splits": rejected_external,
                         "rejected_within_split": rejected_internal, "accepted": count})
    return result


def make_scene(seed, count=6, invert=False, noise=0, outlines=True, touch=False):
    rng = np.random.default_rng(seed)
    h, w = 360, 600
    # Stable non-overlapping grid; test generation randomizes actual geometry.
    mask = np.zeros((h, w), bool)
    annotations = []
    for i in range(count):
        kind = SHAPE_CLASSES[int(rng.integers(4))]
        cx, cy = 100 + (i % 3) * 200 + rng.uniform(-18, 18), 88 + (i // 3) * 180 + rng.uniform(-12, 12)
        if touch:
            cx, cy = 220 + i * 34, 180
        rx = float(rng.uniform(26, 45))
        ry = rx if kind == "circle" else float(rng.uniform(20, 42))
        if kind == "ellipse":
            ry = rx * rng.uniform(.40, .70)
        angle = float(rng.uniform(-np.pi, np.pi))
        outline = int(rng.integers(3, 7)) if outlines and rng.random() < .5 else 0
        obj = raster_shape(kind, h, w, cx, cy, rx, ry, angle, outline)
        ys, xs = np.nonzero(obj)
        annotations.append({"class": kind, "bbox": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1]})
        mask |= obj
    bg, fg = (15, 235) if invert else (245, 20)
    gray = np.where(mask, fg, bg).astype(np.float32)
    if noise:
        gray += rng.normal(0, noise, gray.shape)
        if noise >= 12:
            flip = rng.random(gray.shape) < .003
            gray[flip] = rng.integers(0, 256, int(flip.sum()))
    return np.clip(gray, 0, 255).astype(np.uint8), annotations
