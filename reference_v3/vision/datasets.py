"""Versioned geometry splits, deduplicated at the actual CNN input resolution.

The identity is SHA-256 of the normalized 32x32 image quantized to uint8.
Labels are intentionally NOT part of identity: identical images cannot cross
splits, even if they are assigned different class labels.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = 2
DEFAULT_COUNTS = {"train": 8000, "validation": 1600, "test": 1200}


def shape_fingerprint(image):
    a = np.asarray(image, dtype=np.float32)
    if a.shape != (32, 32) or not np.all(np.isfinite(a)):
        raise ValueError("Fingerprint requires a finite 32x32 normalized geometry image.")
    canonical = np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_splits(splits):
    identities, details = {}, {}
    exact_identities = {}
    for name, (images, labels) in splits.items():
        hashes = [shape_fingerprint(im) for im in images]
        exact = [hashlib.sha256(np.asarray(im, dtype="<f4").tobytes()).hexdigest() for im in images]
        identities[name], exact_identities[name] = set(hashes), set(exact)
        details[name] = {"samples": len(labels), "unique_normalized_uint8": len(set(hashes)),
                         "duplicates_within_split": len(hashes) - len(set(hashes)),
                         "class_counts": {str(i): int(np.count_nonzero(labels == i)) for i in range(4)},
                         "split_signature": hashlib.sha256("\n".join(f"{h}:{int(y)}" for h, y in zip(hashes, labels)).encode()).hexdigest()}
    overlap, exact_overlap = {}, {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}__{b}"] = len(identities[a] & identities[b])
            exact_overlap[f"{a}__{b}"] = len(exact_identities[a] & exact_identities[b])
    passed = not any(overlap.values()) and not any(d["duplicates_within_split"] for d in details.values())
    return {"protocol_version": PROTOCOL_VERSION,
            "identity": "SHA-256 of round(clip(normalized_32x32,0,1)*255), uint8 row-major; labels excluded",
            "splits": details, "cross_split_overlap": overlap, "exact_float32_cross_split_overlap": exact_overlap,
            "passed": passed, "near_duplicate_scope": "Exact float32 and uint8-equivalent images only; not a guarantee against all perceptually similar shapes."}


def prepare_shape_splits(root=ROOT, counts=None, force=False):
    from .synthetic import SEEDS, make_shape_dataset
    root = Path(root)
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(DEFAULT_COUNTS) or any(int(v) < 4 for v in counts.values()):
        raise ValueError("Need train, validation and test counts of at least four images each.")
    directory = root / "data"
    directory.mkdir(parents=True, exist_ok=True)
    path, manifest_path = directory / "geometry_splits.npz", directory / "geometry_splits.json"
    if path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("requested_counts") != counts:
            raise ValueError("Geometry dataset configuration changed. Rebuild explicitly using --force, then retrain/evaluate.")
        if manifest["archive_sha256"] != file_sha256(path):
            raise ValueError("Geometry dataset checksum mismatch. Rebuild the dataset before continuing.")
        return manifest
    if (path.exists() or manifest_path.exists()) and not force:
        raise ValueError("Incomplete geometry dataset. Rebuild explicitly using --force.")
    blocked, splits, generation = set(), {}, {}
    for name in ("train", "validation", "test"):
        x, y, info = make_shape_dataset(int(counts[name]), SEEDS[name], excluded_hashes=blocked, return_audit=True)
        hashes = {shape_fingerprint(im) for im in x}
        if blocked & hashes:
            raise AssertionError("Generator allowed a cross-split duplicate.")
        blocked.update(hashes)
        splits[name], generation[name] = (x, y), info
    audit = audit_splits(splits)
    if not audit["passed"]:
        raise AssertionError("Deduplication audit failed.")
    np.savez_compressed(path, **{f"{axis}_{name}": values[i] for name, values in splits.items() for i, axis in enumerate(("x", "y"))})
    manifest = {**audit, "requested_counts": counts, "seeds": SEEDS, "generation": generation,
                "archive_sha256": file_sha256(path),
                "protocol": "Create ordered train/validation/test splits once, reject within-split and earlier-split duplicate normalized CNN inputs; never select weights using test performance."}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_shape_split(name, root=ROOT):
    if name not in DEFAULT_COUNTS:
        raise ValueError(name)
    path = Path(root) / "data" / "geometry_splits.npz"
    manifest = json.loads((path.parent / "geometry_splits.json").read_text(encoding="utf-8"))
    if manifest["archive_sha256"] != file_sha256(path) or not manifest["passed"]:
        raise ValueError("Geometry split checksum or audit failed.")
    with np.load(path, allow_pickle=False) as data:
        return data[f"x_{name}"], data[f"y_{name}"]


def geometry_fingerprints(root=ROOT):
    with np.load(Path(root) / "data" / "geometry_splits.npz", allow_pickle=False) as data:
        return {shape_fingerprint(im) for name in DEFAULT_COUNTS for im in data[f"x_{name}"]}
