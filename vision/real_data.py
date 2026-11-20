"""Offline real-photo manifest and source-group / byte-integrity validation."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .datasets import ROOT, file_sha256


# Giai đoạn thí nghiệm hiện tại. Mỗi giai đoạn chọn/khóa cấu hình một lần trước khi mở test.
PROTOCOL_PHASE = 5


def test_access_phase(root=ROOT):
    """Giai đoạn đã mở test gần nhất (0 nếu chưa mở). Nhật ký cũ không ghi giai đoạn là giai đoạn 4."""
    path = Path(root) / "data" / "real" / "test_access_log.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("phase", 4))


def read_real_manifest(root=ROOT):
    directory = Path(root) / "data" / "real"
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def real_image_path(case, root=ROOT):
    directory = (Path(root) / "data" / "real").resolve()
    path = (directory / case["image_file"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Image path escapes the real dataset directory.")
    return path


def audit_real_manifest(root=ROOT):
    root = Path(root)
    manifest = read_real_manifest(root)
    sources = {s["id"]: s for s in manifest["sources"]}
    if len(sources) != len(manifest["sources"]):
        raise ValueError("Duplicate real source ID.")
    seen_ids, seen_pixels, groups, source_groups = set(), set(), {}, {}
    counts, class_counts = Counter(), Counter()
    role_counts, role_targets = Counter(), Counter()
    role_classes, negatives = {}, Counter()
    characteristics = []
    from .processing import grayscale

    for case in manifest["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]):
            raise ValueError("Unsafe real case ID.")
        if case["id"] in seen_ids:
            raise ValueError("Duplicate case ID.")
        seen_ids.add(case["id"])
        source = sources[case["source_id"]]
        group, split = case["source_group"], case["split"]
        if split not in ("development", "validation", "test"):
            raise ValueError("Unknown real dataset role.")
        if group in groups and groups[group] != split:
            raise ValueError("Source photograph leaked across real dataset roles.")
        if case["source_id"] in source_groups and source_groups[case["source_id"]] != group:
            raise ValueError("One source photograph was assigned multiple source groups.")
        groups[group], source_groups[case["source_id"]] = split, group
        path = real_image_path(case, root)
        if file_sha256(path) != source["image_sha256"]:
            raise ValueError(f"Photo checksum changed: {case['id']}")
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        import hashlib

        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        if pixel_hash != source["pixel_sha256"] or pixel_hash in seen_pixels:
            raise ValueError("Decoded image mismatch or duplicate real image.")
        seen_pixels.add(pixel_hash)
        h, w = rgb.shape[:2]
        if [w, h] != [case["width"], case["height"]]:
            raise ValueError("Photo dimensions disagree with annotations.")
        if max(w, h) > 800:
            raise ValueError(
                "Distributed real photos must be at most 800 pixels per side, matching GT coordinates."
            )
        if not source.get("license") or not source.get("source_page"):
            raise ValueError("Missing image license/source attribution.")
        if len({a["id"] for a in case["annotations"]}) != len(case["annotations"]):
            raise ValueError("Duplicate annotation ID.")
        for a in case["annotations"]:
            x, y, bw, bh = a["bbox"]
            if a["class"] not in ("circle", "rectangle") or not (
                0 <= x < x + bw <= w and 0 <= y < y + bh <= h
            ):
                raise ValueError("Invalid real-image ground truth.")
            class_counts[a["class"]] += 1
            role_classes.setdefault(split, Counter())[a["class"]] += 1
        for a in case.get("distractors", []):
            x, y, bw, bh = a["bbox"]
            if a["class"] not in ("ellipse", "triangle", "background") or not (
                0 <= x < x + bw <= w and 0 <= y < y + bh <= h
            ):
                raise ValueError("Invalid distractor annotation.")
        counts[case["difficulty"]] += 1
        role_counts[split] += 1
        role_targets[split] += len(case["annotations"])
        negatives[split] += int(not case["annotations"])
        gray = grayscale(rgb)
        characteristics.append(
            {
                "id": case["id"],
                "size": [w, h],
                "targets": len(case["annotations"]),
                "gray_mean": float(gray.mean()),
                "gray_std": float(gray.std()),
                "gray_p05": float(np.percentile(gray, 5)),
                "gray_p95": float(np.percentile(gray, 95)),
                "border_std": float(np.std(np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1])))),
            }
        )
    if manifest.get("schema_version", 0) >= 4:
        if any(role_classes.get("validation", {}).get(cls, 0) == 0 for cls in ("circle", "rectangle")):
            raise ValueError("Validation must contain both target classes.")
        previous_path = root / "results/history/v3/real_manifest.json"
        if previous_path.exists():
            old_groups = {c["source_group"] for c in json.loads(previous_path.read_text())["cases"]}
            if any(c["split"] == "test" and c["source_group"] in old_groups for c in manifest["cases"]):
                raise ValueError("A previously exposed source group was reused as new test.")
    return {
        "passed": True,
        "case_count": len(seen_ids),
        "original_photo_groups": len(groups),
        "duplicate_decoded_images": 0,
        "source_groups_crossing_splits": 0,
        "case_counts_by_difficulty": dict(counts),
        "target_counts": dict(class_counts),
        "case_counts_by_role": dict(role_counts),
        "target_counts_by_role": dict(role_targets),
        "class_counts_by_role": {role: dict(v) for role, v in role_classes.items()},
        "negative_frames_by_role": dict(negatives),
        "image_characteristics": characteristics,
        "annotation_policy": manifest["annotation_policy"],
        "statistical_scope": manifest["protocol"]["statistical_scope"],
    }


def check_real_protocol(root=ROOT):
    root = Path(root)
    lock_path = root / "data" / "real" / "protocol_lock.json"
    if not lock_path.exists():
        return {"matched_original_protocol": False, "changed_items": ["protocol_lock_missing"], "lock": {}}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pairs = {
        "manifest_sha256": root / "data" / "real" / "manifest.json",
        "shapes_checkpoint_sha256": root / "weights" / "shapes_cnn.pt",
        "geometry_dataset_sha256": root / "data" / "geometry_splits.npz",
        "processing_sha256": root / "vision" / "processing.py",
        "service_sha256": root / "vision" / "service.py",
        "segmentation_sha256": root / "vision" / "segmentation.py",
        "geometry_profile_sha256": root / "data" / "geometry_profile.json",
        "metrics_sha256": root / "vision" / "metrics.py",
        "real_evaluation_sha256": root / "vision" / "real_evaluation.py",
        "selection_experiment_sha256": root / "data" / "real" / "selection_experiment.json",
    }
    mismatches = [
        key for key, path in pairs.items() if not path.exists() or file_sha256(path) != lock.get(key)
    ]
    selection_path = root / "data/real/selection_experiment.json"
    if selection_path.exists():
        from .provenance import asset_identity

        selected_assets = json.loads(selection_path.read_text(encoding="utf-8")).get("asset_identity", {})
        mismatches.extend(
            "selected_asset:" + key
            for key, value in asset_identity(root).items()
            if selected_assets.get(key) != value
        )
    return {"matched_original_protocol": not mismatches, "changed_items": mismatches, "lock": lock}
