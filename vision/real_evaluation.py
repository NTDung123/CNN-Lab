"""Full-frame real-photo evaluation with explicit development/validation/test roles.

New test images are inaccessible by default. Freeze the pipeline first, then use
include_test=True. Old phase-2 photos are development data forever in phase 3.
"""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .datasets import ROOT, file_sha256
from .metrics import detection_score, iou, match_detections, summarize_detection
from .models import SHAPE_CLASSES, predict
from .processing import closing, connected_components, foreground, grayscale, normalize_shape, resize_bilinear
from .provenance import asset_identity
from .real_data import audit_real_manifest, check_real_protocol, read_real_manifest, real_image_path


def _roi_input(rgb, bbox):
    # This is explicitly an ORACLE-location diagnostic, not detection.
    x, y, w, h = bbox
    pad = max(3, round(max(w, h) * 0.08))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    crop = rgb[y0 : min(rgb.shape[0], y + h + pad), x0 : min(rgb.shape[1], x + w + pad)]
    mask, _, _, _ = foreground(grayscale(crop), denoise=True)
    parts = connected_components(closing(mask), min_area=3)
    if not parts:
        return None
    expected = [x - x0, y - y0, w, h]
    # A neighboring object in the padding must not win merely by pixel area.
    part = max(parts, key=lambda c: (iou(c.bbox, expected), c.area))
    return normalize_shape(part.mask)


def diagnose_matches(truth, objects, candidates=None):
    matched = {m["truth_index"]: m for m in match_detections(truth, objects)}
    output = []
    for i, a in enumerate(truth):
        overlap, _ = max(
            ((iou(a["bbox"], o["bbox"]), o) for o in objects), key=lambda v: v[0], default=(0.0, None)
        )
        if i in matched:
            reason = "matched"
        elif any(
            o["target"] and o["class"] == a["class"] and iou(a["bbox"], o["bbox"]) >= 0.5 for o in objects
        ):
            reason = "matching_region_already_assigned"
        elif overlap >= 0.5:
            reason = "region_found_but_wrong_class_or_low_score"
        elif candidates and any(iou(a["bbox"], c["bbox"]) >= 0.5 for c in candidates):
            reason = "candidate_removed_by_filter_or_background_cnn"
        else:
            reason = "no_matching_region_after_segmentation"
        output.append(
            {
                "best_region_iou": overlap,
                "failure_category": reason,
                "matched_prediction_index": matched[i]["prediction_index"] if i in matched else None,
            }
        )
    return output


def _comparison_image(rgb, truth, predicted_data, destination):
    annotated = Image.open(
        __import__("io").BytesIO(base64.b64decode(predicted_data.split(",", 1)[1]))
    ).convert("RGB")
    expected = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(expected)
    for index, a in enumerate(truth, start=1):
        x, y, w, h = a["bbox"]
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline="#c34f36", width=2)
        draw.text((x + 2, y + 2), f"{index}: {a['class']}", fill="#c34f36")
    thumb_w = min(500, expected.width)
    thumb_h = round(expected.height * thumb_w / expected.width)
    left = resize_bilinear(np.asarray(expected), (thumb_h, thumb_w)).astype(np.uint8)
    right = resize_bilinear(np.asarray(annotated), (thumb_h, thumb_w)).astype(np.uint8)
    panel = Image.new("RGB", (thumb_w * 2 + 12, thumb_h + 30), "#f5f7f2")
    panel.paste(Image.fromarray(left), (0, 30))
    panel.paste(Image.fromarray(right), (thumb_w + 12, 30))
    d = ImageDraw.Draw(panel)
    d.text((8, 8), "MANUAL GROUND TRUTH", fill="#9b3e2c")
    d.text((thumb_w + 20, 8), "MODEL PREDICTIONS", fill="#17694f")
    panel.save(destination)


def evaluate_real_cases(recognizer, root=ROOT, exploratory=False, write=True, include_test=False, mode=None):
    root = Path(root)
    directory = root / "results"
    audit = audit_real_manifest(root)
    manifest = read_real_manifest(root)
    protocol = check_real_protocol(root)
    if include_test and not protocol["matched_original_protocol"] and not exploratory:
        raise ValueError(
            "Pipeline chưa khớp protocol_lock.json. Khóa cấu hình sau development/validation trước khi mở test; không chỉnh pipeline theo test."
        )
    if include_test:
        from .real_data import PROTOCOL_PHASE, test_access_phase

        access_path = root / "data/real/test_access_log.json"
        if test_access_phase(root) < PROTOCOL_PHASE:
            access_path.write_text(
                json.dumps(
                    {
                        "first_access_utc": datetime.now(timezone.utc).isoformat(),
                        "manifest_sha256": file_sha256(root / "data/real/manifest.json"),
                        "protocol_lock_sha256": file_sha256(root / "data/real/protocol_lock.json"),
                        "geometry_profile_sha256": file_sha256(root / "data/geometry_profile.json"),
                        "first_mode": mode or recognizer.profile["mode"],
                        "phase": PROTOCOL_PHASE,
                        "note": "Pipeline already frozen. After this point these test photos are seen; no tuning on their results.",
                    },
                    indent=2,
                )
            )
    selected = [c for c in manifest["cases"] if include_test or c["split"] != "test"]
    sources = {s["id"]: s for s in manifest["sources"]}
    records, scores, by_difficulty, by_role = [], [], {}, {}
    roi_correct = roi_total = 0
    for case in selected:
        with Image.open(real_image_path(case, root)) as im:
            rgb = np.asarray(im.convert("RGB"))
        prediction = recognizer.geometry(rgb, previews=write, mode=mode)
        truth = case["annotations"]
        score = detection_score(truth, prediction["objects"])
        scores.append(score)
        by_difficulty.setdefault(case["difficulty"], []).append(score)
        by_role.setdefault(case["split"], []).append(score)
        diagnostics = diagnose_matches(truth, prediction["objects"], prediction.get("candidate_diagnostics"))
        for a, diagnostic in zip(truth, diagnostics):
            normalized = _roi_input(rgb, a["bbox"])
            probs = predict(recognizer.shapes, normalized)[0] if normalized is not None else None
            roi_class = recognizer.shape_classes[int(probs.argmax())] if probs is not None else "no_region"
            roi_correct += int(roi_class == a["class"])
            roi_total += 1
            diagnostic.update(
                {
                    "annotation_id": a["id"],
                    "true_class": a["class"],
                    "bbox": a["bbox"],
                    "best_candidate_iou_before_filter": max(
                        (iou(a["bbox"], c["bbox"]) for c in prediction.get("candidate_diagnostics", [])),
                        default=0.0,
                    ),
                    "roi_top1_class": roi_class,
                    "roi_top1_correct": roi_class == a["class"],
                    "roi_top1_score": float(probs.max()) if probs is not None else None,
                }
            )
        per_case = summarize_detection([score], 1, len(truth), recognizer.profile.get("min_confidence", 0.75))
        name = f"real_{case['id']}_comparison.png"
        if write:
            directory.mkdir(exist_ok=True)
            _comparison_image(rgb, truth, prediction["annotated"], directory / name)
        records.append(
            {
                "id": case["id"],
                "title": case["title"],
                "source_group": case["source_group"],
                "split": case["split"],
                "difficulty": case["difficulty"],
                "notes": case["notes"],
                "source": sources[case["source_id"]],
                "image_url": "/real/" + case["image_file"],
                "comparison_image": name,
                "ground_truth": truth,
                "predictions": prediction["objects"],
                "warnings": prediction["warnings"],
                "segmentation": prediction["segmentation"],
                "distractors": case.get("distractors", []),
                "capture_type": case.get("capture_type", "photograph"),
                "candidate_diagnostics": prediction.get("candidate_diagnostics", []),
                "detection": per_case,
                "roi_diagnostics": diagnostics,
            }
        )
    summaries = lambda groups, field: {
        key: summarize_detection(
            rows,
            len(rows),
            sum(len(c["annotations"]) for c in selected if c[field] == key),
            recognizer.profile.get("min_confidence", 0.75),
        )
        for key, rows in groups.items()
    }
    failures = {}
    for record in records:
        for d in record["roi_diagnostics"]:
            failures[d["failure_category"]] = failures.get(d["failure_category"], 0) + 1
    result = {
        "schema_version": 4,
        "dataset": manifest["dataset_name"],
        "asset_identity": asset_identity(root),
        "audit": audit,
        "protocol": {
            "original_lock_verified": protocol["matched_original_protocol"],
            "protocol_phase": protocol["lock"].get("protocol_version"),
            "test_previously_seen_in_phase": protocol["lock"].get("test_previously_seen_in_phase"),
            "test_independent": protocol["lock"].get("test_independent", True),
            "exploratory": exploratory,
            "changed_items": protocol["changed_items"],
            "test_included": include_test,
            "real_images_used_for_training": True,
            "real_training_roles": ["development"],
            "cnn_weights_changed_in_phase4": True,
            "preprocessing_selection_roles": ["validation"],
            "new_test_used_for_pipeline_selection": (
                False if protocol["matched_original_protocol"] and not exploratory else "not_claimed"
            ),
            "checkpoint_sha256": file_sha256(root / "weights/shapes_cnn.pt"),
            "manifest_sha256": file_sha256(root / "data/real/manifest.json"),
            "mode": mode or recognizer.profile["mode"],
        },
        "detection_all": summarize_detection(
            scores, len(records), roi_total, recognizer.profile.get("min_confidence", 0.75)
        ),
        "detection_by_role": summaries(by_role, "split"),
        "detection_by_difficulty": summaries(by_difficulty, "difficulty"),
        "roi_diagnostic": {
            "samples": roi_total,
            "correct": roi_correct,
            "top1_accuracy": roi_correct / max(roi_total, 1),
            "uses_ground_truth_bbox": True,
            "warning": "Oracle-location diagnostic; NOT full-frame detection.",
        },
        "failure_counts": failures,
        "cases": records,
        "limits": [
            "Convenience sample; scores are not population-level real-world accuracy.",
            "Previously exposed phase-3 test photos are development; only new source groups are phase-4 test.",
            "Manual single-annotator boxes; occluded boxes estimated, clipped boxes cover visible extent.",
            "Perspective of a circular physical object may look elliptical and is categorized as stress.",
            "No automatic split of touching objects. All failures are retained.",
        ],
    }
    result["negative_frames_by_role"] = {}
    for role in ("development", "validation", "test"):
        negative = [c for c in records if c["split"] == role and not c["ground_truth"]]
        if negative:
            result["negative_frames_by_role"][role] = {
                "frames": len(negative),
                "frames_with_false_positives": sum(c["detection"]["fp"] > 0 for c in negative),
                "false_positive_detections": sum(c["detection"]["fp"] for c in negative),
            }
    if write:
        comparison = directory / "segmentation_comparison.json"
        if comparison.exists():
            baseline = json.loads(comparison.read_text(encoding="utf-8"))
            if baseline.get("asset_identity") == result["asset_identity"]:
                result["baseline_comparison"] = baseline["by_mode_and_role"]
        (directory / "real_evaluation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (directory / "real_manifest_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return result
