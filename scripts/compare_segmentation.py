"""Paired phase-3/phase-4 comparison on identical phase-4 photos and labels.

Three systems on identical photos and labels: phase 3, phase 4 (before the phase-5 bug
fixes, same CNN weights as phase 5) and the current phase 5. Phase 4 vs 5 isolates the
effect of the phase-5 image-processing fixes, because the CNN checkpoint is unchanged.
New real test is opened only after a verified protocol freeze.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.service import Recognizer
from vision.provenance import asset_identity
from vision.real_data import read_real_manifest, real_image_path, check_real_protocol
from vision.real_evaluation import evaluate_real_cases
from vision.metrics import detection_score, summarize_detection
from vision.datasets import file_sha256
from reference_v3.recognizer import legacy_recognizer
from reference_v4.recognizer import phase4_recognizer
from scripts.evaluate_real import evaluate_real_report
from vision.report_files import comparison_report_name, write_json_atomic, annotate_confidence_policy


def compare(include_test=False):
    r = Recognizer()
    if include_test:
        if not check_real_protocol()["matched_original_protocol"]:
            raise ValueError("Freeze the phase-4 protocol before opening new test.")
        # Register the actual first access before any model sees a new test.
        # This also writes the authoritative current-system comparisons.
        evaluate_real_report(r, root=ROOT, include_test=True)
    systems = {"legacy_v3": legacy_recognizer(), "phase4": phase4_recognizer(), "phase5": r}
    manifest = read_real_manifest()
    cases = [c for c in manifest["cases"] if include_test or c["split"] != "test"]
    summary, records, policies = {}, {}, {}
    for name, recognizer in systems.items():
        by_role = {}
        items = []
        for c in cases:
            image = np.asarray(Image.open(real_image_path(c)).convert("RGB"))
            p = recognizer.geometry(image, previews=False)
            score = detection_score(c["annotations"], p["objects"])
            by_role.setdefault(c["split"], []).append(score)
            items.append(
                {
                    "id": c["id"],
                    "split": c["split"],
                    "predictions": p["objects"],
                    "segmentation": p.get("segmentation", {}),
                    "detection": summarize_detection(
                        [score],
                        1,
                        len(c["annotations"]),
                        p.get("min_confidence", recognizer.profile["min_confidence"]),
                    ),
                }
            )
        summary[name] = {
            role: summarize_detection(
                rows, len(rows), sum(len(c["annotations"]) for c in cases if c["split"] == role)
            )
            for role, rows in by_role.items()
        }
        annotated = annotate_confidence_policy(
            {"protocol": {}, "cases": items, "detection_by_role": summary[name]}, recognizer.profile
        )
        policies[name] = annotated["protocol"]["confidence_policy"]
        records[name] = items
        print(name, json.dumps(summary[name]), flush=True)
    output = {
        "asset_identity": asset_identity(),
        "test_included": include_test,
        "same_photos_and_labels": True,
        "by_mode_and_role": summary,
        "cases_by_mode": records,
        "confidence_policy_by_system": policies,
        "checkpoint_sha256": {
            "legacy_v3": file_sha256(ROOT / "reference_v3/weights/shapes_cnn.pt"),
            "phase4": file_sha256(ROOT / "reference_v4/weights/shapes_cnn.pt"),
            "phase5": file_sha256(ROOT / "weights/shapes_cnn.pt"),
        },
        "note": "Phase-3 and phase-4 code unchanged in reference_v3/ and reference_v4/. Phase 4 and 5 share the CNN weights, so their difference is the phase-5 image-processing fixes. The real test photos were first opened in phase 4; phase-5 test scores are not an unseen benchmark. Label policy is projected 2D; historical scores remain in results/history/.",
    }
    output["report_file"] = comparison_report_name(include_test)
    write_json_atomic(ROOT / "results" / output["report_file"], output)
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--include-test", action="store_true")
    a = p.parse_args()
    compare(a.include_test)
