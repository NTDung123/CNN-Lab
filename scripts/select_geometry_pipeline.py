"""Select a region pipeline on validation only; never open new test photos."""

import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.service import Recognizer
from vision.real_data import read_real_manifest, real_image_path
from vision.metrics import detection_score, summarize_detection
from vision.provenance import asset_identity
from vision.datasets import file_sha256
from vision.real_data import PROTOCOL_PHASE, test_access_phase


def select(amend=None):
    """Chọn pipeline trên validation. amend: chạy lại sau khi test đã mở, kèm lý do (không dựa trên test)."""
    if test_access_phase(ROOT) >= PROTOCOL_PHASE and not amend:
        raise ValueError(
            "Test của giai đoạn này đã được mở. Giữ cấu hình đã khóa; thí nghiệm mới cần nhóm test nguồn mới."
        )
    r = Recognizer()
    manifest = read_real_manifest()
    cases = [c for c in manifest["cases"] if c["split"] == "validation"]
    modes = ("global_regions", "illumination_regions", "edge_regions", "contrast_regions", "repaired_regions")
    policies = (0.75, 0.85, 0.90)
    rows = {(mode, p): [] for mode in modes for p in policies}
    records = {key: [] for key in rows}
    for mode in modes:
        for case in cases:
            image = np.asarray(Image.open(real_image_path(case)).convert("RGB"))
            prepared = r.prepare_geometry(image, mode)
            for complex_threshold in policies:
                r.profile["complex_min_confidence"] = complex_threshold
                prediction = r.geometry(image, previews=False, mode=mode, prepared=prepared)
                score = detection_score(case["annotations"], prediction["objects"])
                rows[(mode, complex_threshold)].append(score)
                metrics = summarize_detection(
                    [score], 1, len(case["annotations"]), prediction["min_confidence"]
                )
                records[(mode, complex_threshold)].append(
                    {
                        "id": case["id"],
                        "split": "validation",
                        "source_group": case["source_group"],
                        "score": metrics,
                        "predictions": prediction["objects"],
                    }
                )
        print("Validated", mode, flush=True)
    choices = {}
    for key, value in rows.items():
        summary = summarize_detection(value, len(cases), sum(len(c["annotations"]) for c in cases))
        macro = sum(v["f1"] for v in summary["per_class"].values()) / 2
        name = key[0] + "_complex_" + str(key[1])
        choices[name] = {
            "mode": key[0],
            "complex_min_confidence": key[1],
            "summary": summary,
            "macro_target_f1": macro,
            "cases": records[key],
        }
    # Prefer simpler settings when target macro-F1 ties; mode order is fixed.
    best = max(
        choices,
        key=lambda name: (
            choices[name]["macro_target_f1"],
            choices[name]["summary"]["f1"],
            -modes.index(choices[name]["mode"]),
            -choices[name]["complex_min_confidence"],
        ),
    )
    selected = choices[best]
    profile_path = ROOT / "data/geometry_profile.json"
    profile = json.loads(profile_path.read_text())
    profile.update(
        {
            "version": PROTOCOL_PHASE,
            "mode": selected["mode"],
            "complex_min_confidence": selected["complex_min_confidence"],
            "classes_include_background": True,
            "selection_roles": ["validation"],
        }
    )
    profile_path.write_text(json.dumps(profile, indent=2))
    output = {
        "experiment_id": f"phase{PROTOCOL_PHASE}_geometry",
        "protocol_phase": PROTOCOL_PHASE,
        "amendment_reason": amend,
        "asset_identity": asset_identity(),
        "selected": best,
        "selection_roles": ["validation"],
        "new_test_predictions_seen": False,
        "by_mode": choices,
        "checkpoint_sha256": file_sha256(ROOT / "weights/shapes_cnn.pt"),
    }
    (ROOT / "data/real/selection_experiment.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": best,
                "validation": selected["summary"],
                "macro_target_f1": selected["macro_target_f1"],
            },
            indent=2,
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--amend", help="Lý do chạy lại sau khi test của giai đoạn này đã mở")
    select(parser.parse_args().amend)
