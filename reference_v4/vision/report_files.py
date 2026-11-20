"""Report persistence and metadata; this module never changes inference.

Development/exploratory commands have their own files. A complete frozen test
report is only published by the explicit include-test path.
"""
import json
import os
import tempfile
from pathlib import Path


def write_json_atomic(path, value):
    """Serialize before touching the destination; replace it in one operation."""
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name+".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def real_report_name(include_test=False, exploratory=False):
    if exploratory:
        return "real_exploratory_evaluation.json"
    return "real_evaluation.json" if include_test else "real_development_evaluation.json"


def comparison_report_name(include_test=False):
    return "segmentation_comparison.json" if include_test else "segmentation_comparison_development.json"


def evaluation_directory(root, exploratory=False, partial=False):
    directory = Path(root) / "results"
    return directory / "exploratory" if exploratory else directory / "partial" if partial else directory


def annotate_confidence_policy(result, profile):
    """Record the threshold actually used per case, including complex scenes."""
    base = float(profile["min_confidence"])
    complex_score = max(base, float(profile.get("complex_min_confidence", base)))
    border_cut = float(profile.get("illumination_border_std", 12))
    policy = {"base_threshold": base, "complex_threshold": complex_score,
              "complex_condition": "background_border_std >= border_std_threshold",
              "border_std_threshold": border_cut}
    for case in result.get("cases", []):
        spread = case.get("segmentation", {}).get("background_border_std")
        threshold = complex_score if spread is not None and spread >= border_cut else base
        case["detection"]["confidence_threshold"] = threshold
    result.setdefault("protocol", {})["confidence_policy"] = policy
    for summary in [result.get("detection_all", {})] + list(result.get("detection_by_role", {}).values()) + list(result.get("detection_by_difficulty", {}).values()):
        if summary:
            summary["confidence_policy"] = policy
            summary["confidence_threshold"] = None if complex_score != base else base
    return result
