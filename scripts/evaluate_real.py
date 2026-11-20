"""Evaluate the frozen real-photo pilot without fitting or changing weights."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.real_evaluation import evaluate_real_cases
from vision.service import Recognizer
from vision.report_files import real_report_name, write_json_atomic, annotate_confidence_policy


def evaluate_real_report(recognizer, root=ROOT, exploratory=False, include_test=False):
    # Only explicit complete frozen evaluation publishes the demo's main file.
    publish = include_test and not exploratory
    result = evaluate_real_cases(
        recognizer, root, exploratory=exploratory, include_test=include_test, write=publish
    )
    annotate_confidence_policy(result, recognizer.profile)
    name = real_report_name(include_test, exploratory)
    result["report_file"] = name
    result["report_scope"] = (
        "frozen_complete_test" if publish else "exploratory" if exploratory else "development_validation"
    )
    if not publish:
        # No new images were rendered; do not link a changed model's numbers
        # to stale PNGs from the authoritative frozen report.
        for case in result["cases"]:
            case["comparison_image"] = None
    write_json_atomic(Path(root) / "results" / name, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exploratory-real",
        action="store_true",
        help="Allow a changed checkpoint/pipeline; mark results as exploratory",
    )
    parser.add_argument(
        "--include-test", action="store_true", help="Evaluate new test only after freezing protocol_lock.json"
    )
    args = parser.parse_args()
    result = evaluate_real_report(
        Recognizer(), ROOT, exploratory=args.exploratory_real, include_test=args.include_test
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "audit",
                    "protocol",
                    "detection_all",
                    "detection_by_difficulty",
                    "roi_diagnostic",
                    "failure_counts",
                ]
            },
            indent=2,
            ensure_ascii=False,
        )
    )
