"""Build and audit the three geometry splits before training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.datasets import DEFAULT_COUNTS, audit_splits, load_shape_split, prepare_shape_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Overwrite split files; retrain and reevaluate afterwards"
    )
    parser.add_argument("--train-count", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--validation-count", type=int, default=DEFAULT_COUNTS["validation"])
    parser.add_argument("--test-count", type=int, default=DEFAULT_COUNTS["test"])
    args = parser.parse_args()
    counts = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    if min(counts.values()) < 4:
        parser.error("Each split must contain at least four images.")
    manifest = prepare_shape_splits(ROOT, counts, force=args.force)
    verified = audit_splits({name: load_shape_split(name, ROOT) for name in DEFAULT_COUNTS})
    if not verified["passed"]:
        raise AssertionError("Dataset is not disjoint.")
    (ROOT / "results" / "geometry_deduplication.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
