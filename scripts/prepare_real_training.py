"""Build phase-4 extra geometry CNN inputs offline, without opening new real test."""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.training_data import prepare_real_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare_real_training(ROOT, force=args.force)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "train_samples",
                    "train_class_counts",
                    "real_validation_samples",
                    "real_validation_class_counts",
                    "generation",
                )
            },
            indent=2,
        ),
        flush=True,
    )
