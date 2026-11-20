"""Cached training tensors must stay tied to their labels and frozen split."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.datasets import file_sha256
from vision.training_data import prepare_real_training
import vision.training_data as training_data


class TrainingCacheReleaseTests(unittest.TestCase):
    def cached_dataset(self, directory):
        root = Path(directory)
        files = {
            "manifest_sha256": "data/real/manifest.json",
            "geometry_sha256": "data/geometry_splits.npz",
            "processing_sha256": "vision/processing.py",
            "segmentation_sha256": "vision/segmentation.py",
        }
        for name in files.values():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unchanged fixture input")
        archive = root / "data/real_training.npz"
        archive.write_bytes(b"frozen cached tensors")
        metadata = {
            "archive_sha256": file_sha256(archive),
            "dependencies": {key: file_sha256(root / name) for key, name in files.items()},
        }
        metadata["dependencies"]["builder_sha256"] = file_sha256(training_data.__file__)
        (root / "data/real_training.json").write_text(json.dumps(metadata), encoding="utf-8")
        return root, metadata

    def test_unchanged_cached_training_inputs_are_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root, metadata = self.cached_dataset(directory)
            self.assertEqual(prepare_real_training(root), metadata)

    def test_changed_labels_or_geometry_cannot_reuse_cached_training_inputs(self):
        for name in ("data/real/manifest.json", "data/geometry_splits.npz"):
            with self.subTest(changed=name), tempfile.TemporaryDirectory() as directory:
                root, metadata = self.cached_dataset(directory)
                manifest_path = root / "data/real_training.json"
                previous_metadata = manifest_path.read_bytes()
                (root / name).write_bytes(b"changed roles, labels, or reserved split")
                with self.assertRaisesRegex(ValueError, "data sources changed.*--force"):
                    prepare_real_training(root)
                self.assertEqual(manifest_path.read_bytes(), previous_metadata)
                self.assertEqual(file_sha256(root / "data/real_training.npz"), metadata["archive_sha256"])

    def test_code_only_drift_keeps_frozen_archive_without_rewriting_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root, metadata = self.cached_dataset(directory)
            metadata["dependencies"]["builder_sha256"] = "historic builder revision"
            manifest_path = root / "data/real_training.json"
            manifest_path.write_text(json.dumps(metadata), encoding="utf-8")
            previous_metadata = manifest_path.read_bytes()
            (root / "vision/processing.py").write_bytes(b"updated phase5 preprocessing")
            result = prepare_real_training(root)
            self.assertEqual(result["archive_sha256"], metadata["archive_sha256"])
            self.assertEqual(result["code_changed_since_build"], ["builder_sha256", "processing_sha256"])
            self.assertEqual(manifest_path.read_bytes(), previous_metadata)
            self.assertEqual(file_sha256(root / "data/real_training.npz"), metadata["archive_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
