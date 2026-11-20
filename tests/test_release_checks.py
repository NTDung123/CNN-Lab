"""Release validation must never regenerate a missing frozen dataset."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_release import check_release


class ReleaseCheckSafetyTests(unittest.TestCase):
    def test_missing_or_partial_dataset_is_rejected_without_creating_files(self):
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if partial:
                    (root / "data").mkdir()
                    (root / "data/geometry_splits.npz").write_bytes(b"existing partial archive")
                before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                with patch("scripts.check_release.ROOT", root), \
                     patch("scripts.check_release.prepare_shape_splits") as geometry, \
                     patch("scripts.check_release.prepare_real_training") as hybrid:
                    with self.assertRaisesRegex(RuntimeError, "Missing frozen dataset"):
                        check_release()
                geometry.assert_not_called()
                hybrid.assert_not_called()
                after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
