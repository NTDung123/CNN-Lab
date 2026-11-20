import base64
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from vision.processing import (closing, connected_components, fill_holes, grayscale,
                               otsu_threshold, preprocess_digit, resize_area, resize_bilinear)
from vision.service import png_data
from vision.synthetic import make_scene
from vision.synthetic import make_shape_dataset
from vision.datasets import DEFAULT_COUNTS, audit_splits, load_shape_split, shape_fingerprint
from vision.metrics import detection_score
from vision.real_data import audit_real_manifest, check_real_protocol, read_real_manifest


class AlgorithmTests(unittest.TestCase):
    def test_grayscale_and_alpha(self):
        a = np.array([[[255, 0, 0, 255], [0, 0, 0, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(grayscale(a), [[76, 255]])

    def test_otsu_separates_two_classes(self):
        image = np.array([[10, 10, 10], [200, 200, 200]], np.uint8)
        threshold = otsu_threshold(image)
        self.assertGreaterEqual(threshold, 10)
        self.assertLess(threshold, 200)
        np.testing.assert_array_equal(image > threshold, [[False] * 3, [True] * 3])

    def test_eight_connectivity_and_area_filter(self):
        mask = np.zeros((7, 8), bool)
        mask[0, 0] = mask[1, 1] = True
        mask[3:6, 5:8] = True
        mask[6, 0] = True
        parts = connected_components(mask, min_area=2)
        self.assertEqual([(p.area, p.bbox) for p in parts], [(2, [0, 0, 2, 2]), (9, [5, 3, 3, 3])])

    def test_union_joining_previously_separate_runs(self):
        mask = np.array([[1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], bool)
        self.assertEqual(len(connected_components(mask)), 1)
        self.assertEqual(connected_components(mask)[0].area, 7)

    def test_holes_and_morphological_closing(self):
        mask = np.zeros((9, 9), bool)
        mask[2:7, 2:7] = True
        mask[3:6, 3:6] = False
        self.assertEqual(int(fill_holes(mask).sum()), 25)
        single_hole = np.zeros((9, 9), bool)
        single_hole[2:7, 2:7] = True
        single_hole[4, 4] = False
        self.assertEqual(int(closing(single_hole).sum()), 25)

    def test_resize_preserves_identity_and_average(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.testing.assert_allclose(resize_bilinear(a, a.shape), a)
        np.testing.assert_allclose(resize_area(a, (1, 1)), [[7.5]])

    def test_digit_polarity_and_blank(self):
        with np.load(ROOT / "data" / "mnist.npz") as d:
            digit = d["x_test"][0]
        light, _ = preprocess_digit(digit)
        dark, _ = preprocess_digit(255 - digit)
        np.testing.assert_allclose(light, dark, atol=1e-6)
        for value in (0, 128, 255):
            with self.assertRaises(ValueError):
                preprocess_digit(np.full((60, 60), value, np.uint8))


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_home_assets_and_models(self):
        for route in ("/", "/static/script.js", "/static/style.css", "/api/health"):
            with self.subTest(route=route):
                with self.client.get(route) as response:
                    self.assertEqual(response.status_code, 200)
        health = self.client.get("/api/health").get_json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["digits"]["training_samples"], 55000)

    def test_ten_unselected_mnist_examples_via_http_pipeline(self):
        # The first test image of each class; same documented examples as the UI.
        for label in range(10):
            raw = (ROOT / "examples" / f"digit_{label}.png").read_bytes()
            data = "data:image/png;base64," + base64.b64encode(raw).decode()
            response = self.client.post("/api/digits", json={"image": data})
            with self.subTest(label=label):
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                self.assertEqual(result["prediction"], label)
                self.assertAlmostEqual(sum(result["probabilities"]), 1, places=5)
                self.assertEqual(len(result["feature_maps"]), 8)

    def test_mixed_rotated_and_outline_shapes(self):
        image, truth = make_scene(1)
        response = self.client.post("/api/shapes", json={"image": png_data(image)})
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        for name in ("circle", "rectangle"):
            self.assertEqual(result["counts"][name], sum(g["class"] == name for g in truth))
        self.assertEqual(len(result["objects"]), 6)
        self.assertEqual(len(result["pipeline"]), 5)

    def test_bad_input_is_not_server_error(self):
        for route in ("/api/digits", "/api/shapes"):
            for payload in ({}, [], {"image": "not base64"}, {"image": 42}, {"image": "aGVsbG8="}):
                with self.subTest(route=route, payload=payload):
                    response = self.client.post(route, json=payload)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", response.get_json())
        response = self.client.post("/api/digits", json={"image": png_data(np.full((100, 100), 255, np.uint8))})
        self.assertEqual(response.status_code, 400)

    def test_empty_shape_image_and_negative_classes(self):
        response = self.client.post("/api/shapes", json={"image": png_data(np.full((80, 100), 255, np.uint8))})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["counts"], {"circle": 0, "rectangle": 0})

    def test_evaluation_protocol_is_independent_and_has_stress_failures(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["digits"]["cnn_raw"]["samples"], 10000)
        self.assertEqual(result["shapes"]["detection"]["scenes"], 90)
        self.assertGreater(result["digits"]["raw_failures"]["total_failures"], 0)
        self.assertLess(result["shapes"]["stress_touching"]["recall"], 1)
        self.assertTrue(result["shapes"]["deduplication"]["passed"])
        self.assertEqual(result["real_images"]["audit"]["case_count"], 40)

    def test_real_photo_gallery_api_and_assets(self):
        response = self.client.get("/api/real")
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(len(result["cases"]), 40)
        self.assertEqual(result["detection_all"]["ground_truth_targets"], 68)
        self.assertTrue(result["protocol"]["original_lock_verified"])
        self.assertTrue(result["protocol"]["real_images_used_for_training"])
        for route in ("/real/images/floppy_front.png", "/real/annotations/brooklyn_coins.png", "/static/real.css"):
            with self.client.get(route) as asset:
                self.assertEqual(asset.status_code, 200)


class GeometryDataTests(unittest.TestCase):
    def test_quantized_fingerprint_catches_float_only_differences(self):
        a = np.zeros((32, 32), np.float32)
        a[10:20, 10:20] = .5001
        b = a.astype(np.float64)
        b[10:20, 10:20] += .0001
        self.assertNotEqual(a.tobytes(), b.astype(np.float32).tobytes())
        self.assertEqual(shape_fingerprint(a), shape_fingerprint(b))

    def test_all_frozen_splits_are_unique_and_disjoint(self):
        splits = {name: load_shape_split(name, ROOT) for name in DEFAULT_COUNTS}
        result = audit_splits(splits)
        self.assertTrue(result["passed"])
        self.assertEqual(list(result["cross_split_overlap"].values()), [0, 0, 0])
        for name, expected in DEFAULT_COUNTS.items():
            self.assertEqual(result["splits"][name]["samples"], expected)
            self.assertEqual(result["splits"][name]["unique_normalized_uint8"], expected)

    def test_generator_actively_rejects_blocked_inputs(self):
        first, _ = make_shape_dataset(40, 998)
        blocked = {shape_fingerprint(im) for im in first}
        second, labels, audit = make_shape_dataset(40, 998, excluded_hashes=blocked, return_audit=True)
        self.assertGreater(audit["rejected_against_previous_splits"], 0)
        self.assertFalse(blocked & {shape_fingerprint(im) for im in second})
        self.assertEqual(np.bincount(labels).tolist(), [10, 10, 10, 10])

    def test_duplicate_and_wrong_class_detections_are_penalized(self):
        truth = [{"class": "circle", "bbox": [10, 10, 20, 20]}]
        predictions = [{"class": "circle", "bbox": [10, 10, 20, 20], "confidence": .99, "target": True},
                       {"class": "circle", "bbox": [10, 10, 20, 20], "confidence": .98, "target": True},
                       {"class": "rectangle", "bbox": [10, 10, 20, 20], "confidence": .95, "target": True}]
        counts, overlaps = detection_score(truth, predictions)
        self.assertEqual(counts["circle"], {"tp": 1, "fp": 1, "fn": 0})
        self.assertEqual(counts["rectangle"]["fp"], 1)
        self.assertEqual(overlaps, [1.])

    def test_real_sources_and_labels_are_valid(self):
        result = audit_real_manifest(ROOT)
        self.assertTrue(result["passed"])
        self.assertEqual(result["original_photo_groups"], 33)
        self.assertEqual(result["duplicate_decoded_images"], 0)
        self.assertEqual(result["source_groups_crossing_splits"], 0)
        self.assertEqual(sum(result["target_counts"].values()), 68)

    def test_source_group_cannot_cross_roles(self):
        manifest = read_real_manifest(ROOT)
        manifest["cases"][1]["source_group"] = manifest["cases"][0]["source_group"]
        manifest["cases"][1]["split"] = "test"
        with patch("vision.real_data.read_real_manifest", return_value=manifest):
            with self.assertRaisesRegex(ValueError, "leaked"):
                audit_real_manifest(ROOT)

    def test_protocol_detects_changed_checkpoint(self):
        from vision.real_data import file_sha256
        self.assertTrue(check_real_protocol(ROOT)["matched_original_protocol"])
        def changed(path):
            return "changed" if Path(path).name == "shapes_cnn.pt" else file_sha256(path)
        with patch("vision.real_data.file_sha256", side_effect=changed):
            result = check_real_protocol(ROOT)
        self.assertFalse(result["matched_original_protocol"])
        self.assertIn("shapes_checkpoint_sha256", result["changed_items"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
