"""Hồi quy cho các lỗi đã sửa ở giai đoạn 5.

Mỗi ca dựng ảnh bằng NumPy, tái hiện đúng tình huống từng làm chương trình sai trước khi sửa.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.processing import (
    chain_code,
    chain_perimeter,
    fill_holes,
    geometric_features,  # noqa: E402
    preprocess_digit,
    remove_ruled_lines,
    resize_bilinear,
    trace_boundary,
)
from vision.service import Recognizer  # noqa: E402
from vision.synthetic import raster_shape  # noqa: E402

H, W = 400, 600


def mnist_validation_digits(count):
    """Chữ số từ tập validation MNIST (cùng hoán vị với scripts/train.py), không dùng tập test."""
    with np.load(ROOT / "data" / "mnist.npz", allow_pickle=False) as data:
        order = np.random.default_rng(42).permutation(60000)[:count]
        return data["x_train"][order], data["y_train"][order]


def paper(digit, k=8, light=0.0, lines=False):
    """Đặt chữ số MNIST (mực đen) lên tờ giấy trắng 300 x 300, có thể thêm bóng và dòng kẻ."""
    big = resize_bilinear(digit / 255.0, (28 * k, 28 * k))
    a = np.zeros((300, 300))
    o = (300 - 28 * k) // 2
    a[o : o + 28 * k, o : o + 28 * k] = big
    img = 255.0 * (1 - a)
    if lines:
        for y in range(15, 300, 30):
            img[y : y + 2] = np.minimum(img[y : y + 2], 170)
    if light:
        img = img * (1 - light * np.linspace(0, 1, 300))[None, :]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def scene(parts, light=0.0):
    img = np.full((H, W, 3), 245.0)
    for mask in parts:
        img[mask] = 20
    if light:
        img *= (1 - light * np.linspace(0, 1, W))[None, :, None]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


class DigitFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = Recognizer()
        cls.x, cls.y = mnist_validation_digits(40)

    def accuracy(self, **kwargs):
        hits = [
            self.recognizer.digit(paper(d, **kwargs), previews=False)["prediction"] == t
            for d, t in zip(self.x, self.y)
        ]
        return float(np.mean(hits))

    def test_digit_result_image_is_colored_by_confidence(self):
        import base64
        import io

        from PIL import Image

        def rgb_of(result):
            png = base64.b64decode(result["annotated"].split(",", 1)[1])
            return np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)

        noisy = paper(self.x[0]).copy()
        noisy[20:34, 250:266] = 40  # vết bẩn nhỏ hơn 18% nét chữ -> bị bỏ qua
        result = self.recognizer.digit(noisy)
        self.assertTrue(result["confident"])
        self.assertGreaterEqual(result["ignored_parts"], 1)
        rgb = rgb_of(result)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        green = (g > r + 60) & (g > 90)
        red = (r > 150) & (g < 110) & (b < 110)
        self.assertGreater(green.sum(), 500)  # nét được đọc tô xanh
        self.assertGreater(red[15:40, 245:272].sum(), 50)  # vết bị bỏ qua tô đỏ
        # Hai chữ số cạnh nhau: được gộp, báo số cụm; điểm thấp thì nét tô đỏ.
        two = np.full((300, 440), 245, np.uint8)
        two[38:262, 10:234] = paper(self.x[0])[38:262, 38:262]
        two[38:262, 200:424] = np.minimum(two[38:262, 200:424], paper(self.x[1])[38:262, 38:262])
        result = self.recognizer.digit(two)
        self.assertEqual(result["merged_parts"], 2)
        if not result["confident"]:
            rgb = rgb_of(result)
            self.assertGreater(((rgb[..., 0] > 150) & (rgb[..., 1] < 110)).sum(), 500)

    def test_uneven_lighting_uses_background_subtraction(self):
        _, trace = preprocess_digit(paper(self.x[0], light=0.6))
        self.assertEqual(trace["foreground_method"], "background_subtraction")
        self.assertLess(trace["bbox"][3], 280)  # trước khi sửa: khung trùm cả ảnh (300)
        self.assertGreaterEqual(self.accuracy(light=0.6), 0.9)  # trước khi sửa: ~0,25

    def test_ruled_paper_lines_are_removed(self):
        _, trace = preprocess_digit(paper(self.x[1], lines=True))
        self.assertGreater(trace["ruled_line_pixels"], 0)
        self.assertGreaterEqual(self.accuracy(lines=True), 0.95)

    def test_mnist_scale_input_is_unchanged(self):
        # ảnh 28 x 28 không bị trừ nền, xóa dòng kẻ hay làm dày nét
        _, trace = preprocess_digit(255 - self.x[2])
        self.assertEqual(trace["foreground_method"], "global_otsu")
        self.assertEqual(trace["stroke_thickened_px"], 0)

    def test_vertical_one_is_not_a_ruled_line(self):
        mask = np.zeros((300, 300), bool)
        mask[40:260, 140:150] = True  # nét số 1 dài 73% chiều cao ảnh
        cleaned, removed = remove_ruled_lines(mask)
        self.assertEqual(removed, 0)
        self.assertTrue((cleaned == mask).all())

    def test_thin_pen_stroke_is_thickened(self):
        img = np.full((300, 300), 255, np.uint8)
        yy, xx = np.indices(img.shape)
        ring = np.abs(np.hypot(yy - 150, xx - 150) - 90) <= 1.5  # số 0 nét bút bi 3 px
        img[ring] = 0
        _, trace = preprocess_digit(img)
        self.assertGreater(trace["stroke_thickened_px"], 0)


class ShapeFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = Recognizer()

    def targets(self, image):
        return sorted(
            o["class"] for o in self.recognizer.geometry(image, previews=False)["objects"] if o["target"]
        )

    def test_open_hand_drawn_outlines_are_bridged(self):
        yy, xx = np.indices((H, W))
        for gap in (6, 10, 16):
            circle = raster_shape("circle", H, W, 150, 200, 80, 80, 0, 6) & ~(
                (np.abs(yy - 200) < gap / 2 + 0.1) & (xx > 200)
            )
            rect = raster_shape("rectangle", H, W, 420, 200, 100, 70, 0, 6) & ~(
                (np.abs(xx - 420) < gap / 2 + 0.1) & (yy < 160)
            )
            self.assertEqual(self.targets(scene([circle, rect])), ["circle", "rectangle"], gap)

    def test_one_pixel_outlines_survive_median_filter(self):
        parts = [
            raster_shape("circle", H, W, 150, 200, 70, 70, 0, 1),
            raster_shape("rectangle", H, W, 420, 200, 110, 60, 0, 1),
        ]
        self.assertEqual(self.targets(scene(parts)), ["circle", "rectangle"])

    def test_tiny_region_is_not_guessed(self):
        result = self.recognizer.geometry(
            scene([raster_shape("circle", H, W, 100, 100, 5, 5)]), previews=False
        )
        self.assertEqual([o["class"] for o in result["objects"]], ["unknown"])
        self.assertTrue(any("12 px" in w for w in result["warnings"]))

    def test_uneven_lighting_keeps_the_rectangle(self):
        parts = [
            raster_shape("circle", H, W, 150, 200, 70, 70),
            raster_shape("rectangle", H, W, 420, 200, 110, 60),
        ]
        self.assertEqual(self.targets(scene(parts, light=0.6)), ["circle", "rectangle"])

    def test_regular_hexagon_is_not_a_circle(self):
        yy, xx = np.indices((H, W))
        hexagon = np.ones((H, W), bool)
        for k in range(6):
            hexagon &= ((xx - 300.0) * np.cos(k * np.pi / 3) + (yy - 200.0) * np.sin(k * np.pi / 3)) <= 90
        result = self.recognizer.geometry(scene([hexagon]), previews=False)
        self.assertEqual([o["class"] for o in result["objects"]], ["polygon"])

    def test_result_image_colors_regions_by_status(self):
        import base64
        import io

        from PIL import Image

        parts = [raster_shape("circle", H, W, 150, 200, 60, 60), raster_shape("circle", H, W, 500, 80, 5, 5)]
        result = self.recognizer.geometry(scene(parts))
        self.assertEqual(result["unknown_count"], 1)
        unknown = next(o for o in result["objects"] if o["class"] == "unknown")
        self.assertEqual(unknown["unknown_reason"], "too_small")
        png = base64.b64decode(result["annotated"].split(",", 1)[1])
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        # Vùng chưa xác định: tô và đóng khung đỏ (#d63a3a).
        x, y, w, h = unknown["bbox"]
        red = (r > 150) & (g < 110) & (b < 110)
        self.assertGreater(red[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3].sum(), 10)
        # Hình tròn đã xác định: bên trong được nhuộm xanh, không còn đen như ảnh gốc.
        self.assertGreater(g[200, 150], r[200, 150] + 60)
        self.assertGreater(g[200, 150], 80)
        # Ảnh "ẩn vùng chưa xác định": không còn màu đỏ quanh vùng đó.
        png = base64.b64decode(result["annotated_known"].split(",", 1)[1])
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
        red = (rgb[..., 0] > 150) & (rgb[..., 1] < 110) & (rgb[..., 2] < 110)
        self.assertEqual(red[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3].sum(), 0)


class Chapter7Descriptors(unittest.TestCase):
    def test_circularity_matches_textbook_and_is_rotation_invariant(self):
        circle = geometric_features(raster_shape("circle", 300, 300, 150, 150, 100, 100))
        square = geometric_features(raster_shape("rectangle", 300, 300, 150, 150, 80, 80))
        rotated = geometric_features(raster_shape("rectangle", 300, 300, 150, 150, 80, 80, np.pi / 4))
        self.assertGreater(circle["circularity"], 0.88)  # lý thuyết: 1 (trước khi sửa: 0,61)
        self.assertAlmostEqual(square["circularity"], np.pi / 4, delta=0.03)
        self.assertAlmostEqual(
            rotated["circularity"], square["circularity"], delta=0.03
        )  # trước: 0,39 vs 0,79
        self.assertGreater(circle["circularity"], square["circularity"])

    def test_chain_code_perimeter_of_a_square(self):
        mask = np.zeros((12, 12), bool)
        mask[2:8, 3:9] = True  # hình vuông 6 x 6: biên qua tâm điểm ảnh dài 4 * 5
        boundary = trace_boundary(mask)
        self.assertEqual(len(boundary), 20)
        self.assertEqual(chain_perimeter(chain_code(boundary)), 20.0)
        self.assertEqual(set(chain_code(boundary)), {0, 2, 4, 6})

    def test_signature_counts_polygon_corners(self):
        triangle = geometric_features(raster_shape("triangle", 300, 300, 150, 150, 90, 80))
        square = geometric_features(raster_shape("rectangle", 300, 300, 150, 150, 80, 80))
        circle = geometric_features(raster_shape("circle", 300, 300, 150, 150, 100, 100))
        self.assertEqual((triangle["corners"], square["corners"], circle["corners"]), (3, 4, 0))

    def test_fast_fill_holes_matches_flood_fill_definition(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            mask = rng.random((int(rng.integers(1, 40)), int(rng.integers(1, 40)))) < rng.uniform(0.2, 0.8)
            # định nghĩa: điểm nền thuộc lỗ nếu không nối (lân cận 4) được với viền ảnh
            exterior = np.zeros_like(mask)
            stack = [
                (y, x)
                for y in range(mask.shape[0])
                for x in range(mask.shape[1])
                if (y in (0, mask.shape[0] - 1) or x in (0, mask.shape[1] - 1)) and not mask[y, x]
            ]
            for y, x in stack:
                exterior[y, x] = True
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if (
                        0 <= ny < mask.shape[0]
                        and 0 <= nx < mask.shape[1]
                        and not mask[ny, nx]
                        and not exterior[ny, nx]
                    ):
                        exterior[ny, nx] = True
                        stack.append((ny, nx))
            self.assertTrue((fill_holes(mask) == ~exterior).all())


if __name__ == "__main__":
    unittest.main()
