"""Course algorithms: point transforms, filtering, Otsu, morphology, region growing.

No cv2/scipy/skimage image-processing calls are used in this module.
Coordinates are (x, y); bounding boxes are (x, y, width, height), half-open.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def grayscale(image):
    a = np.asarray(image)
    if a.ndim == 2:
        return np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("Ảnh phải là grayscale, RGB hoặc RGBA.")
    rgb = a[..., :3].astype(np.float32)
    if a.shape[2] == 4:
        alpha = a[..., 3:4].astype(np.float32) / 255
        rgb = rgb * alpha + 255 * (1 - alpha)
    return np.clip(np.rint(rgb @ np.array([.299, .587, .114])), 0, 255).astype(np.uint8)


def resize_bilinear(image, shape):
    """Bilinear sampling at pixel centers, with replicated boundary values."""
    a = np.asarray(image, dtype=np.float32)
    h, w = map(int, shape)
    if min(h, w) < 1:
        raise ValueError("Kích thước đầu ra phải dương.")
    yy = np.clip((np.arange(h) + .5) * a.shape[0] / h - .5, 0, a.shape[0] - 1)
    xx = np.clip((np.arange(w) + .5) * a.shape[1] / w - .5, 0, a.shape[1] - 1)
    y0, x0 = yy.astype(int), xx.astype(int)
    y1, x1 = np.minimum(y0 + 1, a.shape[0] - 1), np.minimum(x0 + 1, a.shape[1] - 1)
    fy, fx = yy - y0, xx - x0
    if a.ndim == 3:
        fy, fx = fy[:, None, None], fx[None, :, None]
    else:
        fy, fx = fy[:, None], fx[None, :]
    top = a[y0[:, None], x0] * (1 - fx) + a[y0[:, None], x1] * fx
    bottom = a[y1[:, None], x0] * (1 - fx) + a[y1[:, None], x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def _area_weights(old, new):
    edges = np.arange(new + 1, dtype=np.float32) * old / new
    left = np.maximum(edges[:-1, None], np.arange(old)[None, :])
    right = np.minimum(edges[1:, None], np.arange(1, old + 1)[None, :])
    return (np.maximum(right - left, 0) / (old / new)).astype(np.float32)


def resize_area(image, shape):
    """Exact pixel-overlap averaging for downsampling a grayscale image."""
    a = np.asarray(image, dtype=np.float32)
    h, w = shape
    if h >= a.shape[0] and w >= a.shape[1]:
        return resize_bilinear(a, shape)
    return (_area_weights(a.shape[0], h) @ a @ _area_weights(a.shape[1], w).T).astype(np.float32)


def median3(image):
    a = np.pad(np.asarray(image), 1, mode="edge")
    return np.median(sliding_window_view(a, (3, 3)), axis=(-2, -1)).astype(np.uint8)


def correlate2d(image, kernel):
    k = np.asarray(kernel, dtype=np.float32)
    if k.ndim != 2 or min(k.shape) < 1 or any(s % 2 == 0 for s in k.shape):
        raise ValueError("Kernel tích chập phải có hai kích thước lẻ dương.")
    a = np.pad(np.asarray(image, dtype=np.float32), ((k.shape[0] // 2,) * 2, (k.shape[1] // 2,) * 2), mode="edge")
    return np.einsum("ijmn,mn->ij", sliding_window_view(a, k.shape), k, optimize=True)


def sobel(image):
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gx, gy = correlate2d(image, k), correlate2d(image, k.T)
    mag = np.hypot(gx, gy)
    return (mag / max(float(mag.max()), 1) * 255).astype(np.uint8)


def otsu_threshold(image):
    """Maximize between-class variance; choose first maximum deterministically."""
    a = np.clip(image, 0, 255).astype(np.uint8)
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1)
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    denominator = omega * (1 - omega)
    score = np.zeros(256)
    valid = denominator > 1e-12
    score[valid] = (mu[-1] * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    return int(np.argmax(score))


def foreground(gray, denoise=False):
    """Otsu + choose the class differing from the median border background.

    Assumption: a mostly uniform background surrounds separated objects.
    A black canvas/white digit and a white canvas/black digit both work.
    """
    clean = median3(gray) if denoise else np.asarray(gray)
    if int(clean.max()) - int(clean.min()) < 8:
        return np.zeros_like(clean, dtype=bool), clean, 0, "uniform"
    t = otsu_threshold(clean)
    bright = clean > t
    border = np.concatenate((clean[0], clean[-1], clean[:, 0], clean[:, -1]))
    dark_objects = float(np.median(border)) > t
    mask = ~bright if dark_objects else bright
    return mask, clean, t, "dark_on_light" if dark_objects else "light_on_dark"


def dilate(mask, size=3):
    _kernel_size(size)
    pad = size // 2
    windows = sliding_window_view(np.pad(np.asarray(mask, bool), pad, constant_values=False), (size, size))
    return np.any(windows, axis=(-2, -1))


def erode(mask, size=3):
    _kernel_size(size)
    pad = size // 2
    windows = sliding_window_view(np.pad(np.asarray(mask, bool), pad, constant_values=False), (size, size))
    return np.all(windows, axis=(-2, -1))


def closing(mask, size=3):
    _kernel_size(size)
    # Extra padding avoids deleting objects touching the image boundary.
    p = size
    a = np.pad(mask, p, constant_values=False)
    return erode(dilate(a, size), size)[p:-p, p:-p]


def _kernel_size(size):
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1 or size % 2 == 0:
        raise ValueError("Kích thước phần tử cấu trúc phải là số nguyên lẻ dương.")


def box_mean(image, size):
    """Replicated-border mean filter using a NumPy summed-area table (Ch. 2)."""
    _kernel_size(size)
    a = np.pad(np.asarray(image, np.float64), size // 2, mode="edge")
    integral = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return ((integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size]
             + integral[:-size, :-size]) / (size * size)).astype(np.float32)


def fill_holes(mask):
    """Flood the complement from the border using 4-connectivity (Ch. 5)."""
    a = np.asarray(mask, bool)
    h, w = a.shape
    exterior = np.zeros_like(a)
    queue = deque()
    for y, x in [(0, x) for x in range(w)] + [(h - 1, x) for x in range(w)] + [(y, 0) for y in range(h)] + [(y, w - 1) for y in range(h)]:
        if not a[y, x] and not exterior[y, x]:
            exterior[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not a[ny, nx] and not exterior[ny, nx]:
                exterior[ny, nx] = True
                queue.append((ny, nx))
    return ~exterior


@dataclass
class Component:
    x: int
    y: int
    width: int
    height: int
    area: int
    mask: np.ndarray

    @property
    def bbox(self):
        return [self.x, self.y, self.width, self.height]


def connected_components(mask, min_area=1):
    """Two-pass run-length region labeling with an 8-connected union-find.

    Adjacent horizontal runs join if they overlap or touch diagonally.
    This keeps the algorithm explicit without a Python loop over every pixel.
    """
    a = np.asarray(mask, bool)
    parent, runs, previous = [], [], []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for y, row in enumerate(a):
        edges = np.diff(np.pad(row.astype(np.int8), 1))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        current = []
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            label = len(parent)
            parent.append(label)
            for p0, p1, plabel in previous:
                if p0 > x1:
                    break
                if p1 >= x0:
                    root = find(plabel)
                    parent[find(label)] = root
            runs.append((y, x0, x1, label))
            current.append((x0, x1, label))
        previous = current
    groups = {}
    for y, x0, x1, label in runs:
        groups.setdefault(find(label), []).append((y, x0, x1))
    output = []
    for group in groups.values():
        area = sum(x1 - x0 for _, x0, x1 in group)
        if area < min_area:
            continue
        x0 = min(r[1] for r in group)
        x1 = max(r[2] for r in group)
        y0, y1 = group[0][0], group[-1][0] + 1
        crop = np.zeros((y1 - y0, x1 - x0), bool)
        for y, start, end in group:
            crop[y - y0, start - x0:end - x0] = True
        output.append(Component(x0, y0, x1 - x0, y1 - y0, area, crop))
    return sorted(output, key=lambda c: (c.y, c.x))


def centered_crop(crop, size=28, inner=20, center_mass=False):
    a = np.asarray(crop, dtype=np.float32)
    scale = inner / max(a.shape)
    h, w = max(1, round(a.shape[0] * scale)), max(1, round(a.shape[1] * scale))
    resized = resize_area(a, (h, w))
    result = np.zeros((size, size), dtype=np.float32)
    y, x = (size - h) // 2, (size - w) // 2
    result[y:y + h, x:x + w] = resized
    if center_mass and result.sum() > 0:
        yy, xx = np.indices(result.shape)
        cy, cx = (result * yy).sum() / result.sum(), (result * xx).sum() / result.sum()
        dy, dx = round((size - 1) / 2 - cy), round((size - 1) / 2 - cx)
        shifted = np.zeros_like(result)
        ya, yb = max(0, dy), min(size, size + dy)
        xa, xb = max(0, dx), min(size, size + dx)
        shifted[ya:yb, xa:xb] = result[ya - dy:yb - dy, xa - dx:xb - dx]
        result = shifted
    return result


def preprocess_digit(image):
    gray = grayscale(image)
    mask, _, threshold, polarity = foreground(gray)
    parts = connected_components(mask, min_area=max(3, int(mask.size * .00004)))
    if not parts:
        raise ValueError("Chưa thấy nét chữ. Hãy vẽ một chữ số rõ hơn.")
    # Digit mode is single-character classification, never silently concatenate digits.
    significant = [c for c in parts if c.area >= max(c.area for c in parts) * .18]
    x0, y0 = min(c.x for c in significant), min(c.y for c in significant)
    x1, y1 = max(c.x + c.width for c in significant), max(c.y + c.height for c in significant)
    ink = (255 - gray if polarity == "dark_on_light" else gray).astype(np.float32)
    # Otsu gives the support; original gray values retain antialiased strokes.
    support = np.zeros_like(mask)
    for c in significant:
        support[c.y:c.y + c.height, c.x:c.x + c.width] |= c.mask
    ink *= support
    normalized = centered_crop(ink[y0:y1, x0:x1], 28, 20, center_mass=True) / 255
    return normalized, {"gray": gray, "binary": mask.astype(np.uint8) * 255, "normalized": normalized * 255,
                        "bbox": [x0, y0, x1 - x0, y1 - y0], "threshold": threshold, "polarity": polarity,
                        "components": len(significant)}


def normalize_shape(mask, fill=True):
    filled = fill_holes(mask) if fill else np.asarray(mask, bool)
    ys, xs = np.nonzero(filled)
    if len(xs) == 0:
        return np.zeros((32, 32), np.float32)
    return centered_crop(filled[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32), 32, 26)


def geometric_features(mask):
    a = fill_holes(mask)
    ys, xs = np.nonzero(a)
    area = len(xs)
    if not area:
        return {}
    padded = np.pad(a, 1)
    perimeter = int(np.count_nonzero(padded[1:] != padded[:-1]) + np.count_nonzero(padded[:, 1:] != padded[:, :-1]))
    covariance = np.cov(np.vstack((xs, ys)), bias=True) if area > 1 else np.eye(2)
    eig = np.linalg.eigvalsh(covariance)
    return {"area": area, "perimeter_4": perimeter,
            "circularity_4": float(4 * np.pi * area / max(perimeter ** 2, 1)),
            "extent": float(a.mean()), "aspect_ratio": float(a.shape[1] / a.shape[0]),
            "eccentricity": float(np.sqrt(max(0, 1 - eig[0] / max(eig[-1], 1e-8))))}
