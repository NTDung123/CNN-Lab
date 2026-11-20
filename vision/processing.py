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
    return np.clip(np.rint(rgb @ np.array([0.299, 0.587, 0.114])), 0, 255).astype(np.uint8)


def resize_bilinear(image, shape):
    """Bilinear sampling at pixel centers, with replicated boundary values."""
    a = np.asarray(image, dtype=np.float32)
    h, w = map(int, shape)
    if min(h, w) < 1:
        raise ValueError("Kích thước đầu ra phải dương.")
    yy = np.clip((np.arange(h) + 0.5) * a.shape[0] / h - 0.5, 0, a.shape[0] - 1)
    xx = np.clip((np.arange(w) + 0.5) * a.shape[1] / w - 0.5, 0, a.shape[1] - 1)
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
    a = np.pad(
        np.asarray(image, dtype=np.float32), ((k.shape[0] // 2,) * 2, (k.shape[1] // 2,) * 2), mode="edge"
    )
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
    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
        or size % 2 == 0
    ):
        raise ValueError("Kích thước phần tử cấu trúc phải là số nguyên lẻ dương.")


def box_mean(image, size):
    """Replicated-border mean filter using a NumPy summed-area table (Ch. 2)."""
    _kernel_size(size)
    a = np.pad(np.asarray(image, np.float64), size // 2, mode="edge")
    integral = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )
        / (size * size)
    ).astype(np.float32)


def _label_runs(mask, connectivity=8):
    """Gán nhãn thành phần liên thông theo đoạn chạy + union-find, trả về ảnh nhãn.

    Mỗi hàng tách thành các đoạn điểm 1 liên tiếp [x0, x1). Hai đoạn ở hai hàng kề nhau
    thuộc cùng thành phần khi chúng chồng nhau (lân cận 4) hoặc chạm chéo (lân cận 8).
    """
    a = np.asarray(mask, bool)
    h, w = a.shape
    edges = np.diff(np.pad(a.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)
    n = rows.size
    if n == 0:
        return np.zeros((h, w), np.int32), 0
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    slack = 1 if connectivity == 8 else 0
    first = np.searchsorted(rows, np.arange(h + 1))
    s_list, e_list = starts.tolist(), ends.tolist()
    for r in range(1, h):
        i, i_end, j, j_end = int(first[r - 1]), int(first[r]), int(first[r]), int(first[r + 1])
        while i < i_end and j < j_end:
            if s_list[i] < e_list[j] + slack and s_list[j] < e_list[i] + slack:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
            if e_list[i] < e_list[j]:
                i += 1
            else:
                j += 1
    roots = np.array([find(i) for i in range(n)])
    _, run_label = np.unique(roots, return_inverse=True)
    run_label = (run_label + 1).astype(np.int32)
    delta = np.zeros((h, w + 1), np.int32)
    np.add.at(delta, (rows, starts), run_label)
    np.add.at(delta, (rows, ends), -run_label)
    return np.cumsum(delta, axis=1)[:, :w], int(run_label.max())


def fill_holes(mask):
    """Lấp lỗ (Chương 5): lỗ là phần nền liên thông-4 không chạm viền ảnh.

    Cho cùng kết quả với cách loang phần bù từ viền ảnh, nhưng gán nhãn theo đoạn chạy
    nên không phải lặp Python qua từng điểm ảnh.
    """
    a = np.asarray(mask, bool)
    if a.size == 0:
        return a.copy()
    labels, n = _label_runs(~a, connectivity=4)
    if n == 0:
        return np.ones_like(a)
    exterior = np.zeros(n + 1, bool)
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        exterior[edge] = True
    exterior[0] = False
    return ~exterior[labels]


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
            crop[y - y0, start - x0 : end - x0] = True
        output.append(Component(x0, y0, x1 - x0, y1 - y0, area, crop))
    return sorted(output, key=lambda c: (c.y, c.x))


def centered_crop(crop, size=28, inner=20, center_mass=False):
    a = np.asarray(crop, dtype=np.float32)
    scale = inner / max(a.shape)
    h, w = max(1, round(a.shape[0] * scale)), max(1, round(a.shape[1] * scale))
    resized = resize_area(a, (h, w))
    result = np.zeros((size, size), dtype=np.float32)
    y, x = (size - h) // 2, (size - w) // 2
    result[y : y + h, x : x + w] = resized
    if center_mass and result.sum() > 0:
        yy, xx = np.indices(result.shape)
        cy, cx = (result * yy).sum() / result.sum(), (result * xx).sum() / result.sum()
        dy, dx = round((size - 1) / 2 - cy), round((size - 1) / 2 - cx)
        shifted = np.zeros_like(result)
        ya, yb = max(0, dy), min(size, size + dy)
        xa, xb = max(0, dx), min(size, size + dx)
        shifted[ya:yb, xa:xb] = result[ya - dy : yb - dy, xa - dx : xb - dx]
        result = shifted
    return result


# --------------------------------------------------------------------------
# Hình thái học trên ảnh xám (Chương 5): giãn = max, co = min trên phần tử cấu
# trúc vuông phẳng. Phần tử vuông tách được thành hai đoạn 1 chiều nên tính
# max/min theo hàng rồi theo cột.
# --------------------------------------------------------------------------
def _running_extreme(a, size, axis, reducer):
    pad = [(0, 0)] * a.ndim
    pad[axis] = (size // 2, size // 2)
    padded = np.pad(a, pad, mode="edge")
    return reducer(sliding_window_view(padded, size, axis=axis), axis=-1)


def gray_dilate(image, size):
    """Giãn ảnh xám: mỗi điểm nhận giá trị lớn nhất trong cửa sổ size x size."""
    _kernel_size(size)
    a = np.asarray(image, np.float32)
    return _running_extreme(_running_extreme(a, size, 0, np.max), size, 1, np.max)


def gray_erode(image, size):
    """Co ảnh xám: mỗi điểm nhận giá trị nhỏ nhất trong cửa sổ size x size."""
    _kernel_size(size)
    a = np.asarray(image, np.float32)
    return _running_extreme(_running_extreme(a, size, 0, np.min), size, 1, np.min)


def estimate_background(gray, size, dark_objects):
    """Ước lượng nền biến thiên chậm (chiếu sáng không đều) bằng hình thái học xám.

    Vật tối trên nền sáng: phép đóng (giãn rồi co) xóa mọi nét tối mảnh hơn phần tử
    cấu trúc, phần còn lại là nền. Vật sáng trên nền tối: dùng phép mở (co rồi giãn).
    Ảnh lớn được thu nhỏ trước khi lọc vì nền thay đổi chậm, rồi phóng lại kích thước cũ.
    """
    a = np.asarray(gray, np.float32)
    factor = max(1, int(np.ceil(max(a.shape) / 300)))
    small = resize_area(a, (max(1, a.shape[0] // factor), max(1, a.shape[1] // factor))) if factor > 1 else a
    k = max(3, size // factor) | 1
    if dark_objects:
        background = gray_erode(gray_dilate(small, k), k)
    else:
        background = gray_dilate(gray_erode(small, k), k)
    background = box_mean(background, max(3, (k // 2) | 1))  # làm trơn bậc thang của phần tử vuông
    return resize_bilinear(background, a.shape) if factor > 1 else background


def border_variation(gray):
    """Độ biến thiên của nền trên viền ảnh: P90 - P10 sau lọc trung bình 9 x 9.

    Lọc trung bình loại nhiễu hạt; dùng phân vị nên một nét chữ chạm viền không làm
    sai lệch. Giá trị lớn nghĩa là độ sáng nền đổi theo vị trí (bóng, ánh sáng lệch).
    """
    smooth = box_mean(gray, 9)
    border = np.concatenate((smooth[0], smooth[-1], smooth[:, 0], smooth[:, -1]))
    return float(np.percentile(border, 90) - np.percentile(border, 10))


def remove_ruled_lines(mask, edge_margin=0.04):
    """Xóa các đường kẻ ngang/dọc chạy từ mép này sang mép kia của ảnh (giấy kẻ dòng).

    Đường kẻ là một đoạn chạy liên tục bắt đầu sát một mép và kết thúc sát mép đối diện
    (cách mép không quá edge_margin cạnh ảnh); nét chữ số gần như không bao giờ như vậy.
    Sau khi xóa, điểm nào của đường kẻ có nét chữ ở cả hai phía (trên-dưới hoặc
    trái-phải) thì được giữ lại để nét chữ cắt qua đường kẻ không bị đứt.
    """
    a = np.asarray(mask, bool)
    lines = np.zeros_like(a)
    for axis in (1, 0):
        b = a if axis == 1 else a.T
        out = lines if axis == 1 else lines.T
        n = b.shape[1]
        if n < 64:  # ảnh quá nhỏ (ví dụ 28 x 28) không có giấy kẻ
            continue
        margin = max(1, int(edge_margin * n))
        for i, row in enumerate(b):
            if row.mean() < 0.8:  # đường kẻ phủ gần hết hàng
                continue
            edges = np.diff(np.pad(row.astype(np.int8), 1))
            for x0, x1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if x0 <= margin and x1 >= n - margin:
                    out[i, x0:x1] = True
    if not lines.any():
        return a, 0
    kept = a & ~lines
    reach = 4
    up = np.zeros_like(a)
    down = np.zeros_like(a)
    left = np.zeros_like(a)
    right = np.zeros_like(a)
    for d in range(1, reach + 1):
        up[d:] |= kept[:-d]
        down[:-d] |= kept[d:]
        left[:, d:] |= kept[:, :-d]
        right[:, :-d] |= kept[:, d:]
    restored = lines & ((up & down) | (left & right))
    return kept | restored, int(lines.sum())


def stroke_width(mask):
    """Độ dày nét ước lượng w = 2A / B (A: diện tích, B: số điểm biên).

    Một nét dài L, dày w có diện tích w*L và khoảng 2L điểm biên (hai mép nét).
    """
    a = np.asarray(mask, bool)
    boundary = a & ~erode(a)
    return 2.0 * a.sum() / max(int(boundary.sum()), 1)


# Độ dày nét mục tiêu so với cạnh dài của chữ số. Chọn trên ảnh validation của MNIST
# (vẽ lại bằng bút mảnh 3-5 px, có nhiễu): giãn khi nét < 8%, giãn tới khoảng 12%.
STROKE_TARGET, STROKE_MIN = 0.12, 0.08


def thicken_to_mnist(ink, mask):
    """Giãn nét quá mảnh (bút bi trên ảnh lớn) để gần với độ dày nét của MNIST.

    Mạng chỉ học chữ số MNIST có nét tương đối dày; nét chỉ 2-3% chiều cao chữ làm
    sai nhiều. Giãn bằng phép giãn ảnh xám, xen kẽ phần tử chữ thập và vuông 3x3 để xấp
    xỉ một đĩa tròn. Trả về (ink, mask, bán kính đã giãn).
    """
    ys, xs = np.nonzero(mask)
    side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    # đóng 3x3 trước khi đo: lỗ kim do nhiễu trong nét sẽ làm tăng số điểm biên giả
    width = stroke_width(closing(mask, 3))
    # ảnh ở cỡ MNIST (chữ ~20 px) đã đúng phân phối huấn luyện, chỉ chỉnh ảnh độ phân giải cao
    if side < 40 or width / side >= STROKE_MIN:
        return ink, mask, 0
    radius = int(round((STROKE_TARGET * side - width) / 2))
    if radius < 1:
        return ink, mask, 0
    ink = np.pad(np.asarray(ink, np.float32), radius)
    mask = np.pad(np.asarray(mask, bool), radius)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    for step in range(radius):
        if step % 2 == 0:
            padded = np.pad(ink, 1, mode="edge")
            windows = sliding_window_view(padded, (3, 3))
            ink = np.where(cross, windows, -np.inf).max(axis=(-2, -1)).astype(np.float32)
            mask = (
                mask
                | np.pad(mask, 1)[1:-1, :-2]
                | np.pad(mask, 1)[1:-1, 2:]
                | np.pad(mask, 1)[:-2, 1:-1]
                | np.pad(mask, 1)[2:, 1:-1]
            )
        else:
            ink = gray_dilate(ink, 3)
            mask = dilate(mask, 3)
    return ink, mask, radius


def digit_foreground(gray):
    """Tách nét chữ số khỏi nền.

    Nền đều (canvas, ảnh quét): Otsu toàn cục như cũ. Nền không đều (ảnh chụp có bóng,
    ánh sáng lệch, giấy kẻ): ước lượng nền bằng phép đóng/mở ảnh xám, lấy phần chênh
    lệch so với nền (black-hat / top-hat) rồi mới Otsu. Cực tính (mực tối hay phấn sáng)
    được chọn theo nhánh cho phản hồi mạnh hơn.
    Trả về (mask, ink [0..255, nét sáng], threshold, polarity, method).
    """
    gray = np.asarray(gray, np.uint8)
    if int(gray.max()) - int(gray.min()) < 8:
        return np.zeros_like(gray, bool), np.zeros_like(gray, np.float32), 0, "uniform", "uniform"
    # ảnh nhỏ (ví dụ 28 x 28) chỉ chứa chữ số, viền không đủ để ước lượng nền
    if min(gray.shape) < 64 or border_variation(gray) < 20:
        mask, _, t, polarity = foreground(gray)
        ink = (255 - gray if polarity == "dark_on_light" else gray).astype(np.float32)
        return mask, ink, t, polarity, "global_otsu"
    clean = median3(gray).astype(np.float32)
    size = max(15, min(gray.shape) // 5) | 1
    bg_light = estimate_background(clean, size, dark_objects=True)  # nền sáng, nét tối
    bg_dark = estimate_background(clean, size, dark_objects=False)  # nền tối, nét sáng
    black_hat = np.clip(bg_light - clean, 0, None)
    white_hat = np.clip(clean - bg_dark, 0, None)
    # Chọn cực tính: làm phẳng ánh sáng bằng cách chia cho bản làm mờ diện rộng, khi đó nền
    # gần như đều; rồi như foreground(): lớp Otsu nào khác màu viền ảnh là nét chữ.
    # (So độ mạnh của black-hat/white-hat không tin cậy: khoảng giấy bị nét bao quanh, như
    # lòng số 6, cũng tạo phản hồi white-hat mạnh.)
    shading = box_mean(clean, max(15, min(gray.shape) // 2) | 1)
    flat = np.clip(clean / np.maximum(shading, 1) * 128, 0, 255)
    t_flat = otsu_threshold(flat)
    border = np.concatenate((flat[0], flat[-1], flat[:, 0], flat[:, -1]))
    dark_objects = float(np.median(border)) > t_flat
    if dark_objects:
        # chiếu sáng là nhân tính: chia cho độ sáng nền để vùng tối không mất tương phản
        ink = black_hat / np.maximum(bg_light, 16) * 255
    else:
        ink = white_hat / np.maximum(255 - bg_dark, 16) * 255
    ink = np.clip(ink, 0, 255).astype(np.float32)
    t = max(otsu_threshold(ink), 12)
    return ink > t, ink, t, "dark_on_light" if dark_objects else "light_on_dark", "background_subtraction"


def preprocess_digit(image):
    gray = grayscale(image)
    mask, ink, threshold, polarity, method = digit_foreground(gray)
    mask, ruled = remove_ruled_lines(mask)
    parts = connected_components(mask, min_area=max(3, int(mask.size * 0.00004)))
    if not parts:
        raise ValueError("Chưa thấy nét chữ. Hãy vẽ một chữ số rõ hơn.")
    # Digit mode is single-character classification, never silently concatenate digits.
    cutoff = max(c.area for c in parts) * 0.18
    significant = [c for c in parts if c.area >= cutoff]
    x0, y0 = min(c.x for c in significant), min(c.y for c in significant)
    x1, y1 = max(c.x + c.width for c in significant), max(c.y + c.height for c in significant)
    # Otsu gives the support; original gray values retain antialiased strokes.
    support = np.zeros_like(mask)
    for c in significant:
        support[c.y : c.y + c.height, c.x : c.x + c.width] |= c.mask
    crop_ink = (ink * support)[y0:y1, x0:x1]
    crop_mask = support[y0:y1, x0:x1]
    if method == "background_subtraction":
        # nền đã bị trừ nên độ đậm của nét phụ thuộc ánh sáng: đưa lõi nét về 255
        core = np.percentile(crop_ink[crop_mask], 90) if crop_mask.any() else 255
        crop_ink = np.clip(crop_ink * (255 / max(core, 1)), 0, 255)
    crop_ink, crop_mask, thickened = thicken_to_mnist(crop_ink, crop_mask)
    normalized = centered_crop(crop_ink, 28, 20, center_mass=True) / 255
    return normalized, {
        "gray": gray,
        "binary": mask.astype(np.uint8) * 255,
        "normalized": normalized * 255,
        "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "threshold": int(threshold),
        "polarity": polarity,
        "components": len(significant),
        "foreground_method": method,
        "ruled_line_pixels": ruled,
        "stroke_thickened_px": thickened,
        # Chỉ dùng để vẽ ảnh kết quả: nét được đưa vào CNN và các vết nhỏ bị bỏ qua.
        "support": support,
        "ignored": [c for c in parts if c.area < cutoff],
    }


def normalize_shape(mask, fill=True):
    filled = fill_holes(mask) if fill else np.asarray(mask, bool)
    ys, xs = np.nonzero(filled)
    if len(xs) == 0:
        return np.zeros((32, 32), np.float32)
    return centered_crop(filled[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32), 32, 26)


# --------------------------------------------------------------------------
# Chương 7: theo dõi biên, mã chuỗi và bộ mô tả vùng
# --------------------------------------------------------------------------
# 8 lân cận theo chiều kim đồng hồ bắt đầu từ phía tây, toạ độ (dy, dx), trục y hướng xuống
_MOORE = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
_MOORE_INDEX = {d: i for i, d in enumerate(_MOORE)}
# Mã Freeman 8 hướng: 0 = đông, tăng dần ngược chiều kim đồng hồ
_FREEMAN = {(0, 1): 0, (-1, 1): 1, (-1, 0): 2, (-1, -1): 3, (0, -1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7}


def trace_boundary(mask):
    """Theo dõi biên ngoài bằng thuật toán Moore (Ch.7, tr.9). Trả về mảng (K, 2) điểm (y, x).

    b0 là điểm 1 trên cùng bên trái, c0 là lân cận phía tây (luôn là nền). Xét 8 lân cận
    của b theo chiều kim đồng hồ bắt đầu từ c; điểm 1 đầu tiên là điểm biên kế tiếp, điểm
    nền ngay trước nó thành c mới. Dừng khi quay lại b0 và điểm kế tiếp lại là b1.
    """
    a = np.pad(np.asarray(mask, bool), 1)
    points = np.argwhere(a)
    if not len(points):
        return np.zeros((0, 2), int)
    b0 = (int(points[0][0]), int(points[0][1]))

    def step(b, c):
        start = _MOORE_INDEX[(c[0] - b[0], c[1] - b[1])]
        previous = c
        for k in range(1, 9):
            dy, dx = _MOORE[(start + k) % 8]
            q = (b[0] + dy, b[1] + dx)
            if a[q]:
                return q, previous
            previous = q
        return None, None

    b1, c1 = step(b0, (b0[0], b0[1] - 1))
    if b1 is None:  # một điểm cô lập
        return np.array([[b0[0] - 1, b0[1] - 1]])
    boundary, b, c = [b0], b1, c1
    for _ in range(4 * a.size):
        nb, nc = step(b, c)
        if b == b0 and nb == b1:
            break
        boundary.append(b)
        b, c = nb, nc
    return np.array(boundary) - 1


def chain_code(boundary):
    """Mã chuỗi Freeman 8 hướng của biên khép kín (Ch.7, tr.11)."""
    if len(boundary) < 2:
        return []
    steps = np.roll(boundary, -1, axis=0) - boundary
    return [_FREEMAN[(int(dy), int(dx))] for dy, dx in steps]


def chain_perimeter(code):
    """Chu vi theo mã chuỗi (Ch.7, tr.20): bước ngang/dọc dài 1, bước chéo dài sqrt(2)."""
    code = np.asarray(code)
    diagonal = int(np.count_nonzero(code % 2))
    return float(len(code) - diagonal + np.sqrt(2) * diagonal)


def boundary_signature(boundary, centroid, bins=180):
    """Biên dạng r(theta) (Ch.7, tr.17): khoảng cách từ trọng tâm tới biên theo góc, chia cho giá trị lớn nhất."""
    dy, dx = boundary[:, 0] - centroid[0], boundary[:, 1] - centroid[1]
    angle = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    signature = np.zeros(bins)
    np.maximum.at(signature, angle, np.hypot(dy, dx))
    empty = signature == 0
    if empty.all():
        return signature
    if empty.any():  # nội suy vòng cho góc không có điểm biên
        idx = np.arange(bins)
        good = idx[~empty]
        signature[empty] = np.interp(
            idx[empty], np.concatenate((good - bins, good, good + bins)), np.tile(signature[~empty], 3)
        )
    return signature / signature.max()


def signature_corners(signature, window=7, prominence=0.05):
    """Đếm số "góc" của biên: số cực đại rõ rệt của biên dạng đã làm trơn vòng.

    Hình tròn có biên dạng gần như hằng số (0 góc); đa giác n cạnh có n cực đại cách đều.
    """
    n = len(signature)
    padded = np.concatenate((signature[-window:], signature, signature[:window]))
    smooth = np.convolve(padded, np.ones(window) / window, mode="same")[window:-window]
    half = max(3, n // 16)
    corners = 0
    for i in range(n):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[(i + 1) % n]:
            around = smooth[np.arange(i - half, i + half + 1) % n]
            if smooth[i] >= around.max() - 1e-12 and smooth[i] - around.min() >= prominence:
                corners += 1
    return corners


def geometric_features(mask):
    """Bộ mô tả vùng của Chương 7 tính trên mặt nạ đã lấp lỗ.

    perimeter   : chu vi theo mã chuỗi (tr.20)
    circularity : độ tròn 4*pi*A / p^2 (tr.26): ~1 với hình tròn, pi/4 với hình vuông,
                  gần như không đổi khi quay hình
    compactness : độ nén p^2 / A
    extent      : A / diện tích hộp bao thẳng trục
    eccentricity: từ trị riêng ma trận hiệp phương sai toạ độ (tr.27)
    signature_std, corners: độ biến thiên và số cực đại của biên dạng r(theta)
    """
    a = fill_holes(mask)
    ys, xs = np.nonzero(a)
    area = len(xs)
    if not area:
        return {}
    boundary = trace_boundary(a)
    perimeter = chain_perimeter(chain_code(boundary)) if len(boundary) > 1 else 1.0
    covariance = np.cov(np.vstack((xs, ys)), bias=True) if area > 1 else np.eye(2)
    eig = np.linalg.eigvalsh(covariance)
    signature = boundary_signature(boundary, (ys.mean(), xs.mean())) if len(boundary) > 2 else np.ones(180)
    return {
        "area": area,
        "perimeter": round(perimeter, 2),
        "circularity": float(min(1.0, 4 * np.pi * area / max(perimeter**2, 1))),
        "compactness": float(perimeter**2 / area),
        "extent": float(a.mean()),
        "aspect_ratio": float(a.shape[1] / a.shape[0]),
        "eccentricity": float(np.sqrt(max(0, 1 - eig[0] / max(eig[-1], 1e-8)))),
        "signature_std": float(signature.std()),
        "corners": signature_corners(signature),
    }
