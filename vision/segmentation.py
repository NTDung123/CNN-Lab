"""Explicit NumPy region proposals: Ch. 2 filtering, Ch. 5 morphology,
Ch. 6 Otsu/region labeling, Ch. 7 region geometry. No pretrained detector.
"""

from collections import Counter

import numpy as np

from .processing import (
    box_mean,
    closing,
    connected_components,
    dilate,
    estimate_background,
    fill_holes,
    foreground,
    gray_dilate,
    gray_erode,
    median3,
    otsu_threshold,
    sobel,
)


def fast_closing(mask, size):
    """Phép đóng nhị phân với phần tử vuông size x size (Chương 5).

    Phần tử vuông tách được thành hai đoạn thẳng, nên giãn/co theo hàng rồi theo cột
    (chi phí tỉ lệ với size thay vì size^2). Đệm thêm viền 0 để vật chạm mép không bị co mất.
    """
    p = size
    a = np.pad(np.asarray(mask, np.uint8), p)
    closed = gray_erode(gray_dilate(a, size), size) > 0
    return closed[p:-p, p:-p]


def bridge_open_outline(mask, max_size, min_gain=2.5):
    """Nối khe hở của nét viền vẽ tay chưa khép kín.

    Thử phép đóng với phần tử cấu trúc lớn dần (khe hở rộng g điểm ảnh cần phần tử
    cỡ khoảng g + 1). Dừng ở cỡ nhỏ nhất làm đường viền khép lại, tức là sau khi lấp lỗ
    diện tích tăng ít nhất min_gain lần. Phép đóng không vượt ra ngoài hộp bao của vùng
    nên không dính sang vật bên cạnh. Trả về mặt nạ đã lấp đầy, hoặc None.
    """
    area = int(np.count_nonzero(mask))
    sizes = [s for s in (5, 7, 9, 13, 17, 23, 31, 41, 55) if s <= max_size]
    if not sizes:
        return None
    # Cỡ lớn nhất không khép được thì cỡ nhỏ hơn cũng không: kiểm tra trước để bỏ qua nhanh.
    if fill_holes(fast_closing(mask, sizes[-1])).sum() < min_gain * area:
        return None
    for size in sizes:
        filled = fill_holes(fast_closing(mask, size))
        if filled.sum() >= min_gain * area:
            return filled
    return None


def region_proposals(gray, mode="illumination_regions", profile=None, rgb=None):
    profile = profile or {}
    if mode == "baseline_v2":
        mask, clean, threshold, polarity = foreground(gray, denoise=True)
        joined = closing(mask)
        parts = [
            c
            for c in connected_components(joined, max(25, int(gray.size * 0.0001)))
            if min(c.width, c.height) >= 5
        ]
        parts = sorted(parts, key=lambda c: c.area, reverse=True)[:40]
        return [(c, c.mask, "baseline") for c in parts], {
            "clean": clean,
            "mask": mask,
            "joined": joined,
            "threshold": threshold,
            "polarity": polarity,
            "rejected": {},
            "candidate_count": len(parts),
            "banks": ["baseline"],
        }
    if mode not in (
        "global_regions",
        "illumination_regions",
        "edge_regions",
        "contrast_regions",
        "repaired_regions",
    ):
        raise ValueError("Chế độ tách nền không hợp lệ.")
    clean = median3(gray)
    threshold = otsu_threshold(clean)
    border = np.concatenate((clean[0], clean[-1], clean[:, 0], clean[:, -1]))
    border_std = float(border.std())
    banks = []
    if int(clean.max()) - int(clean.min()) >= 8:
        banks = [("global_dark", clean <= threshold), ("global_light", clean > threshold)]
        # Uniform backgrounds already separate well globally. Local residuals
        # around a dark object can create a false bright halo; do not invent
        # additional regions when the observed background is nearly uniform.
        if mode in ("illumination_regions", "contrast_regions", "repaired_regions") and border_std >= float(
            profile.get("illumination_border_std", 12)
        ):
            size = int(profile.get("background_window", 101))
            if profile.get("background_method", "morphology") == "morphology":
                # Nền ước lượng bằng phép đóng (vật tối) / phép mở (vật sáng) ảnh xám, Ch.5.
                # Phần tử cấu trúc phải lớn hơn vật nên lấy theo cỡ ảnh. Nền đóng luôn >= ảnh,
                # nền mở luôn <= ảnh, nên không sinh quầng sáng/tối giả quanh vật như lọc
                # trung bình (nền trung bình bị chính vật kéo lệch).
                size = max(size, min(gray.shape) // 2) | 1
                flat = clean.astype(np.float32)
                residuals = (
                    ("local_dark", estimate_background(flat, size, dark_objects=True) - flat),
                    ("local_light", flat - estimate_background(flat, size, dark_objects=False)),
                )
            else:
                residual = clean.astype(np.float32) - box_mean(clean, size)
                residuals = (("local_dark", -residual), ("local_light", residual))
            for name, signed in residuals:
                contrast = np.clip(signed, 0, 255).astype(np.uint8)
                cut = max(int(profile.get("minimum_contrast", 10)), otsu_threshold(contrast))
                banks.append((name, contrast > cut))
        if mode == "repaired_regions" and profile.get("thin_line_bank", True):
            # Lọc trung vị 3x3 xóa luôn nét rộng 1 điểm ảnh (hình vẽ bằng bút mảnh, ảnh
            # xuất từ phần mềm). Với ảnh ít nhiễu, thêm nhánh ngưỡng trên ảnh gốc nếu
            # lọc trung vị đã làm mất một lượng nét đáng kể.
            noise = 1.4826 * float(np.median(np.abs(gray.astype(np.int16) - clean.astype(np.int16))))
            if noise < float(profile.get("thin_line_max_noise", 3)):
                for name, raw, cleaned in (
                    ("thin_dark", gray <= threshold, clean <= threshold),
                    ("thin_light", gray > threshold, clean > threshold),
                ):
                    lost = raw & ~dilate(cleaned, 3)
                    # chỉ khi lọc trung vị xóa mất phần lớn vật (dấu hiệu nét rộng 1 px),
                    # không phải vài điểm mép bình thường của ảnh chụp
                    if lost.sum() >= max(50, gray.size * 0.0002) and lost.sum() >= 0.25 * min(
                        raw.sum(), (~raw).sum()
                    ):
                        banks.append((name, raw))
        if mode in ("edge_regions", "contrast_regions", "repaired_regions"):
            edges = sobel(clean)
            cut = max(12, otsu_threshold(edges))
            banks.extend((("edges_otsu", edges > cut), ("edges_low", edges > max(8, cut // 2))))
        if mode in ("contrast_regions", "repaired_regions") and rgb is not None:
            # Ch. 3 color components: a colored object can have similar gray
            # intensity to the background. Threshold each RGB component using
            # exactly the same explicit Otsu implementation.
            for channel, channel_name in enumerate(("r", "g", "b")):
                component = median3(rgb[..., channel])
                t = otsu_threshold(component)
                banks.extend(
                    (
                        ("color_" + channel_name + "_dark", component <= t),
                        ("color_" + channel_name + "_light", component > t),
                    )
                )
    rejected, candidates = Counter(), []
    min_area = max(25, int(gray.size * float(profile.get("min_area_fraction", 0.0001))))
    h, w = gray.shape
    combined = np.zeros_like(clean, bool)
    branches = []
    bridge_attempts = 0
    for name, mask in banks:
        joined = closing(mask, 5 if name.startswith("edges_") else 3)
        branch_candidates = np.zeros_like(clean, bool)
        for c in connected_components(joined, min_area):
            if min(c.width, c.height) < 5:
                rejected["too_thin"] += 1
                continue
            edges = sum((c.x == 0, c.y == 0, c.x + c.width == w, c.y + c.height == h))
            # Border-connected complements must not become a giant rectangle.
            border_pixels = (int(c.mask[:, 0].sum()) if c.x == 0 else 0) + (
                int(c.mask[:, -1].sum()) if c.x + c.width == w else 0
            )
            border_pixels += (int(c.mask[0].sum()) if c.y == 0 else 0) + (
                int(c.mask[-1].sum()) if c.y + c.height == h else 0
            )
            # A circular object tangent to four image sides only occupies a
            # few border pixels; edge-count alone incorrectly removed it.
            if edges >= 3 and border_pixels / max(2 * (h + w), 1) >= 0.55:
                rejected["frame_background"] += 1
                continue
            filled = fill_holes(c.mask)
            # Nét viền hở (vẽ tay chưa khép kín): lấp lỗ không được gì và vùng chỉ là
            # một đường mảnh. Thử nối khe hở; nếu khép lại được thì thêm thành ứng viên.
            if (
                mode == "repaired_regions"
                and profile.get("bridge_open_outlines", True)
                and bridge_attempts < 60
                and filled.sum() < 1.3 * c.area
                and c.area < 0.35 * c.width * c.height
                and min(c.width, c.height) >= 15
            ):
                bridge_attempts += 1
                limit_size = max(
                    5, int(float(profile.get("bridge_gap_fraction", 0.3)) * min(c.width, c.height))
                )
                bridged = bridge_open_outline(c.mask, limit_size)
                if bridged is not None and bridged.mean() >= float(profile.get("minimum_extent", 0.22)):
                    candidates.append((c, bridged, name + "_bridged"))
                    branch_candidates[c.y : c.y + c.height, c.x : c.x + c.width] |= bridged
                    continue
            if filled.mean() < float(profile.get("minimum_extent", 0.22)):
                rejected["sparse_region"] += 1
                continue
            candidates.append((c, filled, name))
            branch_candidates[c.y : c.y + c.height, c.x : c.x + c.width] |= filled
        # A union of both raw polarities is necessarily white. Keep each
        # branch separately and show the surviving candidate regions instead.
        combined |= branch_candidates
        branches.append({"name": name, "binary": mask, "closed": joined, "candidates": branch_candidates})
    limit = int(profile.get("candidate_limit", 160))
    if mode == "repaired_regions":
        additions = []
        for c, filled, bank in sorted(candidates, key=lambda v: v[0].area, reverse=True)[:24]:
            if filled.sum() < max(100, gray.size * 0.001):
                continue
            hull = convex_hull_mask(filled)
            ratio = float(filled.sum() / max(int(hull.sum()), 1))
            if 0.78 <= ratio < 0.95:
                additions.append((c, hull, bank + "_hull"))
                original_preview = np.zeros_like(clean, bool)
                hull_preview = np.zeros_like(clean, bool)
                original_preview[c.y : c.y + c.height, c.x : c.x + c.width] = filled
                hull_preview[c.y : c.y + c.height, c.x : c.x + c.width] = hull
                branches.append(
                    {
                        "name": bank + "_hull",
                        "binary": original_preview,
                        "closed": hull_preview,
                        "candidates": hull_preview,
                    }
                )
        candidates.extend(additions)
    if len(candidates) > limit:
        rejected["candidate_limit"] += len(candidates) - limit
        candidates = sorted(candidates, key=lambda v: v[0].area, reverse=True)[:limit]
    combined.fill(False)
    for c, filled, _ in candidates:
        combined[c.y : c.y + c.height, c.x : c.x + c.width] |= filled
    representative = clean <= threshold if float(np.median(border)) > threshold else clean > threshold
    return candidates, {
        "clean": clean,
        "mask": representative if banks else combined,
        "joined": combined,
        "threshold": threshold,
        "polarity": "both_polarities",
        "rejected": dict(rejected),
        "candidate_count": len(candidates),
        "background_border_std": border_std,
        "branches": branches,
        "banks": [name for name, _ in banks],
    }


def convex_hull_mask(mask):
    """Ch. 5 convex hull: monotone boundary chain and NumPy half-plane raster.

    Used only for mildly concave proposals; it cannot recover arbitrary hidden
    objects. The original proposal remains available for CNN comparison.
    """
    from .processing import erode

    ys, xs = np.nonzero(mask & ~erode(mask))
    points = sorted(set(zip(xs.tolist(), ys.tolist())))
    if len(points) < 3:
        return np.asarray(mask, bool).copy()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return np.asarray(mask, bool).copy()
    yy, xx = np.indices(mask.shape)
    result = np.ones_like(mask, bool)
    for a, b in zip(hull, hull[1:] + hull[:1]):
        result &= (b[0] - a[0]) * (yy - a[1]) - (b[1] - a[1]) * (xx - a[0]) >= 0
    return result


def containment(child, parent):
    """Fraction of a child's filled pixels inside a parent's filled region."""
    c, cm, _ = child
    p, pm, _ = parent
    x0, y0 = max(c.x, p.x), max(c.y, p.y)
    x1, y1 = min(c.x + c.width, p.x + p.width), min(c.y + c.height, p.y + p.height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = cm[y0 - c.y : y1 - c.y, x0 - c.x : x1 - c.x] & pm[y0 - p.y : y1 - p.y, x0 - p.x : x1 - p.x]
    return float(overlap.sum() / max(int(cm.sum()), 1))


def oriented_extent(mask):
    """Area / smallest sampled rotated bounding rectangle (Ch. 7 geometry).

    Boundary projections every 3 degrees avoid penalizing a rotated rectangle
    just because its axis-aligned bbox contains empty corners.
    """
    from .processing import erode

    yy, xx = np.nonzero(mask & ~erode(mask))
    if len(xx) > 2048:
        indices = np.linspace(0, len(xx) - 1, 2048).astype(int)
        xx, yy = xx[indices], yy[indices]
    angles = np.deg2rad(np.arange(0, 90, 3))
    c, s = np.cos(angles), np.sin(angles)
    u, v = xx[:, None] * c + yy[:, None] * s, -xx[:, None] * s + yy[:, None] * c
    areas = (np.ptp(u, axis=0) + 1) * (np.ptp(v, axis=0) + 1)
    return float(min(1, mask.sum() / max(float(areas.min()), 1)))


def select_regions(candidates, probabilities, classes, min_confidence, profile=None):
    """CNN assigns classes; hierarchy/NMS remove internal details and duplicates.

    These rules implement the annotation policy of OUTER physical silhouettes.
    They do not split occluded/touching physical objects.
    """
    from .metrics import iou

    profile = profile or {}
    rejected = Counter()
    labels = [
        classes[int(p.argmax())] if float(p.max()) >= min_confidence else "unknown" for p in probabilities
    ]
    extents = [oriented_extent(filled) for _, filled, _ in candidates]
    eligible = []
    for i, (_, filled, bank) in enumerate(candidates):
        if labels[i] == "background":
            rejected["cnn_background"] += 1
        elif labels[i] == "rectangle" and extents[i] < float(profile.get("rectangle_extent", 0.78)):
            rejected["irregular_rectangle"] += 1
        elif bank.startswith("local_") and extents[i] < 0.70:
            rejected["irregular_local_region"] += 1
        else:
            eligible.append(i)
    # Parent selection uses CNN confidence, plus filled-mask containment rather
    # than bbox containment alone (a hole between separate objects stays free).
    valid = []
    for i in eligible:
        c, filled, _ = candidates[i]
        internal = any(
            j != i
            and labels[j] in ("circle", "rectangle", "triangle", "ellipse")
            and float(probabilities[j].max()) >= min_confidence
            and candidates[j][1].sum() > filled.sum() * 1.18
            and (
                containment(candidates[i], candidates[j]) >= 0.96
                or (
                    extents[j] >= 0.78
                    and c.x >= candidates[j][0].x
                    and c.y >= candidates[j][0].y
                    and c.x + c.width <= candidates[j][0].x + candidates[j][0].width
                    and c.y + c.height <= candidates[j][0].y + candidates[j][0].height
                )
            )
            for j in eligible
        )
        if internal:
            rejected["internal_detail"] += 1
        else:
            valid.append(i)
    # Prefer the complete silhouette among near-identical multi-bank proposals.
    ordered = sorted(
        valid,
        key=lambda i: (
            labels[i] == "unknown",
            -int(candidates[i][1].sum()),
            -float(probabilities[i].max()),
            i,
        ),
    )
    kept = []
    for i in ordered:
        if any(
            iou(candidates[i][0].bbox, candidates[j][0].bbox) >= float(profile.get("nms_iou", 0.75))
            for j in kept
        ):
            rejected["duplicate_region"] += 1
        elif len(kept) < 40:
            kept.append(i)
        else:
            rejected["output_limit"] += 1
    return sorted(kept, key=lambda i: (candidates[i][0].y, candidates[i][0].x)), dict(rejected)
