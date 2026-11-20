"""Explicit NumPy region proposals: Ch. 2 filtering, Ch. 5 morphology,
Ch. 6 Otsu/region labeling, Ch. 7 region geometry. No pretrained detector.
"""
from collections import Counter

import numpy as np

from .processing import (box_mean, closing, connected_components, fill_holes,
                         foreground, median3, otsu_threshold, sobel)


def region_proposals(gray, mode="illumination_regions", profile=None):
    profile = profile or {}
    if mode == "baseline_v2":
        mask, clean, threshold, polarity = foreground(gray, denoise=True)
        joined = closing(mask)
        parts = [c for c in connected_components(joined, max(25, int(gray.size * .0001)))
                 if min(c.width, c.height) >= 5]
        parts = sorted(parts, key=lambda c: c.area, reverse=True)[:40]
        return [(c, c.mask, "baseline") for c in parts], {
            "clean": clean, "mask": mask, "joined": joined, "threshold": threshold,
            "polarity": polarity, "rejected": {}, "candidate_count": len(parts), "banks": ["baseline"]}
    if mode not in ("global_regions", "illumination_regions", "edge_regions"):
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
        if mode == "illumination_regions" and border_std >= float(profile.get("illumination_border_std", 12)):
            size = int(profile.get("background_window", 101))
            residual = clean.astype(np.float32) - box_mean(clean, size)
            for name, signed in (("local_dark", -residual), ("local_light", residual)):
                contrast = np.clip(signed, 0, 255).astype(np.uint8)
                cut = max(int(profile.get("minimum_contrast", 10)), otsu_threshold(contrast))
                banks.append((name, contrast > cut))
        if mode == "edge_regions":
            edges = sobel(clean)
            cut = max(12, otsu_threshold(edges))
            banks.extend((("edges_otsu", edges > cut), ("edges_low", edges > max(8, cut // 2))))
    rejected, candidates = Counter(), []
    min_area = max(25, int(gray.size * float(profile.get("min_area_fraction", .0001))))
    h, w = gray.shape
    combined = np.zeros_like(clean, bool)
    for name, mask in banks:
        joined = closing(mask, 5 if name.startswith("edges_") else 3)
        for c in connected_components(joined, min_area):
            if min(c.width, c.height) < 5:
                rejected["too_thin"] += 1
                continue
            edges = sum((c.x == 0, c.y == 0, c.x + c.width == w, c.y + c.height == h))
            # Border-connected complements must not become a giant rectangle.
            if edges >= 3:
                rejected["frame_background"] += 1
                continue
            filled = fill_holes(c.mask)
            if filled.mean() < float(profile.get("minimum_extent", .22)):
                rejected["sparse_region"] += 1
                continue
            candidates.append((c, filled, name))
        combined |= joined
    limit = int(profile.get("candidate_limit", 160))
    if len(candidates) > limit:
        rejected["candidate_limit"] += len(candidates) - limit
        candidates = sorted(candidates, key=lambda v: v[0].area, reverse=True)[:limit]
    return candidates, {"clean": clean, "mask": clean <= threshold if banks else combined,
                        "joined": combined, "threshold": threshold, "polarity": "both_polarities",
                        "rejected": dict(rejected), "candidate_count": len(candidates),
                        "background_border_std": border_std,
                        "banks": [name for name, _ in banks]}


def containment(child, parent):
    """Fraction of a child's filled pixels inside a parent's filled region."""
    c, cm, _ = child
    p, pm, _ = parent
    x0, y0 = max(c.x, p.x), max(c.y, p.y)
    x1, y1 = min(c.x + c.width, p.x + p.width), min(c.y + c.height, p.y + p.height)
    if x1 <= x0 or y1 <= y0:
        return 0.
    overlap = cm[y0-c.y:y1-c.y, x0-c.x:x1-c.x] & pm[y0-p.y:y1-p.y, x0-p.x:x1-p.x]
    return float(overlap.sum() / max(int(cm.sum()), 1))


def oriented_extent(mask):
    """Area / smallest sampled rotated bounding rectangle (Ch. 7 geometry).

    Boundary projections every 3 degrees avoid penalizing a rotated rectangle
    just because its axis-aligned bbox contains empty corners.
    """
    from .processing import erode
    yy, xx = np.nonzero(mask & ~erode(mask))
    if len(xx) > 2048:
        indices = np.linspace(0, len(xx)-1, 2048).astype(int)
        xx, yy = xx[indices], yy[indices]
    angles = np.deg2rad(np.arange(0, 90, 3))
    c, s = np.cos(angles), np.sin(angles)
    u, v = xx[:, None]*c + yy[:, None]*s, -xx[:, None]*s + yy[:, None]*c
    areas = (np.ptp(u, axis=0)+1)*(np.ptp(v, axis=0)+1)
    return float(min(1, mask.sum()/max(float(areas.min()), 1)))


def select_regions(candidates, probabilities, classes, min_confidence, profile=None):
    """CNN assigns classes; hierarchy/NMS remove internal details and duplicates.

    These rules implement the annotation policy of OUTER physical silhouettes.
    They do not split occluded/touching physical objects.
    """
    from .metrics import iou
    profile = profile or {}
    rejected = Counter()
    labels = [classes[int(p.argmax())] if float(p.max()) >= min_confidence else "unknown" for p in probabilities]
    extents = [oriented_extent(filled) for _, filled, _ in candidates]
    eligible = []
    for i, (_, filled, bank) in enumerate(candidates):
        if labels[i] == "rectangle" and extents[i] < float(profile.get("rectangle_extent", .78)):
            rejected["irregular_rectangle"] += 1
        elif bank.startswith("local_") and extents[i] < .70:
            rejected["irregular_local_region"] += 1
        else:
            eligible.append(i)
    # Parent selection uses CNN confidence, plus filled-mask containment rather
    # than bbox containment alone (a hole between separate objects stays free).
    valid = []
    for i in eligible:
        c, filled, _ = candidates[i]
        internal = any(j != i and labels[j] in ("circle", "rectangle", "triangle", "ellipse")
                       and float(probabilities[j].max()) >= min_confidence
                       and candidates[j][1].sum() > filled.sum() * 1.18
                       and (containment(candidates[i], candidates[j]) >= .96 or
                            (extents[j] >= .78 and c.x >= candidates[j][0].x and c.y >= candidates[j][0].y
                             and c.x+c.width <= candidates[j][0].x+candidates[j][0].width
                             and c.y+c.height <= candidates[j][0].y+candidates[j][0].height))
                       for j in eligible)
        if internal:
            rejected["internal_detail"] += 1
        else:
            valid.append(i)
    # Prefer the complete silhouette among near-identical multi-bank proposals.
    ordered = sorted(valid, key=lambda i: (-int(candidates[i][1].sum()), -float(probabilities[i].max()), i))
    kept = []
    for i in ordered:
        if any(iou(candidates[i][0].bbox, candidates[j][0].bbox) >= float(profile.get("nms_iou", .75)) for j in kept):
            rejected["duplicate_region"] += 1
        elif len(kept) < 40:
            kept.append(i)
        else:
            rejected["output_limit"] += 1
    return sorted(kept, key=lambda i: (candidates[i][0].y, candidates[i][0].x)), dict(rejected)
