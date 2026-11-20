"""Bộ đo dùng khi sửa lỗi giai đoạn 5: ca thử dựng sẵn + ảnh thực development/validation.

Không bao giờ nạp ảnh thực TEST. Ca thử chữ số dùng ảnh validation MNIST (cùng hoán vị
với scripts/train.py), không dùng 10 000 ảnh test.

    python scripts/benchmark_phase5.py                       # đo mã hiện tại
    python scripts/benchmark_phase5.py reference_root out.json   # đo một bản mã khác
Kết quả: results/phase5_benchmark_current.json (mặc định).
"""

import json
import os
import sys
import time

args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = os.path.abspath(args[0]) if args else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_path = (
    os.path.abspath(args[1])
    if len(args) > 1
    else os.path.join(root, "results", "phase5_benchmark_current.json")
)
sys.path.insert(0, root)
os.chdir(root)
import numpy as np
from PIL import Image

from vision.metrics import detection_score, summarize_detection
from vision.processing import dilate, erode, preprocess_digit, resize_bilinear
from vision.real_data import read_real_manifest, real_image_path
from vision.service import Recognizer
from vision.synthetic import raster_shape

r = Recognizer()
out = {}
with np.load("data/mnist.npz") as d:
    xt, yt = d["x_test"], d["y_test"]
    # Tuning probes use the MNIST VALIDATION split (same permutation as scripts/train.py),
    # never the official 10k test set.
    _order = np.random.default_rng(42).permutation(60000)[:5000]
    xv, yv = d["x_train"][_order], d["y_train"][_order]


# ---------------------------------------------------------------- digits
def canvas(
    digit,
    rng,
    k=8,
    thin=0,
    thick=0,
    noise=0,
    sp=0,
    light=0,
    ink=(0, 0, 0),
    paper=(255, 255, 255),
    lines=False,
    size=300,
    pen=0,
):
    big = resize_bilinear(digit / 255.0, (28 * k, 28 * k))
    m = big
    if thin:
        m = np.where(erode(big > 0.5, 2 * thin + 1), big, 0)
    if thick:
        m = np.where(dilate(big > 0.5, 2 * thick + 1), 1.0, big)
    a = np.zeros((size, size))
    o = (size - 28 * k) // 2
    a[o : o + 28 * k, o : o + 28 * k] = m
    img = np.array(paper, float) * (1 - a[..., None]) + np.array(ink, float) * a[..., None]
    if pen:
        from numpy.lib.stride_tricks import sliding_window_view as swv

        local_max = swv(np.pad(big, 3, mode="edge"), (7, 7)).max(axis=(-2, -1))
        ridge = (big >= local_max - 0.02) & (big > 0.4)
        m = dilate(ridge, pen).astype(float)
        a = np.zeros((size, size))
        a[o : o + 28 * k, o : o + 28 * k] = m
        img = np.array(paper, float) * (1 - a[..., None]) + np.array(ink, float) * a[..., None]
    if lines:
        for y in range(15, size, 30):
            img[y : y + 2] = np.minimum(img[y : y + 2], np.array([150, 170, 220], float))
    if light:
        img *= (1 - light * np.linspace(0, 1, size))[None, :, None]
    if noise:
        img += rng.normal(0, noise, img.shape)
    if sp:
        u = rng.random((size, size))
        img[u < sp / 2] = 0
        img[(u >= sp / 2) & (u < sp)] = 255
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


DIGIT_CONDITIONS = {
    "chuan": {},
    "net_manh": dict(thin=3),
    "but_bi_3px": dict(k=9, pen=3),
    "but_bi_5px": dict(k=9, pen=5),
    "net_dam": dict(thick=6),
    "gauss25": dict(noise=25),
    "muoi_tieu_5": dict(sp=0.05),
    "sang_lech_40": dict(light=0.4),
    "sang_lech_60": dict(light=0.6),
    "sang_lech_60_nhieu": dict(light=0.6, noise=12),
    "muc_xanh_giay_vang": dict(ink=(25, 45, 150), paper=(245, 236, 200)),
    "phan_bang_den": dict(ink=(240, 240, 235), paper=(28, 42, 36)),
    "phan_bang_den_lech": dict(ink=(240, 240, 235), paper=(60, 80, 70), light=0.5),
    "but_chi_mo": dict(ink=(150, 150, 150), paper=(195, 195, 195)),
    "giay_ke_dong": dict(lines=True),
}
N = 200
digits = {}
t0 = time.time()
for name, kw in DIGIT_CONDITIONS.items():
    rng = np.random.default_rng(7)
    ok = 0
    for i in range(N):
        try:
            ok += int(r.digit(canvas(xv[i], rng, **kw), previews=False)["prediction"] == yv[i])
        except ValueError:
            pass
    digits[name] = round(ok / N, 4)
out["digits_canvas"] = digits
out["digits_seconds"] = round(time.time() - t0, 1)
if "--skip-mnist" not in sys.argv:
    t0 = time.time()
    inputs = np.stack([preprocess_digit(np.rint(255 - im).astype(np.uint8))[0] for im in xt[:3000]])
    from vision.models import predict

    p = np.concatenate([predict(r.digits, inputs[i : i + 500]) for i in range(0, len(inputs), 500)])
    out["mnist_pipeline_first3000"] = round(float((p.argmax(1) == yt[:3000]).mean()), 4)
    out["mnist_pipeline_seconds"] = round(time.time() - t0, 1)

# ---------------------------------------------------------------- shape probes
H, W = 400, 600
yy, xx = np.indices((H, W))


def S(kind, cx, cy, rx, ry, ang=0, outline=0):
    return raster_shape(kind, H, W, cx, cy, rx, ry, ang, outline)


def scene(parts, bg=245, fg=20, noise=0, light=0, seed=1):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), bg, float)
    for m in parts:
        img[m] = fg
    if light:
        img *= (1 - light * np.linspace(0, 1, W))[None, :, None]
    if noise:
        img += rng.normal(0, noise, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def open_pair(gap, stroke):
    c = S("circle", 150, 200, 80, 80, 0, stroke)
    c &= ~((np.abs(yy - 200) < gap / 2 + 0.1) & (xx > 200))
    rr = S("rectangle", 420, 200, 100, 70, 0, stroke)
    rr &= ~((np.abs(xx - 420) < gap / 2 + 0.1) & (yy < 160))
    return [c, rr]


u, v = xx - 300.0, yy - 200.0
hexm = np.ones((H, W), bool)
for k in range(6):
    hexm &= (u * np.cos(k * np.pi / 3) + v * np.sin(k * np.pi / 3)) <= 90


def wobbly(cx, cy, rad, amp, seed):
    rng = np.random.default_rng(seed)
    ang = np.arctan2(yy - cy, xx - cx)
    rr = rad * (1 + sum(rng.uniform(0, amp) * np.cos(k * ang + rng.uniform(0, 6.3)) for k in range(2, 7)))
    return np.hypot(yy - cy, xx - cx) <= rr


octa = np.ones((H, W), bool)
for k in range(8):
    octa &= (u * np.cos(k * np.pi / 4 + 0.3) + v * np.sin(k * np.pi / 4 + 0.3)) <= 95
PROBES = {
    "dac_chuan": (
        [S("circle", 150, 200, 70, 70), S("rectangle", 420, 200, 110, 60, 0.3)],
        {},
        ["circle", "rectangle"],
    ),
    "vuong_xoay45": ([S("rectangle", 300, 200, 80, 80, np.pi / 4)], {}, ["rectangle"]),
    "vien_3px": (
        [S("circle", 150, 200, 70, 70, 0, 3), S("rectangle", 420, 200, 110, 60, 0, 3)],
        {},
        ["circle", "rectangle"],
    ),
    "vien_1px": (
        [S("circle", 150, 200, 70, 70, 0, 1), S("rectangle", 420, 200, 110, 60, 0, 1)],
        {},
        ["circle", "rectangle"],
    ),
    "ho_3px_net6": (open_pair(3, 6), {}, ["circle", "rectangle"]),
    "ho_6px_net6": (open_pair(6, 6), {}, ["circle", "rectangle"]),
    "ho_10px_net6": (open_pair(10, 6), {}, ["circle", "rectangle"]),
    "ho_16px_net6": (open_pair(16, 6), {}, ["circle", "rectangle"]),
    "ho_10px_net3": (open_pair(10, 3), {}, ["circle", "rectangle"]),
    "nho_r8": ([S("circle", 100, 100, 8, 8), S("rectangle", 300, 100, 12, 7)], {}, ["circle", "rectangle"]),
    "nho_r5_khong_nham": ([S("circle", 100, 100, 5, 5), S("rectangle", 300, 100, 7, 5)], {}, ["rectangle"]),
    "sang_lech_60": (
        [S("circle", 150, 200, 70, 70), S("rectangle", 420, 200, 110, 60)],
        dict(light=0.6),
        ["circle", "rectangle"],
    ),
    "nhieu_30": (
        [S("circle", 150, 200, 70, 70), S("rectangle", 420, 200, 110, 60)],
        dict(noise=30),
        ["circle", "rectangle"],
    ),
    "tam_giac_ellipse": ([S("triangle", 150, 200, 70, 70), S("ellipse", 420, 200, 110, 45, 0.4)], {}, []),
    "luc_giac": ([hexm], {}, []),
    "tron_ve_tay_meo3": (
        [wobbly(150, 200, 75, 0.03, 3), wobbly(420, 200, 60, 0.03, 5)],
        {},
        ["circle", "circle"],
    ),
    "tron_ve_tay_meo5": (
        [wobbly(150, 200, 75, 0.05, 4), wobbly(420, 200, 60, 0.05, 6)],
        {},
        ["circle", "circle"],
    ),
    "tron_ve_tay_vien": (
        [
            wobbly(150, 200, 75, 0.04, 3) & ~wobbly(150, 200, 69, 0.04, 3),
            wobbly(420, 200, 60, 0.04, 6) & ~wobbly(420, 200, 54, 0.04, 6),
        ],
        {},
        ["circle", "circle"],
    ),
    "bat_giac": ([octa], {}, []),
    "cham_nhau": (
        [S("circle", 230, 200, 70, 70), S("rectangle", 370, 200, 72, 50)],
        {},
        ["circle", "rectangle"],
    ),
}
probes = {}
for name, (parts, kw, expect) in PROBES.items():
    res = r.geometry(scene(parts, **kw), previews=False)
    got = sorted(o["class"] for o in res["objects"] if o["target"])
    others = sorted(o["class"] for o in res["objects"] if not o["target"])
    probes[name] = {"ok": got == sorted(expect), "targets": got, "others": others}
out["shape_probes"] = probes
out["shape_probes_passed"] = f"{sum(p['ok'] for p in probes.values())}/{len(probes)}"

# ---------------------------------------------------------------- real dev/val only
if "--skip-real" not in sys.argv:
    manifest = read_real_manifest()
    by_role, per_case = {}, {}
    t0 = time.time()
    for case in manifest["cases"]:
        if case["split"] == "test":
            continue
        rgb = np.asarray(Image.open(real_image_path(case)).convert("RGB"))
        pred = r.geometry(rgb, previews=False)
        sc = detection_score(case["annotations"], pred["objects"])
        by_role.setdefault(case["split"], []).append(sc)
        per_case[case["id"]] = {k: sum(v[k] for v in sc[0].values()) for k in ("tp", "fp", "fn")}
        per_case[case["id"]]["shown_boxes"] = len(pred["objects"])
    out["real"] = {}
    for role, rows in by_role.items():
        s = summarize_detection(rows, len(rows), 0)
        out["real"][role] = {
            k: round(s[k], 4) if isinstance(s[k], float) else s[k]
            for k in ("tp", "fp", "fn", "precision", "recall", "f1")
        }
    out["real_per_case"] = per_case
    out["real_seconds"] = round(time.time() - t0, 1)

print(json.dumps({k: v for k, v in out.items() if k != "real_per_case"}, indent=1))
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(out, stream, indent=1, ensure_ascii=False)
