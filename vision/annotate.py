"""Vẽ ảnh kết quả nhận dạng hình học (chỉ để hiển thị, không ảnh hưởng tới nhãn).

Quy ước màu, dùng chung cho ảnh kết quả và danh sách đối tượng trên giao diện:
  - xanh lá: hình mục tiêu đã xác định (tròn, chữ nhật);
  - cam: hình đã xác định nhưng ngoài hai lớp mục tiêu (tam giác, ellipse, đa giác);
  - đỏ: vùng chưa xác định (điểm CNN dưới ngưỡng hoặc vùng quá nhỏ).
Mỗi vùng được tô bán trong suốt bằng đúng mặt nạ của nó, viền khung và chữ nhãn cùng màu.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {"target": "#008768", "other": "#ce7820", "unknown": "#d63a3a"}
FILL_ALPHA = {"target": 0.8, "other": 0.8, "unknown": 0.45}
TINT_LIGHTEN = 0.72  # điểm ảnh sáng thành màu nhạt, điểm ảnh tối thành màu đậm
BOX_WIDTH = {"target": 3, "other": 3, "unknown": 2}


def category(item):
    if item["class"] == "unknown":
        return "unknown"
    return "target" if item["target"] else "other"


def _rgb(hex_color):
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], np.float32)


def _font(size=13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 không có tham số size
        return ImageFont.load_default()


def label_text(item):
    # Điểm softmax của vùng chưa xác định không phải độ tin cậy của nhãn nào, nên không in ra.
    if item["class"] == "unknown":
        return f"{item['id']}: unknown"
    return f"{item['id']}: {item['class']} {item['confidence']:.0%}"


def draw_annotations(gray, objects, masks, include_unknown=True):
    """gray: ảnh xám HxW; objects: danh sách đối tượng; masks: mặt nạ bool theo bbox của từng đối tượng.

    include_unknown=False cho ảnh gọn chỉ có vùng đã xác định (giao diện cho phép bật/tắt).
    """
    canvas = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    # Thứ tự vẽ: vùng chưa xác định trước (lớn trước, nhỏ sau) rồi mới tới vùng đã xác định,
    # để vùng quan trọng và vùng nhỏ không bị vùng lớn che mất.
    keep = [i for i in range(len(objects)) if include_unknown or category(objects[i]) != "unknown"]
    order = sorted(
        keep,
        key=lambda i: (category(objects[i]) != "unknown", -objects[i]["bbox"][2] * objects[i]["bbox"][3]),
    )
    for i in order:
        item, mask = objects[i], masks[i]
        x, y, w, h = item["bbox"]
        kind = category(item)
        alpha = FILL_ALPHA[kind]
        region = canvas[y : y + h, x : x + w]
        m = mask[: region.shape[0], : region.shape[1]]
        # Nhuộm màu giữ cấu trúc: nét tối -> màu đậm, nền sáng bên trong -> màu nhạt; nhờ vậy
        # cả hình đặc tối lẫn hình chỉ có đường viền đều thấy rõ màu của vùng.
        color = _rgb(COLORS[kind])
        pixels = region[m]
        tint = color + (255 - color) * TINT_LIGHTEN * (pixels[:, :1] / 255.0)
        region[m] = pixels * (1 - alpha) + tint * alpha
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = _font()
    height, width = gray.shape
    for i in order:
        item = objects[i]
        x, y, w, h = item["bbox"]
        kind = category(item)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=COLORS[kind], width=BOX_WIDTH[kind])
    # Nhãn vẽ sau cùng để không bị khung của vùng khác đè lên.
    for i in order:
        item = objects[i]
        x, y, w, h = item["bbox"]
        color = COLORS[category(item)]
        text = label_text(item)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw, th = right - left + 6, bottom - top + 4
        tx = min(max(0, x), max(0, width - tw))
        ty = y - th - 1 if y - th - 1 >= 0 else min(y + h + 1, height - th)
        draw.rectangle((tx, ty, tx + tw, ty + th), fill="white", outline=color)
        draw.text((tx + 3 - left, ty + 2 - top), text, fill=color, font=font)
    return np.asarray(image)


# --------------------------------------------------------------------------
# Chế độ chữ số: một chữ số mỗi ảnh
# --------------------------------------------------------------------------
DIGIT_DISPLAY_MAX = 480  # ảnh chụp lớn được thu nhỏ để nhãn còn đọc được
DIGIT_ALPHA = 0.85  # vùng tô là chính nét chữ nên tô gần như đặc


def _tint_mask(canvas, mask, hex_color, alpha=DIGIT_ALPHA):
    color = _rgb(hex_color)
    canvas[mask] = canvas[mask] * (1 - alpha) + color * alpha


def draw_digit_annotation(gray, support, ignored, bbox, prediction, confidence, confident, merged_parts=1):
    """Tô nét được đưa vào CNN (xanh nếu chắc chắn, đỏ nếu dưới ngưỡng) và vết nhỏ bị bỏ qua (đỏ).

    gray: ảnh xám HxW; support: mặt nạ bool HxW của nét được đọc; ignored: các Component bị bỏ qua;
    bbox: (x, y, w, h) của vùng được đọc.
    """
    canvas = np.repeat(np.asarray(gray)[..., None], 3, axis=2).astype(np.float32)
    read_color = COLORS["target"] if confident else COLORS["unknown"]
    for c in ignored:
        region = canvas[c.y : c.y + c.height, c.x : c.x + c.width]
        _tint_mask(region, c.mask, COLORS["unknown"])
    _tint_mask(canvas, np.asarray(support, bool), read_color)
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    height, width = canvas.shape[:2]
    scale = min(1.0, DIGIT_DISPLAY_MAX / max(height, width))
    if scale < 1:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    # Ảnh vẽ tay 280 px thì giữ nguyên; ảnh nhỏ như MNIST 28 px được phóng to để vẽ được nhãn.
    if max(image.size) < 240:
        up = 240 / max(image.size)
        image = image.resize((round(image.size[0] * up), round(image.size[1] * up)), Image.NEAREST)
        scale *= up
    draw = ImageDraw.Draw(image)
    font = _font(14)
    W, H = image.size

    def box(b, color, width_px):
        x, y, w, h = (v * scale for v in b)
        draw.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), outline=color, width=width_px)
        return x, y, w, h

    ignored_boxes = 0
    for c in sorted(ignored, key=lambda c: -c.area):
        # Vết rất nhỏ (nhiễu muối tiêu) chỉ được tô màu; vết đủ lớn mới vẽ khung để ảnh không rối.
        if min(c.width, c.height) * scale < 6 or ignored_boxes >= 6:
            continue
        box((c.x, c.y, c.width, c.height), COLORS["unknown"], 1)
        ignored_boxes += 1
    x, y, w, h = box(bbox, read_color, 3)
    text = f"{prediction}  {confidence:.0%}" + ("" if confident else " (uncertain)")
    if merged_parts > 1:
        text += f"  [{merged_parts} parts merged]"
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    tw, th = right - left + 8, bottom - top + 6
    tx = min(max(0, x - 2), max(0, W - tw))
    ty = y - th - 4 if y - th - 4 >= 0 else min(y + h + 4, H - th)
    draw.rectangle((tx, ty, tx + tw, ty + th), fill="white", outline=read_color)
    draw.text((tx + 4 - left, ty + 3 - top), text, fill=read_color, font=font)
    return np.asarray(image)
