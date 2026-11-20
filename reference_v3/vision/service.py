"""Inference pipelines shared by the web app and end-to-end evaluation."""
import base64
import json
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageDraw

from .models import SHAPE_CLASSES, first_feature_maps, load_cnn, predict
from .models import ROOT, ModelLoadError
from .provenance import asset_identity
from .segmentation import region_proposals, select_regions
from .processing import (closing, connected_components, foreground, geometric_features,
                         grayscale, normalize_shape, preprocess_digit, resize_bilinear, sobel)

DISPLAY_NAMES = {"circle": "Hình tròn", "rectangle": "Hình chữ nhật", "triangle": "Tam giác (ngoài phạm vi)",
                 "ellipse": "Ellipse (ngoài phạm vi)", "unknown": "Chưa xác định"}


def png_data(image):
    a = np.clip(image, 0, 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(a).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class Recognizer:
    def __init__(self, threads=2):
        torch.set_num_threads(threads)
        self.digits, self.digit_metadata = load_cnn("digits")
        self.shapes, self.shape_metadata = load_cnn("shapes")
        try:
            self.profile = json.loads((ROOT / "data" / "geometry_profile.json").read_text(encoding="utf-8"))
            if self.profile["mode"] not in ("baseline_v2", "global_regions", "illumination_regions", "edge_regions"):
                raise ValueError("mode")
            if not 0 <= float(self.profile["min_confidence"]) <= 1:
                raise ValueError("confidence")
            self.profile["min_confidence"] = float(self.profile["min_confidence"])
            if int(self.profile["background_window"]) % 2 != 1 or int(self.profile["background_window"]) < 1:
                raise ValueError("window")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ModelLoadError("Cấu hình geometry_profile.json không hợp lệ. Khôi phục cấu hình của bản chạy thử.") from exc
        self.asset_identity = asset_identity()

    def digit(self, image, previews=True):
        normalized, trace = preprocess_digit(image)
        probs = predict(self.digits, normalized)[0]
        prediction = int(probs.argmax())
        warnings = []
        if trace["components"] > 1:
            warnings.append("Có nhiều vùng nét tách rời; chế độ này chỉ phân loại MỘT chữ số, không đọc chuỗi số.")
        if probs[prediction] < .75:
            warnings.append("Kết quả chưa chắc chắn. Hãy viết chữ số lớn, rõ và ở giữa khung.")
        result = {"prediction": prediction, "confidence": float(probs[prediction]),
                  "probabilities": probs.astype(float).tolist(), "bbox": trace["bbox"],
                  "threshold": trace["threshold"], "polarity": trace["polarity"], "warnings": warnings,
                  "model": "CNN MNIST", "confidence_note": "Softmax là điểm mô hình, chưa được hiệu chỉnh xác suất."}
        if previews:
            result["pipeline"] = {key: png_data(trace[key]) for key in ("gray", "binary", "normalized")}
            result["feature_maps"] = [png_data(m) for m in first_feature_maps(self.digits, normalized)]
        return result

    def geometry(self, image, previews=True, min_confidence=None, mode=None):
        min_confidence = self.profile["min_confidence"] if min_confidence is None else min_confidence
        mode = mode or self.profile["mode"]
        gray = grayscale(image)
        original_size = [gray.shape[1], gray.shape[0]]
        if max(gray.shape) > 800:
            factor = 800 / max(gray.shape)
            gray = np.rint(resize_bilinear(gray, (max(1, round(gray.shape[0] * factor)), max(1, round(gray.shape[1] * factor))))).astype(np.uint8)
        candidates, trace = region_proposals(gray, mode, self.profile)
        clean, mask, joined = trace["clean"], trace["mask"], trace["joined"]
        threshold, polarity = trace["threshold"], trace["polarity"]
        warnings = []
        normalized = [normalize_shape(filled, fill=mode == "baseline_v2") for _, filled, _ in candidates]
        probabilities = predict(self.shapes, normalized) if normalized else []
        selected, rejected = (list(range(len(candidates))), {}) if mode == "baseline_v2" else select_regions(
            candidates, probabilities, SHAPE_CLASSES, min_confidence, self.profile)
        objects, counts = [], {"circle": 0, "rectangle": 0}
        if trace["rejected"].get("candidate_limit") or rejected.get("output_limit"):
            warnings.append("Ảnh có quá nhiều vùng; đã giới hạn số vùng xử lý. Hãy thử nền đơn giản hơn.")
        for i in selected:
            component, filled, bank = candidates[i]
            probs = probabilities[i]
            label = SHAPE_CLASSES[int(probs.argmax())]
            confidence = float(probs.max())
            if confidence < min_confidence:
                label = "unknown"
            target = label in counts
            if target:
                counts[label] += 1
            item = {"id": len(objects) + 1, "class": label, "label": DISPLAY_NAMES[label], "confidence": confidence,
                    "target": target, "bbox": component.bbox, "features": geometric_features(component.mask),
                    "region_source": bank,
                    "probabilities": dict(zip(SHAPE_CLASSES, probs.astype(float).tolist()))}
            if previews:
                item["crop"] = png_data(normalized[i] * 255)
            objects.append(item)
        if not objects:
            warnings.append("Không tìm thấy vùng hình đủ lớn; thử nền đồng nhất và nét vẽ đậm hơn.")
        if any(o["bbox"][0] == 0 or o["bbox"][1] == 0 or
               o["bbox"][0] + o["bbox"][2] == gray.shape[1] or
               o["bbox"][1] + o["bbox"][3] == gray.shape[0] for o in objects):
            warnings.append("Có hình chạm mép ảnh; hình bị cắt có thể được nhận dạng sai.")
        result = {"objects": objects, "counts": counts, "threshold": threshold, "polarity": polarity,
                  "processed_size": [gray.shape[1], gray.shape[0]], "original_size": original_size,
                  "warnings": warnings, "model": "NumPy segmentation + shape CNN", "min_confidence": min_confidence,
                  "bbox_coordinates": "processed_image_pixels_xywh",
                  "segmentation": {"mode": mode, "banks": trace["banks"], "candidate_count": trace["candidate_count"],
                                   "background_border_std": trace.get("background_border_std"),
                                   "retained_count": len(objects), "rejected": {**trace["rejected"], **rejected}},
                  "confidence_note": "Softmax là điểm mô hình, chưa được hiệu chỉnh xác suất."}
        if previews:
            annotated = Image.fromarray(gray).convert("RGB")
            draw = ImageDraw.Draw(annotated)
            for item in objects:
                x, y, w, h = item["bbox"]
                color = "#008768" if item["target"] else "#ce7820"
                draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=3)
                text = f"{item['id']}: {item['class']} {item['confidence']:.0%}"
                ty = max(0, y - 15)
                draw.rectangle((x, ty, min(gray.shape[1], x + len(text) * 7), ty + 14), fill="white")
                draw.text((x + 2, ty), text, fill=color)
            result["annotated"] = png_data(np.asarray(annotated))
            result["pipeline"] = {"gray": png_data(gray), "denoised": png_data(clean),
                                  "binary": png_data(mask.astype(np.uint8) * 255),
                                  "closed": png_data(joined.astype(np.uint8) * 255), "edges": png_data(sobel(clean))}
        return result
