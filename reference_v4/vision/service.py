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
                 "ellipse": "Ellipse (ngoài phạm vi)", "background": "Nền / chi tiết phụ", "unknown": "Chưa xác định"}


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
        self.shape_classes = self.shape_metadata["classes"]
        try:
            self.profile = json.loads((ROOT / "data" / "geometry_profile.json").read_text(encoding="utf-8"))
            if self.profile["mode"] not in ("baseline_v2", "global_regions", "illumination_regions", "edge_regions", "contrast_regions", "repaired_regions"):
                raise ValueError("mode")
            if not 0 <= float(self.profile["min_confidence"]) <= 1:
                raise ValueError("confidence")
            self.profile["min_confidence"] = float(self.profile["min_confidence"])
            if "complex_min_confidence" in self.profile:
                self.profile["complex_min_confidence"] = float(self.profile["complex_min_confidence"])
                if not 0 <= self.profile["complex_min_confidence"] <= 1:
                    raise ValueError("complex confidence")
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

    def prepare_geometry(self, image, mode=None):
        """All actual normalized CNN candidates, available before inference."""
        mode = mode or self.profile["mode"]
        gray = grayscale(image)
        original_size = [gray.shape[1], gray.shape[0]]
        rgb = np.asarray(image)[..., :3] if np.asarray(image).ndim == 3 else None
        if np.asarray(image).ndim == 3 and np.asarray(image).shape[2] == 4:
            a = np.asarray(image).astype(np.float32)
            rgb = np.rint(a[..., :3] * (a[..., 3:4]/255) + 255*(1-a[..., 3:4]/255)).astype(np.uint8)
        if max(gray.shape) > 800:
            factor = 800 / max(gray.shape)
            gray = np.rint(resize_bilinear(gray, (max(1, round(gray.shape[0] * factor)), max(1, round(gray.shape[1] * factor))))).astype(np.uint8)
            if rgb is not None:
                rgb = np.rint(resize_bilinear(rgb, gray.shape)).astype(np.uint8)
        candidates, trace = region_proposals(gray, mode, self.profile, rgb=rgb)
        normalized = [normalize_shape(filled, fill=mode == "baseline_v2") for _, filled, _ in candidates]
        return {"gray": gray, "original_size": original_size, "candidates": candidates,
                "trace": trace, "normalized": normalized, "mode": mode}

    def geometry(self, image, previews=True, min_confidence=None, mode=None, prepared=None):
        prepared = prepared or self.prepare_geometry(image, mode)
        gray, original_size, candidates, trace, normalized, mode = (prepared[k] for k in
            ("gray", "original_size", "candidates", "trace", "normalized", "mode"))
        if min_confidence is None:
            min_confidence = self.profile["min_confidence"]
            if trace.get("background_border_std", 0) >= self.profile.get("illumination_border_std", 12):
                min_confidence = max(min_confidence, self.profile.get("complex_min_confidence", min_confidence))
        clean, mask, joined = trace["clean"], trace["mask"], trace["joined"]
        threshold, polarity = trace["threshold"], trace["polarity"]
        warnings = []
        probabilities = predict(self.shapes, normalized) if normalized else []
        selected, rejected = (list(range(len(candidates))), {}) if mode == "baseline_v2" else select_regions(
            candidates, probabilities, self.shape_classes, min_confidence, self.profile)
        objects, counts = [], {"circle": 0, "rectangle": 0}
        if trace["rejected"].get("candidate_limit") or rejected.get("output_limit"):
            warnings.append("Ảnh có quá nhiều vùng; đã giới hạn số vùng xử lý. Hãy thử nền đơn giản hơn.")
        for i in selected:
            component, filled, bank = candidates[i]
            probs = probabilities[i]
            label = self.shape_classes[int(probs.argmax())]
            confidence = float(probs.max())
            if confidence < min_confidence:
                label = "unknown"
            target = label in counts
            if target:
                counts[label] += 1
            item = {"id": len(objects) + 1, "class": label, "label": DISPLAY_NAMES[label], "confidence": confidence,
                    "target": target, "bbox": component.bbox, "features": geometric_features(filled) if previews else {"area": int(filled.sum())},
                    "region_source": bank,
                    "probabilities": dict(zip(self.shape_classes, probs.astype(float).tolist()))}
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
        result["candidate_diagnostics"] = [{"bbox": c.bbox, "region_source": bank,
            "class": self.shape_classes[int(probabilities[i].argmax())], "confidence": float(probabilities[i].max()),
            "retained": i in selected} for i, (c, _, bank) in enumerate(candidates)]
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
            retained_mask = np.zeros_like(gray, bool)
            for i in selected:
                c, filled, _ = candidates[i]
                retained_mask[c.y:c.y+c.height, c.x:c.x+c.width] |= filled
            result["pipeline_retained"] = png_data(retained_mask.astype(np.uint8)*255)
            def preview(mask):
                a = mask.astype(np.uint8)*255
                if max(a.shape) > 400:
                    scale = 400/max(a.shape)
                    a = resize_bilinear(a, (max(1, round(a.shape[0]*scale)), max(1, round(a.shape[1]*scale))))
                return png_data(a)
            result["pipeline_branches"] = [{"name": b["name"], "binary": preview(b["binary"]),
                "closed": preview(b["closed"]), "candidates": preview(b["candidates"])} for b in trace.get("branches", [])]
        return result
