"""Local Flask demo; no dataset download or training occurs on application startup."""

import argparse
import base64
import binascii
import json
import warnings
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from vision.service import Recognizer
from vision.models import ModelLoadError
from vision.provenance import asset_identity

ROOT = Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS = 12_000_000
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
_recognizer = None
_recognizer_lock = threading.Lock()


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        with _recognizer_lock:
            if _recognizer is None:
                _recognizer = Recognizer()
    if asset_identity() != _recognizer.asset_identity:
        raise ModelLoadError(
            "Mô hình hoặc cấu hình đã đổi trên đĩa. Khởi động lại ứng dụng để nạp đúng phiên bản."
        )
    return _recognizer


def decode_image(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Thiếu ảnh. Hãy vẽ hoặc chọn một tệp ảnh.")
    if "," in value:
        header, value = value.split(",", 1)
        if not header.startswith("data:image/") or ";base64" not in header:
            raise ValueError("Ảnh phải được mã hóa base64.")
    try:
        raw = base64.b64decode(value, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                if image.format not in ("PNG", "JPEG", "WEBP", "BMP"):
                    raise ValueError("Chỉ hỗ trợ PNG, JPG, WebP và BMP.")
                if max(image.size) > 4096 or image.width * image.height > 12_000_000:
                    raise ValueError(
                        "Ảnh quá lớn. Hãy dùng ảnh có mỗi cạnh tối đa 4096 pixel, không quá 12 megapixel."
                    )
                orientation = image.getexif().get(274, 1)
                # RGBA conversion is file decoding; grayscale/alpha math is NumPy.
                a = np.asarray(image.convert("RGBA"))
                if orientation == 2:
                    a = a[:, ::-1]
                elif orientation == 3:
                    a = a[::-1, ::-1]
                elif orientation == 4:
                    a = a[::-1]
                elif orientation == 5:
                    a = np.swapaxes(a, 0, 1)
                elif orientation == 6:
                    a = np.rot90(a, 3)
                elif orientation == 7:
                    a = np.swapaxes(a, 0, 1)[::-1, ::-1]
                elif orientation == 8:
                    a = np.rot90(a)
                return np.ascontiguousarray(a)
    except (
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Không đọc được ảnh. Hãy chọn một tệp PNG hoặc JPG hợp lệ.") from exc


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    try:
        r = get_recognizer()
        return jsonify({"status": "ok", "digits": r.digit_metadata, "shapes": r.shape_metadata})
    except (FileNotFoundError, RuntimeError) as exc:
        return jsonify({"status": "error", "error": str(exc)}), 503


def image_payload():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Cần gửi JSON chứa trường image.")
    return decode_image(payload.get("image"))


@app.post("/predict")
@app.post("/api/digits")
def digits():
    try:
        image = image_payload()
        return jsonify(get_recognizer().digit(image))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (FileNotFoundError, ModelLoadError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/shapes")
def shapes():
    try:
        image = image_payload()
        return jsonify(get_recognizer().geometry(image))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (FileNotFoundError, ModelLoadError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/api/metrics")
def metrics():
    path = ROOT / "results" / "evaluation.json"
    if not path.exists():
        return jsonify({"error": "Chưa có số đo. Chạy python scripts/evaluate.py."}), 404
    return report_response(path)


@app.get("/examples/<path:name>")
def examples(name):
    return send_from_directory(ROOT / "examples", name)


@app.get("/api/real")
def real_metrics():
    path = ROOT / "results" / "real_evaluation.json"
    if not path.exists():
        return jsonify({"error": "Chưa có số đo ảnh thực. Chạy python scripts/evaluate_real.py."}), 404
    return report_response(path)


def report_response(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Report must be an object")
        identity = result.get("asset_identity", {})
        if not isinstance(identity, dict):
            raise ValueError("Invalid report identity")
        required = (
            {"digits": dict, "shapes": dict}
            if path.name == "evaluation.json"
            else {"cases": list, "audit": dict, "protocol": dict, "detection_by_role": dict}
        )
        if any(not isinstance(result.get(key), kind) for key, kind in required.items()):
            raise ValueError("Invalid report structure")
    except (OSError, ValueError) as exc:
        return jsonify({"error": "Không đọc được kết quả đánh giá. Chạy lại scripts/evaluate.py."}), 503
    current = asset_identity()
    try:
        loaded = get_recognizer().asset_identity
    except (FileNotFoundError, ModelLoadError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 503
    stale = [key for key in current if identity.get(key) != current[key]]
    if current != loaded or stale:
        return (
            jsonify(
                {
                    "error": "Kết quả đã cũ so với mô hình hoặc pipeline. Chạy lại scripts/evaluate.py và khởi động lại ứng dụng.",
                    "stale": True,
                    "changed_assets": stale,
                    "restart_required": current != loaded,
                }
            ),
            409,
        )
    response = jsonify(result)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/real/<path:name>")
def real_asset(name):
    return send_from_directory(ROOT / "data" / "real", name)


@app.get("/results/<path:name>")
def result_file(name):
    return send_from_directory(ROOT / "results", name)


@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "Dữ liệu gửi lên quá lớn (tối đa 8 MB)."}), 413


@app.errorhandler(HTTPException)
def http_error(error):
    return jsonify({"error": error.description}), error.code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
