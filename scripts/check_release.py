"""Read-only release check: offline assets, frozen identities and Flask routes.

This does not train, regenerate datasets, reopen the real test for inference,
or overwrite evaluation reports. JSON is written only to standard output.
"""

import base64
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from vision.datasets import file_sha256, prepare_shape_splits
from vision.provenance import asset_identity
from vision.real_data import audit_real_manifest, check_real_protocol
from vision.training_data import prepare_real_training


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_release():
    require(sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32,
            "Use Python 3.12 64-bit, matching requirements.txt and CI.")
    identities = asset_identity(ROOT)
    require("missing" not in identities.values(), "A required inference asset is missing.")
    report_checks = {}
    for name in ("evaluation.json", "real_evaluation.json", "segmentation_comparison.json"):
        report = json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        changed = [key for key, value in identities.items()
                   if report.get("asset_identity", {}).get(key) != value]
        require(not changed, f"Stale report {name}: {', '.join(changed)}")
        report_checks[name] = {"assets_checked": len(identities), "changed_items": changed}
    source = json.loads((ROOT / "data/source.json").read_text(encoding="utf-8"))
    require(file_sha256(ROOT / "data/mnist.npz") == source["sha256"], "MNIST checksum mismatch.")
    geometry = prepare_shape_splits(ROOT)
    hybrid = prepare_real_training(ROOT)
    real = audit_real_manifest(ROOT)
    protocol = check_real_protocol(ROOT)
    require(real["passed"], "Real image manifest audit failed.")
    require(protocol["matched_original_protocol"],
            f"Frozen protocol changed: {protocol['changed_items']}")

    routes = ("/", "/api/health", "/api/metrics", "/api/real", "/static/script.js",
              "/static/style.css", "/static/real.css", "/examples/digit_6.png",
              "/real/images/floppy_front.png", "/results/real_floppy_front_comparison.png")
    http = {}
    with app.test_client() as client:
        for route in routes:
            with client.get(route) as response:
                http[route] = response.status_code
                require(response.status_code == 200, f"{route}: HTTP {response.status_code}")
        predictions = {}
        for route, filename in (("/api/digits", "digit_6.png"), ("/api/shapes", "shapes_mixed.png")):
            payload = "data:image/png;base64," + base64.b64encode(
                (ROOT / "examples" / filename).read_bytes()).decode("ascii")
            with client.post(route, json={"image": payload}) as response:
                require(response.status_code == 200, f"{route}: HTTP {response.status_code}")
                result = response.get_json()
                if route == "/api/digits":
                    require(result["prediction"] == 6, "Bundled digit example failed.")
                    predictions[route] = {"prediction": result["prediction"]}
                else:
                    require(sum(result["counts"].values()) > 0, "Bundled geometry example failed.")
                    predictions[route] = {"counts": result["counts"]}
    return {
        "status": "ok",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "Flask", "Pillow")},
        "frozen_reports": report_checks,
        "geometry_split_counts": geometry["requested_counts"],
        "hybrid_training_samples": hybrid["train_samples"],
        "hybrid_code_changed_since_build": hybrid.get("code_changed_since_build", []),
        "real_images": real["case_count"],
        "source_groups": real["original_photo_groups"],
        "real_targets": sum(real["target_counts"].values()),
        "protocol_changed_items": protocol["changed_items"],
        "real_test_independent": protocol["lock"]["test_independent"],
        "http_get": http,
        "example_predictions": predictions,
        "scope": "Read-only smoke checks; stored benchmark scores are not remeasured.",
    }


if __name__ == "__main__":
    try:
        result = check_release()
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
