"""Evaluation freshness: do not show old scores for changed inference assets."""
from .datasets import ROOT, file_sha256

ASSETS = {"digits_checkpoint": "weights/digits_cnn.pt", "shapes_checkpoint": "weights/shapes_cnn.pt",
          "processing": "vision/processing.py", "segmentation": "vision/segmentation.py",
          "service": "vision/service.py", "models": "vision/models.py", "metrics": "vision/metrics.py",
          "real_evaluation": "vision/real_evaluation.py", "real_manifest": "data/real/manifest.json",
          "geometry_profile": "data/geometry_profile.json"}


def asset_identity(root=ROOT):
    return {key: file_sha256(root / path) for key, path in ASSETS.items()}
