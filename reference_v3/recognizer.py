"""Load original phase-3 geometry behavior, without its unrelated web startup."""
import json
from pathlib import Path
import torch
from .vision.service import Recognizer
from .vision.models import load_cnn

def legacy_recognizer():
    root=Path(__file__).resolve().parent
    torch.set_num_threads(2)
    r=Recognizer.__new__(Recognizer)
    r.shapes,r.shape_metadata=load_cnn("shapes",root)
    r.profile=json.loads((root/"data/geometry_profile.json").read_text())
    r.asset_identity={}
    r.shape_classes=r.shape_metadata["classes"]
    return r
