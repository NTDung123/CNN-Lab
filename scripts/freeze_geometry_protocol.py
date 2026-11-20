"""Freeze a development/validation-selected pipeline BEFORE new real test access.

This command cannot overwrite a protocol once phase-3 test has been accessed.
Keep seen test photos as development in a later experiment and collect new test.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.datasets import file_sha256
from vision.provenance import asset_identity
from vision.real_data import audit_real_manifest, read_real_manifest


def freeze(amend=None):
    """Khóa cấu hình. amend: lý do sửa đổi bản khóa của CHÍNH giai đoạn này sau khi test đã mở.

    Sửa đổi chỉ dành cho thay đổi không xuất phát từ kết quả test (ví dụ lỗi tìm thấy khi
    kiểm tra giao diện). Mọi sửa đổi được ghi lại kèm lý do, thời điểm và tệp đã đổi.
    """
    real = ROOT / "data/real"
    from vision.real_data import PROTOCOL_PHASE, test_access_phase

    previously_opened = test_access_phase(ROOT)
    old_lock = (
        json.loads((real / "protocol_lock.json").read_text())
        if (real / "protocol_lock.json").exists()
        else {}
    )
    if previously_opened >= PROTOCOL_PHASE and not amend:
        raise ValueError(
            "Test đã được mở. Không được ghi lại cấu hình rồi tiếp tục coi cùng ảnh là test chưa thấy."
            ' Nếu thay đổi không xuất phát từ kết quả test, dùng --amend "lý do" để ghi lại sửa đổi.'
        )
    if amend:
        if old_lock.get("protocol_version") != PROTOCOL_PHASE:
            raise ValueError("Chỉ sửa đổi được bản khóa của giai đoạn hiện tại.")
        previously_opened = old_lock.get("test_previously_seen_in_phase") or 0
    audit_real_manifest(ROOT)
    manifest = read_real_manifest(ROOT)
    labels = json.loads((real / "annotation_freeze.json").read_text())
    if labels["manifest_sha256"] != file_sha256(real / "manifest.json"):
        raise ValueError("Nhãn/vai trò dữ liệu đã đổi sau khi khóa nhãn.")
    selection = json.loads((real / "selection_experiment.json").read_text())
    if selection["asset_identity"] != asset_identity(ROOT):
        raise ValueError(
            "Phải chạy lại development/validation bằng đúng mã và cấu hình hiện tại trước khi khóa."
        )
    mode = json.loads((ROOT / "data/geometry_profile.json").read_text())["mode"]
    expected = {c["id"] for c in manifest["cases"] if c["split"] == "validation"}
    for result in selection["by_mode"].values():
        if {c["id"] for c in result["cases"]} != expected or any(
            c["split"] == "test" for c in result["cases"]
        ):
            raise ValueError("Pipeline selection experiment must contain the complete validation set only.")
    if mode not in [v["mode"] for v in selection["by_mode"].values()]:
        raise ValueError("Selected mode not evaluated on development/validation.")
    files = {
        "manifest": "data/real/manifest.json",
        "shapes_checkpoint": "weights/shapes_cnn.pt",
        "geometry_dataset": "data/geometry_splits.npz",
        "processing": "vision/processing.py",
        "service": "vision/service.py",
        "segmentation": "vision/segmentation.py",
        "geometry_profile": "data/geometry_profile.json",
        "metrics": "vision/metrics.py",
        "real_evaluation": "vision/real_evaluation.py",
        "selection_experiment": "data/real/selection_experiment.json",
    }
    lock = {name + "_sha256": file_sha256(ROOT / path) for name, path in files.items()}
    from vision.models import load_cnn

    _, metadata = load_cnn("shapes", ROOT)
    if metadata.get("real_training_archive_sha256") != file_sha256(ROOT / "data/real_training.npz"):
        raise ValueError("CNN does not match the audited phase-4 training inputs.")
    lock.update(
        {
            "protocol_version": PROTOCOL_PHASE,
            # Test ảnh thực đã được mở ở giai đoạn trước: kết quả test giai đoạn này KHÔNG độc lập.
            "test_previously_seen_in_phase": previously_opened or None,
            "test_independent": previously_opened == 0,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_mode": mode,
            "training_roles": ["development"],
            "selection_roles": ["validation"],
            "test_previously_used_for_selection": False,
            "selection_rule": "Maximize validation target macro-F1 across two target classes, including negative-frame FP. CNN epochs selected on validation with synthetic validation guard; no new test used.",
            "notes": (
                "Roles/manual labels frozen first. Previously observed phase-3 test sources are development. "
                "Do not modify pipeline after reading new test results."
                + (
                    f" The real test photos were already opened in phase {previously_opened}; their phase-{previously_opened}"
                    " summary scores were known while phase-5 bugs were fixed using synthetic probes and"
                    " development/validation only. No test photo was run during phase-5 development, but the"
                    " test is no longer an unseen benchmark."
                    if previously_opened
                    else ""
                )
            ),
        }
    )
    if amend:
        changed = sorted(k for k, v in lock.items() if k.endswith("_sha256") and old_lock.get(k) != v)
        lock["frozen_at_utc"] = old_lock.get("frozen_at_utc", lock["frozen_at_utc"])
        lock["amendments"] = old_lock.get("amendments", []) + [
            {
                "amended_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": amend,
                "changed_items": changed,
                "previous_lock_sha256": file_sha256(real / "protocol_lock.json"),
                "test_already_opened_in_this_phase": True,
            }
        ]
    (real / "protocol_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amend", help="Lý do sửa đổi bản khóa của giai đoạn hiện tại (không dựa trên kết quả test)"
    )
    freeze(parser.parse_args().amend)
