"""NumPy classification and class-aware IoU detection metrics."""
import numpy as np


def classification_metrics(labels, predictions, classes):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(cm, (labels, predictions), 1)
    rows = []
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        precision = tp / max(int(cm[:, i].sum()), 1)
        recall = tp / max(int(cm[i].sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": int(cm[i].sum())})
    return {"samples": len(labels), "correct": int(np.trace(cm)), "accuracy": float(np.trace(cm) / len(labels)),
            "macro_f1": float(np.mean([r["f1"] for r in rows])), "per_class": rows, "confusion_matrix": cm.tolist()}



def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    return overlap / max(a[2] * a[3] + b[2] * b[3] - overlap, 1)



def match_detections(truth, objects, threshold=.5):
    """One prediction and one GT can occur in at most one class-aware match."""
    matched, matches = set(), []
    pred = sorted((i for i, o in enumerate(objects) if o["target"]),
                  key=lambda i: (-objects[i]["confidence"], i))
    for pi in pred:
        p = objects[pi]
        candidates = [(iou(p["bbox"], g["bbox"]), -i, i) for i, g in enumerate(truth)
                      if i not in matched and p["class"] == g["class"]]
        overlap, _, index = max(candidates, default=(0, 0, -1))
        if overlap >= threshold:
            matched.add(index)
            matches.append({"prediction_index": pi, "truth_index": index, "iou": overlap})
    return matches


def detection_score(truth, objects):
    matches = match_detections(truth, objects)
    matched = {m["truth_index"] for m in matches}
    predicted = {m["prediction_index"] for m in matches}
    counts = {}
    for name in ("circle", "rectangle"):
        tp = sum(objects[m["prediction_index"]]["class"] == name for m in matches)
        fp = sum(o["class"] == name and o["target"] and i not in predicted for i, o in enumerate(objects))
        fn = sum(g["class"] == name and i not in matched for i, g in enumerate(truth))
        counts[name] = {"tp": tp, "fp": fp, "fn": fn}
    return counts, [m["iou"] for m in matches]



def summarize_detection(rows, scene_count, all_objects, confidence_threshold=.75):
    output = {"scenes": scene_count, "ground_truth_all_classes": all_objects,
              "iou_threshold": .5, "confidence_threshold": confidence_threshold, "per_class": {}}
    total_tp, total_fp, total_fn, overlaps = 0, 0, 0, []
    for name in ("circle", "rectangle"):
        tp = sum(r[0][name]["tp"] for r in rows)
        fp = sum(r[0][name]["fp"] for r in rows)
        fn = sum(r[0][name]["fn"] for r in rows)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        output["per_class"][name] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
                                     "f1": 2 * p * r / max(p + r, 1e-12)}
        total_tp += tp
        total_fp += fp
        total_fn += fn
    for row in rows:
        overlaps.extend(row[1])
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    output.update({"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall,
                   "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                   "mean_iou_on_true_positives": float(np.mean(overlaps)) if overlaps else 0.,
                   "ground_truth_targets": total_tp + total_fn})
    return output

