"""Shared scikit-learn/TorchMetrics metrics and missed-defect-first selection."""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torchmetrics.functional.classification import binary_jaccard_index, binary_f1_score


def image_metrics(labels, predictions) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "Accuracy": float(accuracy_score(labels, predictions)),
            "Precision": float(precision_score(labels, predictions, zero_division=0)),
            "Recall": float(recall_score(labels, predictions, zero_division=0)),
            "F1": float(f1_score(labels, predictions, zero_division=0)),
            "miss_rate": float(fn / (tp + fn)) if tp + fn else 0.0,
            "overkill_rate": float(fp / (tn + fp)) if tn + fp else 0.0}


def candidates(scores) -> np.ndarray:
    values = np.unique(scores)
    return np.append(values, np.nextafter(values[-1], np.inf))


def select_image_threshold(scores, labels) -> tuple[float, dict]:
    results = [(float(t), image_metrics(labels, scores >= t)) for t in candidates(scores)]
    return min(results, key=lambda item: (item[1]["FN"], item[1]["FP"], -item[0]))


def select_pixel_threshold(maps, gt_masks) -> float:
    """Highest threshold retaining a GT-overlapping region on each dev NG.

    Localization FN is an NG image with no overlap, consistent with the
    project's case-level missed-defect priority. Among zero-miss thresholds,
    this minimizes background FP pixels. Pixel FN and Dice are also reported;
    zero localization misses does not mean every GT pixel is covered.
    """
    peaks = [float(anomaly_map[truth].max()) for anomaly_map, truth in zip(maps, gt_masks, strict=True)
             if truth.any()]
    if not peaks:
        raise ValueError("Pixel calibration needs dev defect masks")
    return min(peaks)


def localization_metrics(masks, gt_masks, labels) -> dict:
    ious, dices = [], []
    misses, fp_pixels, fn_pixels = 0, 0, 0
    for predicted, truth, label in zip(masks, gt_masks, labels, strict=True):
        fp_pixels += int(np.count_nonzero(predicted & ~truth))
        fn_pixels += int(np.count_nonzero(~predicted & truth))
        if label:
            misses += int(not np.any(predicted & truth))
            prediction = torch.from_numpy(predicted.astype(np.uint8))
            target = torch.from_numpy(truth.astype(np.uint8))
            ious.append(float(binary_jaccard_index(prediction, target)))
            dices.append(float(binary_f1_score(prediction, target)))
    return {"IoU": float(np.mean(ious)), "Dice": float(np.mean(dices)),
            "localization_FN_images": misses, "pixel_FN": fn_pixels, "pixel_FP": fp_pixels,
            "pixel_FP_rate_background": fp_pixels / sum(int(np.count_nonzero(~m)) for m in gt_masks),
            "mask_metric_population": f"{int(np.count_nonzero(labels))} dev NG; normal backgrounds included in pixel_FP"}


def joint_or_thresholds(ea_scores, pc_scores, labels) -> tuple[dict, dict]:
    best = None
    for ea in candidates(ea_scores):
        ea_flags = ea_scores >= ea
        for pc in candidates(pc_scores):
            pc_flags = pc_scores >= pc
            metrics = image_metrics(labels, ea_flags | pc_flags)
            hits_ea = int(np.count_nonzero(ea_flags & labels.astype(bool)))
            hits_pc = int(np.count_nonzero(pc_flags & labels.astype(bool)))
            key = (metrics["FN"], metrics["FP"], -min(hits_ea, hits_pc), -hits_ea - hits_pc, -float(ea), -float(pc))
            if best is None or key < best[0]:
                best = (key, {"efficientad": float(ea), "patchcore": float(pc)}, metrics)
    return best[1], best[2]
