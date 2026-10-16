# 開發階段的偵測與定位評估工具，保留用來理解門檻採用依據。
# label=1 表示異常，label=0 表示正常；分數大於等於門檻時預測為異常。
# 這些選門檻函式只應讀 dev 資料；新圖片推論只讀已採用的門檻，不重新最佳化。

"""Shared scikit-learn/TorchMetrics metrics and missed-defect-first selection."""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from torchmetrics.functional.classification import binary_jaccard_index, binary_f1_score


# 由正常／異常標籤與布林預測計算 TP、FN、FP、TN 及常用二元指標。
# 漏檢率的分母為真 NG，過殺率的分母為真正常；分母為零時回傳 0。
def image_metrics(labels, predictions) -> dict:
    # 固定 labels=[0,1]，即使某次母體缺一類仍保持混淆矩陣四格的順序。
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "Accuracy": float(accuracy_score(labels, predictions)),
            "Precision": float(precision_score(labels, predictions, zero_division=0)),
            "Recall": float(recall_score(labels, predictions, zero_division=0)),
            "F1": float(f1_score(labels, predictions, zero_division=0)),
            "miss_rate": float(fn / (tp + fn)) if tp + fn else 0.0,
            "overkill_rate": float(fp / (tn + fp)) if tn + fp else 0.0}


# 取所有不同分數作為 >= 判定的候選門檻，另加略高於最大值的一點。
# 最後一點代表全判正常；相同分數會一同跨越門檻，不被任意拆開。
def candidates(scores) -> np.ndarray:
    values = np.unique(scores)
    return np.append(values, np.nextafter(values[-1], np.inf))


# 遍歷單模型影像門檻，先最小化 FN，再最小化 FP。
# 同樣 FN／FP 時選較高門檻；回傳所選門檻及該操作點的指標。
def select_image_threshold(scores, labels) -> tuple[float, dict]:
    results = [(float(t), image_metrics(labels, scores >= t)) for t in candidates(scores)]
    return min(results, key=lambda item: (item[1]["FN"], item[1]["FP"], -item[0]))


# 對每張含 GT 瑕疵的 dev 圖，取瑕疵區內的最高分，再取所有圖片的最小峰值。
# 在 >= 規則下保留每張圖至少一處 GT 重疊；這不代表每個瑕疵區或全部 GT 像素都被覆蓋。
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


# 逐張累計像素 FP／FN，並在真 NG 圖上計算 IoU 與 Dice 平均。
# 同時記錄完全沒碰到 GT 的 NG 張數；背景 FPR 也納入正常圖的背景像素。
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


# 搜尋 EA 與 PC 的影像門檻組合，對 OR 判決依序最小化 FN、FP。
# 同分時優先平衡兩個模型的真異常命中數，再比較總命中及較高門檻，回傳兩門檻與 OR 指標。
def joint_or_thresholds(ea_scores, pc_scores, labels) -> tuple[dict, dict]:
    best = None
    for ea in candidates(ea_scores):
        ea_flags = ea_scores >= ea
        for pc in candidates(pc_scores):
            pc_flags = pc_scores >= pc
            metrics = image_metrics(labels, ea_flags | pc_flags)
            hits_ea = int(np.count_nonzero(ea_flags & labels.astype(bool)))
            hits_pc = int(np.count_nonzero(pc_flags & labels.astype(bool)))
            # 此 tuple 是逐欄比較的排序鍵；負號把原本要最大化的量改成最小化排序。
            key = (metrics["FN"], metrics["FP"], -min(hits_ea, hits_pc), -hits_ea - hits_pc, -float(ea), -float(pc))
            if best is None or key < best[0]:
                best = (key, {"efficientad": float(ea), "patchcore": float(pc)}, metrics)
    return best[1], best[2]
