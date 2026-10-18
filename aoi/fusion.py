"""影像分數與像素門檻各自做 OR；mask 保留原圖座標。"""
import cv2
import numpy as np


def combine(ea: dict, pc: dict, thresholds: dict) -> dict:
    ea_ng = ea["score"] >= thresholds["efficientad"]["image"]
    pc_ng = pc["score"] >= thresholds["patchcore"]["image"]
    ea_mask = ea["map"] >= thresholds["efficientad"]["pixel"]
    pc_mask = pc["map"] >= thresholds["patchcore"]["pixel"]
    if ea_mask.shape != pc_mask.shape:
        raise ValueError("Maps must have identical original-image coordinates")
    # 兩張圖必須已回映到同一原圖座標；不做侵蝕或最小面積過濾，以保留微小候選區。
    mask = ea_mask | pc_mask
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = []
    # 連通區 0 是背景；其餘每個區域保存面積與 bbox，便於 GUI 顯示及後續追溯。
    for index in range(1, count):
        x, y, width, height, area = (int(x) for x in stats[index])
        regions.append({"region_id": index, "bbox_xyxy": [x, y, x + width, y + height], "area": area})
    return {"decision": "NG" if ea_ng or pc_ng else "OK", "efficientad_ng": bool(ea_ng),
            "patchcore_ng": bool(pc_ng), "mask": mask, "region_count": len(regions), "regions": regions,
            "localization_status": "available" if regions else "missing"}


def contour_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    return image
