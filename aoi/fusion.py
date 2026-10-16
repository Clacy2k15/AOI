# 將兩個偵測器的原圖分數圖轉成定位遮罩、連通區域與顯示輪廓。
# 影像級 OK／NG 使用 image 門檻；像素級定位使用 pixel 門檻，兩者不可互相代替。
# Qwen 的紅圈來源可與對外保存的 OR 遮罩不同；這不會改變雙模型 OR 的 OK／NG 判決。

"""OpenCV localization in original coordinates; no erosion/min-area pruning."""
import cv2
import numpy as np


# 繪圖版本與線寬也會進入分類指紋，避免不同紅圈輸入共用舊分類快取。
QWEN_CONTOUR_RENDERING = {
    'version': 'input_resolution_v1',
    'thickness': 2,
    'coordinates': 'original_contours_scaled_to_input',
}


# 對 EA／PC 分別套用自己的影像與像素門檻，再做 OR。
# 回傳判決、布林遮罩、8 連通區域及原圖 bbox；bbox 的右／下界採不包含端點的 xyxy 慣例。
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


# 複製原始 RGB 並在原圖尺寸畫紅色外輪廓，用於一般結果顯示。
# 只改顯示副本，不修改輸入圖或布林遮罩；送 Qwen 的圖由另一個函式繪製。
def contour_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    return image


# 依長邊上限等比例縮圖，回傳 RGB 影像。
# scale 不超過 1，避免放大小圖；繪圖與 PNG 傳輸共用此尺寸計算。
def resize_qwen_image(rgb: np.ndarray, long_edge: int = 448) -> np.ndarray:
    """Use the same aspect-preserving dimensions for rendering and PNG transport."""
    height, width = rgb.shape[:2]
    scale = min(1.0, long_edge / max(height, width))
    return cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


# 先把圖縮到 Qwen 輸入尺寸，再投影原圖輪廓並畫固定 2 px 紅線。
# 投影輪廓點而非縮放二值遮罩，可避免一像素候選區在下採樣時整個消失。
def qwen_contour_image(rgb: np.ndarray, mask: np.ndarray, long_edge: int = 448) -> np.ndarray:
    """Draw nominal 2 px contours AFTER resizing, as in VELM's 448 px input.

    Project original contours instead of resampling the binary mask so even a
    one-pixel proposed region survives downsampling. Detector masks stay intact.
    """
    if mask.shape != rgb.shape[:2]:
        raise ValueError('Qwen mask must use original-image coordinates')
    image = resize_qwen_image(rgb, long_edge)
    height, width = image.shape[:2]
    scale = np.array([width / rgb.shape[1], height / rgb.shape[0]])
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # x、y 分別依實際輸出寬／高縮放，四捨五入後裁到有效畫布範圍。
    projected = [np.rint(contour * scale).astype(np.int32) for contour in contours]
    for contour in projected:
        contour[..., 0] = np.clip(contour[..., 0], 0, width - 1)
        contour[..., 1] = np.clip(contour[..., 1], 0, height - 1)
    cv2.drawContours(image, projected, -1, (255, 0, 0), QWEN_CONTOUR_RENDERING['thickness'])
    return image


# 只選擇 Qwen 應看的候選區來源，回傳 mask 與來源／回退資訊。
# patchcore_or_fallback 在 PC 遮罩完全為空時才回退 OR；兩者皆空仍交由 Qwen 看完整 query。
def qwen_localization(ea: dict, pc: dict, thresholds: dict, source: str = 'OR', fused: dict | None = None) -> dict:
    """Select only Qwen's proposed regions; image decision/OR mask stay separate."""
    if source == 'OR':
        fused = fused if fused is not None else combine(ea, pc, thresholds)
        return {'source': source, 'mask': fused['mask'], 'region_count': fused['region_count'],
                'localization_status': fused['localization_status']}
    if source not in ['patchcore', 'patchcore_or_fallback']:
        raise ValueError('Qwen mask_source must be OR, patchcore, or patchcore_or_fallback')
    mask = pc['map'] >= thresholds['patchcore']['pixel']
    count = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    if source == 'patchcore_or_fallback' and not count:
        result = qwen_localization(ea, pc, thresholds, 'OR', fused)
        return {**result, 'policy':source, 'fallback_reason':'empty_patchcore_mask'}
    return {'source': 'patchcore', 'policy':source, 'mask': mask, 'region_count': int(count),
            'localization_status': 'available' if count else 'missing', 'fallback_reason':None}
