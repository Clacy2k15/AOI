# 三組固定前處理：P0 只縮放；PG 嘗試剛性對齊；PD 在亮度通道降噪並增強對比。
# OpenCV 尺寸參數採 (width, height)，而 NumPy shape／設定採 [height, width]，不可混用。
# 回傳的 metadata 記錄座標轉換，異常圖須先回到原圖座標才套用像素門檻及融合。

"""Fixed OpenCV recipes; scores/maps remain raw for stage B calibration."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


# 以 Otsu 二值化及最大外輪廓建立保守的工件外形遮罩。
# 只使用 query 影像，不讀 GT；用途是檢查 metal_nut 對齊後是否會把工件裁掉。
def _foreground(rgb: np.ndarray) -> np.ndarray:
    """A conservative exterior silhouette for detecting cropped metal nuts.

    This safeguard uses the query alone, never a ground-truth mask.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if contours:
        cv2.drawContours(result, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return result


# 封裝訓練與推論共用的 P0／PG／PD 配方，保留正常參考圖作為對齊模板。
# 幾何配方與亮度配方互斥，不會把兩種處理疊加成未經校正的新輸入。
class OpenCVRecipe:
    """Resize once, then optionally align OR denoise/enhance, before normalize."""

    # 核對組別，保存產品、輸入尺寸與配方，並將正常參考圖轉成灰階浮點模板。
    # 模板與 query 必須以相同尺寸送進 ECC，避免矩陣座標單位不一致。
    def __init__(self, group: str, config: dict, reference_rgb: np.ndarray):
        if group not in {"P0", "PG", "PD"}:
            raise ValueError(group)
        self.group = group
        self.product = config["project"]["category"]
        self.size = tuple(config["project"]["image_size"])
        self.options = config["preprocessing"]
        height, width = self.size
        template = cv2.resize(reference_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        self.template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255

    # 輸入原圖 RGB，產生固定大小的影像及完整幾何 metadata。
    # PG 無安全依據或對齊失敗時回退縮放圖；PD 只改亮度，不刻意改變色彩通道的瑕疵線索。
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        height, width = self.size
        image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        # 2×3 單位仿射矩陣代表只有縮放後的同座標輸入，也是 PG 失敗時的回退紀錄。
        identity = np.eye(2, 3, dtype=np.float32)
        metadata = {
            "group": self.group,
            "original_size_hw": list(rgb.shape[:2]),
            "resized_size_hw": [height, width],
            "alignment_failed": False,
            "aligned_to_query": identity.tolist(),
            "query_to_aligned": identity.tolist(),
            "matrix_coordinates": "resized_pixels",
            "warp_direction": "aligned/template -> resized query; OpenCV WARP_INVERSE_MAP",
        }
        if self.group == "PG":
            options = self.options["geometry"]
            # 目前只有 metal_nut 的外形裁切保護已驗證；其他產品不可僅因設定 PG 就宣稱安全對齊。
            guard = options.get("crop_guard", "metal_nut_silhouette" if self.product == "metal_nut" else "unverified")
            if guard != "metal_nut_silhouette":
                metadata.update(alignment_failed=True, crop_guard=guard,
                                alignment_failure_reason="alignment safety unverified for product; fallback P0")
                return np.ascontiguousarray(image), metadata
            query = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
            try:
                # ECC 只允許旋轉與平移；matrix 的方向是 template／aligned → resized query。
                correlation, matrix = cv2.findTransformECC(
                    self.template, query, identity.copy(), cv2.MOTION_EUCLIDEAN,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                     options["iterations"], options["epsilon"]),
                )
                angle = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
                metadata.update(ecc_correlation=float(correlation), rotation_degrees=angle,
                                attempted_aligned_to_query=matrix.tolist())
                if not np.isfinite(matrix).all():
                    raise ValueError("non-finite ECC transform")
                if abs(angle) > options["max_rotation_degrees"]:
                    raise ValueError("rotation exceeds 15 degrees")
                fraction = options["max_translation_fraction"]
                if abs(matrix[0, 2]) > width * fraction or abs(matrix[1, 2]) > height * fraction:
                    raise ValueError("translation exceeds 10 percent")
                # 以反矩陣把 query 外形投影到對齊畫布，先確認所有工件點都留在畫面內。
                forward = cv2.invertAffineTransform(matrix)
                rows, cols = np.nonzero(_foreground(image))
                if not len(rows):
                    raise ValueError("foreground silhouette unavailable")
                points = np.column_stack((cols, rows)).astype(np.float32)[None]
                aligned_points = cv2.transform(points, forward)[0]
                if ((aligned_points[:, 0] < 0).any() or (aligned_points[:, 0] > width - 1).any()
                        or (aligned_points[:, 1] < 0).any() or (aligned_points[:, 1] > height - 1).any()):
                    raise ValueError("alignment would crop the workpiece")
                # WARP_INVERSE_MAP 讓 OpenCV 以已知的反向取樣矩陣產生對齊影像。
                image = cv2.warpAffine(
                    image, matrix, (width, height),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                metadata.update(aligned_to_query=matrix.tolist(), query_to_aligned=forward.tolist())
            except (cv2.error, ValueError) as error:
                metadata.update(alignment_failed=True, alignment_failure_reason=str(error))
        elif self.group == "PD":
            options = self.options["denoise_enhance"]
            # LAB 的 L 是亮度；雙邊濾波與 CLAHE 只作用於 L，以減少對色彩瑕疵的干擾。
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            luminance = cv2.bilateralFilter(
                lab[:, :, 0], options["bilateral_d"], options["sigma_color"], options["sigma_space"],
            )
            clahe = cv2.createCLAHE(options["clahe_clip_limit"], tuple(options["clahe_tile_grid"]))
            lab[:, :, 0] = clahe.apply(luminance)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return np.ascontiguousarray(image), metadata


# 將模型的連續異常圖回到原始 query 尺寸，回傳 float32 分數圖。
# PG 成功時先撤銷對齊，再縮回原圖；失敗回退或 P0／PD 則只需尺寸回映。
def inverse_anomaly_map(anomaly_map: np.ndarray, metadata: dict) -> np.ndarray:
    """Return a floating map in original query coordinates, before thresholding."""
    height, width = metadata["resized_size_hw"]
    result = cv2.resize(np.asarray(anomaly_map, dtype=np.float32).squeeze(), (width, height))
    if metadata["group"] == "PG" and not metadata["alignment_failed"]:
        result = cv2.warpAffine(
            result, np.asarray(metadata["aligned_to_query"], dtype=np.float32), (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    original_height, original_width = metadata["original_size_hw"]
    return cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
