# capsule 專屬前處理：找外形、PCA 剛性對齊、固定 ROI 裁切、等比例縮放及背景處理。
# fit 只使用正常訓練圖；部署時 prepare 讀取已保存的參數，不會根據新圖重新選 ROI。
# 本檔的原始位元組會被部署 bundle 雜湊驗證；註釋版需同步更新模板中的程式雜湊。

"""Isolated capsule experiment: rigid alignment, train-fitted ROI, RGB and foreground.

No ground-truth masks are consumed here. Existing P0/PG/PD recipes stay unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np


# 由 HSV 的暗色或高飽和色區域建立種子，閉運算後填滿最大外輪廓。
# 填滿外形可涵蓋表面偏白的瑕疵；回傳 bool 前景，前景比例超出保守範圍時拒絕推論。
def silhouette(rgb: np.ndarray) -> np.ndarray:
    """Join the black/orange body and fill its exterior, including pale defects."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    seed = ((hsv[..., 2] < 110) | ((hsv[..., 1] > 50) & (hsv[..., 2] < 245))).astype(np.uint8)
    # 核大小隨原圖尺寸調整，位元 OR 1 使直徑為奇數，讓橢圓核有清楚的中心。
    diameter = max(3, round(min(rgb.shape[:2]) * .007) | 1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Capsule foreground not found')
    contour = max(contours, key=cv2.contourArea)
    result = np.zeros(seed.shape, np.uint8)
    cv2.drawContours(result, [contour], -1, 1, -1)
    fraction = result.mean()
    if not .03 < fraction < .65:
        raise ValueError(f'Unsafe capsule foreground fraction: {fraction}')
    return result.astype(bool)


# 由前景座標的協方差矩陣求 PCA 長軸，估計旋轉角並把工件中心平移到畫布中心。
# 只旋轉／平移，不縮放；長軸方向統一朝右，傾斜超過 30 度視為超出已驗證範圍。
def rigid_transform(mask: np.ndarray) -> tuple[np.ndarray, float]:
    y, x = np.nonzero(mask)
    points = np.column_stack([x, y]).astype(np.float64)
    center = points.mean(0)
    # eigh 的特徵值由小到大排列，最後一欄特徵向量對應工件最長方向。
    _, vectors = np.linalg.eigh(np.cov(points.T))
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis = -axis
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if abs(angle) > 30:
        raise ValueError(f'Capsule tilt outside conservative rigid alignment: {angle}')
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    h, w = mask.shape
    matrix[:, 2] += np.array([w / 2, h / 2]) - center
    return matrix, angle


# 保存 capsule 固定 ROI 參數，提供前處理及異常圖逆向回映。
# 參數包含原圖尺寸、模型輸入尺寸、裁切框、前景邊界餘裕與背景顏色。
class CapsuleROI:
    # 保存校正好的參數字典。
    # 不在建構時 fit，確保每次部署都使用同一個已採用的裁切與縮放規則。
    def __init__(self, parameters: dict):
        self.parameters = parameters

    # 僅對正常訓練圖求對齊後外框的聯集，再加固定比例的水平／垂直餘裕。
    # 要求原圖尺寸一致；回傳含訓練張數、角度範圍與幾何參數的 CapsuleROI。
    @classmethod
    def fit(cls, train_images: list[np.ndarray], input_size=(256, 256)):
        """A single crop and scale fitted to NORMAL TRAIN images only."""
        shapes = {im.shape[:2] for im in train_images}
        if len(shapes) != 1:
            raise ValueError('Fixed capsule ROI requires a uniform source image size')
        h, w = train_images[0].shape[:2]
        boxes, angles = [], []
        for rgb in train_images:
            mask = silhouette(rgb)
            matrix, angle = rigid_transform(mask)
            aligned = cv2.warpAffine(mask.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
            y, x = np.nonzero(aligned)
            boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
            angles.append(angle)
        boxes = np.asarray(boxes)
        low = boxes[:, :2].min(0); high = boxes[:, 2:].max(0)
        # 訓練外框的 x 方向加 10%、y 方向加 25% 餘裕；這些比例是固定配方，不在推論時調整。
        margin = np.ceil((high - low) * np.array([.10, .25])).astype(int)
        low = np.maximum(low - margin, 0); high = np.minimum(high + margin, [w, h])
        return cls({'version': 1, 'source_size_hw': [h, w], 'input_size_hw': list(input_size),
                    'crop_xyxy': [*low.tolist(), *high.tolist()],
                    'foreground_margin_pixels': max(8, round(min(h, w) * .015)),
                    'background_rgb': [255, 255, 255], 'alignment': 'PCA_rigid_translation_rotation_no_scale',
                    'clahe': False, 'bilateral': False, 'fit_population': 'train_normal_only',
                    'train_count': len(train_images), 'train_angle_range': [min(angles), max(angles)]})

    # 輸入一張符合訓練尺寸的 RGB 圖，依固定規則對齊、去除遠離工件的背景並產生模型輸入。
    # 回傳 output、原圖到輸入的 metadata，以及原圖 allowed 前景餘裕遮罩；裁掉工件時直接失敗。
    def prepare(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        if [h, w] != self.parameters['source_size_hw']:
            raise ValueError('Unexpected capsule image dimensions')
        mask = silhouette(rgb)
        matrix, angle = rigid_transform(mask)
        radius = self.parameters['foreground_margin_pixels']
        allowed = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))).astype(bool)
        aligned_mask = cv2.warpAffine(allowed.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
        x0, y0, x1, y1 = self.parameters['crop_xyxy']
        # 用包含前景邊界餘裕的遮罩檢查裁切安全；不能靠先裁掉超界像素來隱藏問題。
        outside = aligned_mask.copy(); outside[y0:y1, x0:x1] = 0
        if outside.any():
            raise ValueError('Fixed training ROI would crop capsule foreground/margin')
        # Constant background is applied equally in normal training and inference.
        cleaned = rgb.copy(); cleaned[~allowed] = self.parameters['background_rgb']
        aligned = cv2.warpAffine(cleaned, matrix, (w, h), borderValue=(255, 255, 255))
        cropped = aligned[y0:y1, x0:x1]
        ih, iw = self.parameters['input_size_hw']
        # 以較小縮放比放入模型畫布，再用白邊補齊，保留 capsule 的長寬比例。
        scale = min(iw / (x1-x0), ih / (y1-y0))
        nw, nh = round((x1-x0)*scale), round((y1-y0)*scale)
        left, top = (iw-nw)//2, (ih-nh)//2
        output = np.full((ih, iw, 3), 255, np.uint8)
        output[top:top+nh, left:left+nw] = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
        # Match OpenCV resize pixel-center coordinates for precise inverse remapping.
        # 使用實際取整後的 nw／nh 計算比例；(s-1)/2 補償 OpenCV resize 的像素中心偏移。
        sx, sy = nw/(x1-x0), nh/(y1-y0)
        crop_resize = np.array([[sx, 0, left-sx*x0+(sx-1)/2],
                                [0, sy, top-sy*y0+(sy-1)/2], [0, 0, 1]])
        # 齊次座標中先剛性對齊、再裁切／縮放／補邊；矩陣相乘次序對應此先後順序。
        affine = (crop_resize @ np.vstack([matrix, [0, 0, 1]]))[:2]
        metadata = {'angle_degrees': angle, 'original_to_input': affine.tolist(),
                    'foreground_fraction': float(mask.mean()), 'allowed_fraction': float(allowed.mean()),
                    'crop_xyxy': [x0, y0, x1, y1], 'scale_xy': [sx, sy]}
        return output, metadata, allowed

    # 反轉 metadata 的仿射矩陣，把模型連續分數圖插值回原圖尺寸。
    # allowed 範圍外先填 0；部署 adapter 隨後會設定與門檻配套的背景 gate_floor。
    @staticmethod
    def restore_map(anomaly_map, metadata, allowed):
        h, w = allowed.shape
        inverse = cv2.invertAffineTransform(np.asarray(metadata['original_to_input'], np.float64))
        restored = cv2.warpAffine(np.asarray(anomaly_map, np.float32), inverse, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        restored[~allowed] = 0
        return restored
