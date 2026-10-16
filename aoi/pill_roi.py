# pill 專屬的 PCA 長軸對齊與正常訓練圖擬合固定 ROI。
# 與 capsule 不同：此配方不移除背景，ROI 直接雙線性縮放，且不處理長軸的 180 度方向歧義。
# 回映時的 support 是完整 ROI 的可用範圍，不是藥丸前景分割遮罩。

"""Pill-only rigid alignment and a fixed ROI fitted to normal training images."""
from __future__ import annotations
import cv2
import numpy as np


# 以灰階 Otsu 二值化、閉運算及最大外輪廓取得藥丸的實心外形。
# 前景比例必須落在已設定的合理範圍，否則拋出例外而不勉強估計姿態。
def silhouette(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, seed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Pill foreground not found')
    mask = np.zeros(seed.shape, np.uint8)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    if not .1 < mask.mean() < .85:
        raise ValueError('Unsafe pill foreground area')
    return mask.astype(bool)


# 由前景座標的 PCA 長軸估計旋轉，並將中心平移到畫布中央。
# 長短軸特徵值比太接近時拒絕不明確的方向；也在連續座標階段檢查旋轉後是否裁切前景。
def rigid_transform(mask):
    y, x = np.nonzero(mask)
    points = np.column_stack([x, y]).astype(np.float64)
    center = points.mean(0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    # 長短軸差異不足時，微小像素變化可能讓主軸大幅旋轉，因此拒絕這類不可靠姿態。
    if values[-1] / values[0] < 1.2:
        raise ValueError('Pill orientation is ambiguous')
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis = -axis
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.)
    h, w = mask.shape
    matrix[:, 2] += np.array([w / 2, h / 2]) - center
    # Verify coordinates before rasterization, so clipping cannot hide a failure.
    corners = cv2.transform(points.astype(np.float32)[None], matrix)[0]
    if (corners[:, 0].min() < 0 or corners[:, 0].max() > w-1
            or corners[:, 1].min() < 0 or corners[:, 1].max() > h-1):
        raise ValueError('Alignment would crop pill foreground')
    return matrix, angle


# 保存 pill 固定裁切參數，提供 RGB 前處理與連續異常圖回映。
# 同一套 prepare 規則應同時用於建庫與部署，避免最近鄰特徵分布改變。
class PillROI:
    # 保存已完成擬合的參數，不讀取新的訓練資料。
    # 每張新圖只估計自己的剛性姿態，裁切框與模型尺寸保持固定。
    def __init__(self, parameters):
        self.parameters = parameters

    # 對尺寸一致的正常訓練圖對齊後求所有外框聯集，四周增加 8% 餘裕。
    # 保存訓練張數、角度範圍與背景／縮放規則，回傳可序列化參數的 PillROI。
    @classmethod
    def fit(cls, images, input_size=(256, 256)):
        if not images or len({im.shape for im in images}) != 1:
            raise ValueError('Need uniform normal training images')
        h, w = images[0].shape[:2]
        boxes, angles = [], []
        for rgb in images:
            matrix, angle = rigid_transform(silhouette(rgb))
            aligned = cv2.warpAffine(silhouette(rgb).astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
            y, x = np.nonzero(aligned)
            boxes.append([x.min(), y.min(), x.max()+1, y.max()+1])
            angles.append(angle)
        boxes = np.asarray(boxes)
        low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
        margin = np.ceil((high-low) * .08).astype(int)
        low, high = np.maximum(low-margin, 0), np.minimum(high+margin, [w, h])
        return cls({'version': 1, 'product': 'pill', 'source_size_hw': [h, w],
                    'input_size_hw': list(input_size), 'crop_xyxy': [*low.tolist(), *high.tolist()],
                    'alignment': 'PCA_long_axis_horizontal_rigid_translation_rotation_no_scale',
                    'rotation_180_disambiguation': False, 'resize': 'bilinear_ROI_to_input',
                    'background_removal': False, 'clahe': False, 'bilateral': False,
                    'fit_population': 'train_normal_only', 'train_count': len(images),
                    'margin_fraction': .08, 'train_angle_range': [min(angles), max(angles)]})

    # 檢查原圖尺寸，旋轉／平移後依固定 ROI 裁切，再縮放成模型輸入。
    # 回傳 prepared、metadata、原圖 support；若固定裁切會切到前景則直接報錯。
    def prepare(self, rgb):
        h, w = rgb.shape[:2]
        if [h, w] != self.parameters['source_size_hw']:
            raise ValueError('Unexpected pill source image dimensions')
        foreground = silhouette(rgb)
        matrix, angle = rigid_transform(foreground)
        aligned_mask = cv2.warpAffine(foreground.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
        x0, y0, x1, y1 = self.parameters['crop_xyxy']
        outside = aligned_mask.copy()
        outside[y0:y1, x0:x1] = 0
        if outside.any():
            raise ValueError('Fixed training ROI would crop pill foreground')
        ih, iw = self.parameters['input_size_hw']
        aligned = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        # 此處直接把 ROI 縮到 (iw,ih)，不使用 capsule 的等比例補白；需與各自訓練配方一致。
        prepared = cv2.resize(aligned[y0:y1, x0:x1], (iw, ih), interpolation=cv2.INTER_LINEAR)
        sx, sy = iw/(x1-x0), ih/(y1-y0)
        # 平移項包含裁切原點及像素中心補償，讓回映輪廓與原圖像素對齊。
        resize = np.array([[sx, 0, -sx*x0+(sx-1)/2], [0, sy, -sy*y0+(sy-1)/2], [0, 0, 1]])
        affine = (resize @ np.vstack([matrix, [0, 0, 1]]))[:2]
        # This support is the complete ROI, not a foreground segmentation gate.
        # 建立對齊座標的完整裁切框，再逆向轉回原圖，標示哪些位置有可靠的模型分數。
        support = np.zeros((h, w), np.uint8)
        support[y0:y1, x0:x1] = 1
        support = cv2.warpAffine(support, cv2.invertAffineTransform(matrix), (w, h), flags=cv2.INTER_NEAREST)>0
        metadata = {'group': 'PILL_ROI_ALIGN', 'angle_degrees': angle,
                    'original_to_input': affine.tolist(), 'scale_xy': [sx, sy],
                    'crop_xyxy': [x0, y0, x1, y1], 'original_size_hw': [h, w],
                    'resized_size_hw': [ih, iw], 'alignment_failed': False}
        return prepared, metadata, support

    # 利用原圖到模型輸入矩陣的反矩陣，把 float32 異常圖雙線性回映到原尺寸。
    # support 外設為 0，保留的是完整 ROI 範圍，不能解讀成只在藥丸前景上評分。
    @staticmethod
    def restore_map(anomaly_map, metadata, support):
        h, w = support.shape
        inverse = cv2.invertAffineTransform(np.asarray(metadata['original_to_input'], np.float64))
        result = cv2.warpAffine(np.asarray(anomaly_map, np.float32), inverse, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result[~support] = 0
        return result
