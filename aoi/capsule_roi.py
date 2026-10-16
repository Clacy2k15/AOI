"""Isolated capsule experiment: rigid alignment, train-fitted ROI, RGB and foreground.

No ground-truth masks are consumed here. Existing P0/PG/PD recipes stay unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np


def silhouette(rgb: np.ndarray) -> np.ndarray:
    """Join the black/orange body and fill its exterior, including pale defects."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    seed = ((hsv[..., 2] < 110) | ((hsv[..., 1] > 50) & (hsv[..., 2] < 245))).astype(np.uint8)
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


def rigid_transform(mask: np.ndarray) -> tuple[np.ndarray, float]:
    y, x = np.nonzero(mask)
    points = np.column_stack([x, y]).astype(np.float64)
    center = points.mean(0)
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


class CapsuleROI:
    def __init__(self, parameters: dict):
        self.parameters = parameters

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
        margin = np.ceil((high - low) * np.array([.10, .25])).astype(int)
        low = np.maximum(low - margin, 0); high = np.minimum(high + margin, [w, h])
        return cls({'version': 1, 'source_size_hw': [h, w], 'input_size_hw': list(input_size),
                    'crop_xyxy': [*low.tolist(), *high.tolist()],
                    'foreground_margin_pixels': max(8, round(min(h, w) * .015)),
                    'background_rgb': [255, 255, 255], 'alignment': 'PCA_rigid_translation_rotation_no_scale',
                    'clahe': False, 'bilateral': False, 'fit_population': 'train_normal_only',
                    'train_count': len(train_images), 'train_angle_range': [min(angles), max(angles)]})

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
        outside = aligned_mask.copy(); outside[y0:y1, x0:x1] = 0
        if outside.any():
            raise ValueError('Fixed training ROI would crop capsule foreground/margin')
        # Constant background is applied equally in normal training and inference.
        cleaned = rgb.copy(); cleaned[~allowed] = self.parameters['background_rgb']
        aligned = cv2.warpAffine(cleaned, matrix, (w, h), borderValue=(255, 255, 255))
        cropped = aligned[y0:y1, x0:x1]
        ih, iw = self.parameters['input_size_hw']
        scale = min(iw / (x1-x0), ih / (y1-y0))
        nw, nh = round((x1-x0)*scale), round((y1-y0)*scale)
        left, top = (iw-nw)//2, (ih-nh)//2
        output = np.full((ih, iw, 3), 255, np.uint8)
        output[top:top+nh, left:left+nw] = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)
        # Match OpenCV resize pixel-center coordinates for precise inverse remapping.
        sx, sy = nw/(x1-x0), nh/(y1-y0)
        crop_resize = np.array([[sx, 0, left-sx*x0+(sx-1)/2],
                                [0, sy, top-sy*y0+(sy-1)/2], [0, 0, 1]])
        affine = (crop_resize @ np.vstack([matrix, [0, 0, 1]]))[:2]
        metadata = {'angle_degrees': angle, 'original_to_input': affine.tolist(),
                    'foreground_fraction': float(mask.mean()), 'allowed_fraction': float(allowed.mean()),
                    'crop_xyxy': [x0, y0, x1, y1], 'scale_xy': [sx, sy]}
        return output, metadata, allowed

    @staticmethod
    def restore_map(anomaly_map, metadata, allowed):
        h, w = allowed.shape
        inverse = cv2.invertAffineTransform(np.asarray(metadata['original_to_input'], np.float64))
        restored = cv2.warpAffine(np.asarray(anomaly_map, np.float32), inverse, (w, h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        restored[~allowed] = 0
        return restored
