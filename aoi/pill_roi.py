"""Pill-only rigid alignment and a fixed ROI fitted to normal training images."""
from __future__ import annotations
import cv2
import numpy as np


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


def rigid_transform(mask):
    y, x = np.nonzero(mask)
    points = np.column_stack([x, y]).astype(np.float64)
    center = points.mean(0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
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


class PillROI:
    def __init__(self, parameters):
        self.parameters = parameters

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
        prepared = cv2.resize(aligned[y0:y1, x0:x1], (iw, ih), interpolation=cv2.INTER_LINEAR)
        sx, sy = iw/(x1-x0), ih/(y1-y0)
        resize = np.array([[sx, 0, -sx*x0+(sx-1)/2], [0, sy, -sy*y0+(sy-1)/2], [0, 0, 1]])
        affine = (resize @ np.vstack([matrix, [0, 0, 1]]))[:2]
        # This support is the complete ROI, not a foreground segmentation gate.
        support = np.zeros((h, w), np.uint8)
        support[y0:y1, x0:x1] = 1
        support = cv2.warpAffine(support, cv2.invertAffineTransform(matrix), (w, h), flags=cv2.INTER_NEAREST)>0
        metadata = {'group': 'PILL_ROI_ALIGN', 'angle_degrees': angle,
                    'original_to_input': affine.tolist(), 'scale_xy': [sx, sy],
                    'crop_xyxy': [x0, y0, x1, y1], 'original_size_hw': [h, w],
                    'resized_size_hw': [ih, iw], 'alignment_failed': False}
        return prepared, metadata, support

    @staticmethod
    def restore_map(anomaly_map, metadata, support):
        h, w = support.shape
        inverse = cv2.invertAffineTransform(np.asarray(metadata['original_to_input'], np.float64))
        result = cv2.warpAffine(np.asarray(anomaly_map, np.float32), inverse, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        result[~support] = 0
        return result
