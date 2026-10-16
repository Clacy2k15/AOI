"""Region-weighted PRO calibration on original-coordinate frozen dev maps.

AU-PRO is reported for the continuous single-detector maps, using a dense
background-quantile sweep with integration at an explicit FPR limit. Its
monotone integration bounds disclose discretization error. A separate joint
OR operating point maximizes PRO under a background-pixel FPR budget.
"""
from __future__ import annotations

import cv2
import numpy as np


def gt_regions(mask):
    truth = np.asarray(mask, dtype=bool)
    count, labels = cv2.connectedComponents(truth.astype(np.uint8), connectivity=8)
    areas = np.bincount(labels.ravel(), minlength=count)
    weights = np.zeros(truth.shape, dtype=np.float64)
    weights[truth] = 1.0 / areas[labels[truth]]
    return labels, weights, count - 1


def background_thresholds(background, all_min, all_max, fpr_values, extra=()):
    """Strict float32 quantile boundaries; equal scores enter/leave together."""
    quantiles = np.quantile(background, 1 - np.asarray(fpr_values), method='higher').astype(np.float32)
    # Include both sides of tied score events. Otherwise a large background tie
    # could be interpolated together with unrelated lower-scoring GT pixels.
    values = np.concatenate([quantiles, np.nextafter(quantiles, np.float32(np.inf))])
    endpoints = [np.nextafter(np.float32(all_min), np.float32(-np.inf)),
                 np.nextafter(np.float32(all_max), np.float32(np.inf))]
    return np.unique(np.asarray([*values, *endpoints, *extra], dtype=np.float32))


def curve_integral(fpr, pro, limit):
    """Clip and normalize by the requested limit, preserving vertical events."""
    x, y = np.asarray(fpr, dtype=float), np.asarray(pro, dtype=float)
    if not 0 < limit <= 1 or len(x) != len(y) or np.any(np.diff(x) < -1e-12):
        raise ValueError('Expected an increasing PRO/FPR curve and 0 < limit <= 1')
    area = lower = upper = 0.0
    for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:], strict=True):
        if x0 >= limit:
            break
        if x1 <= x0:
            continue
        end = min(x1, limit)
        end_y = y0 + (y1 - y0) * (end - x0) / (x1 - x0)
        width = end - x0
        area += width * (y0 + end_y) / 2
        lower += width * y0
        # The true monotone curve can rise anywhere inside an unsampled interval.
        upper += width * y1
    if x[-1] < limit - 1e-12:
        raise ValueError('Curve does not reach the requested integration limit')
    return {'value': float(np.clip(area / limit, 0, 1)),
            'lower_bound': float(np.clip(lower / limit, 0, 1)),
            'upper_bound': float(np.clip(upper / limit, 0, 1)),
            'fpr_limit': float(limit), 'method': 'background_quantile_trapezoids',
            'threshold_independent': True}


def point_metrics(masks, truths):
    regions = hits = image_misses = fp = background = 0
    pro_sum = 0.0
    dice, coverage, precision, normal_area, per_image = [], [], [], [], []
    for mask, gt in zip(masks, truths, strict=True):
        gt = gt.astype(bool)
        labels, _, count = gt_regions(gt)
        tp, area, garea = int(np.sum(mask & gt)), int(mask.sum()), int(gt.sum())
        fp += area - tp
        background += gt.size - garea
        local_coverage = []
        if count:
            sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
            overlaps = np.bincount(labels[mask].ravel(), minlength=count + 1)[1:]
            local_coverage = (overlaps / sizes).tolist()
            pro_sum += sum(local_coverage); regions += count
            hits += int(np.sum(overlaps > 0)); image_misses += int(tp == 0)
            dice.append(2 * tp / (area + garea))
            coverage.append(tp / garea); precision.append(tp / area if area else 0)
        else:
            normal_area.append(area / gt.size)
        per_image.append({'GT_regions': count, 'region_coverages': local_coverage,
                          'TP_pixels': tp, 'FP_pixels': area - tp, 'GT_pixels': garea,
                          'mask_pixels': area, 'image_pixels': int(gt.size)})
    return {'PRO': pro_sum / regions, 'GT_regions': regions,
            'regions_without_overlap': regions - hits, 'NG_images_without_overlap': image_misses,
            'background_FPR': fp / background, 'background_pixels': background, 'FP_pixels': fp,
            'NG_Dice_mean': float(np.mean(dice)), 'NG_GT_coverage_mean': float(np.mean(coverage)),
            'NG_precision_mean': float(np.mean(precision)),
            'normal_mask_fraction_mean': float(np.mean(normal_area)) if normal_area else None, 'images': per_image}


def calibrate_pair(ea_maps, pc_maps, truths, baseline_pixels, *, budget=.01, limit=.30,
                   curve_quantiles=2001, low_fpr_quantiles=1001, joint_quantiles=161):
    if not 0 < budget <= limit <= 1:
        raise ValueError('Require 0 < operating background FPR <= AU-PRO FPR limit <= 1')
    if not (len(ea_maps) == len(pc_maps) == len(truths)):
        raise ValueError('Mismatched dev maps/masks')
    truths = [np.asarray(g, dtype=bool) for g in truths]
    for e, p, g in zip(ea_maps, pc_maps, truths, strict=True):
        if e.shape != p.shape or e.shape != g.shape or not np.isfinite(e).all() or not np.isfinite(p).all():
            raise ValueError('Invalid original-coordinate maps')
        if e.dtype != np.float32 or p.dtype != np.float32:
            raise ValueError('Expected frozen float32 detector maps')
    background = [np.concatenate([m[~g] for m, g in zip(maps, truths, strict=True)])
                  for maps in [ea_maps, pc_maps]]
    if not len(background[0]) or not any(g.any() for g in truths):
        raise ValueError('Need both background pixels and GT regions on dev')
    bounds = [(min(float(m.min()) for m in maps), max(float(m.max()) for m in maps)) for maps in [ea_maps, pc_maps]]
    jt, ct = [], []
    for bg, (low, high), old in zip(background, bounds, baseline_pixels, strict=True):
        jt.append(background_thresholds(bg, low, high, np.linspace(0, budget, joint_quantiles), [old]))
        curve_fprs = np.unique(np.concatenate([np.linspace(0, limit, curve_quantiles),
                                               np.linspace(0, budget, low_fpr_quantiles)]))
        ct.append(background_thresholds(bg, low, high, curve_fprs))
    del background
    shape = (len(jt[0]), len(jt[1]))
    fp = np.zeros(shape, dtype=np.int64)
    pro_sum = np.zeros(shape)
    dice_sum = np.zeros(shape)
    curve_bg = [np.zeros(len(t) + 1, dtype=np.int64) for t in ct]
    curve_region = [np.zeros(len(t) + 1) for t in ct]
    region_count = ng_count = background_count = 0
    for e, p, g in zip(ea_maps, pc_maps, truths, strict=True):
        _, weights, nr = gt_regions(g)
        region_count += nr; ng_count += int(bool(nr)); background_count += int((~g).sum())
        bins = [np.searchsorted(t, m.ravel(), side='right') for t, m in zip(jt, [e, p], strict=True)]
        pairs = bins[0] * (shape[1] + 1) + bins[1]
        dims = (shape[0] + 1, shape[1] + 1)
        bg_hist = np.bincount(pairs[~g.ravel()], minlength=np.prod(dims)).reshape(dims)
        fg_hist = np.bincount(pairs[g.ravel()], minlength=np.prod(dims)).reshape(dims)
        weighted = np.bincount(pairs[g.ravel()], weights=weights[g], minlength=np.prod(dims)).reshape(dims)
        image_fp = int((~g).sum()) - bg_hist.cumsum(0).cumsum(1)[:-1, :-1]
        image_tp = int(g.sum()) - fg_hist.cumsum(0).cumsum(1)[:-1, :-1]
        fp += image_fp
        pro_sum += nr - weighted.cumsum(0).cumsum(1)[:-1, :-1]
        if nr:
            dice_sum += 2 * image_tp / (image_tp + image_fp + int(g.sum()))
        for k, (t, m) in enumerate(zip(ct, [e, p], strict=True)):
            b = np.searchsorted(t, m.ravel(), side='right')
            curve_bg[k] += np.bincount(b[~g.ravel()], minlength=len(t) + 1)
            curve_region[k] += np.bincount(b[g.ravel()], weights=weights[g], minlength=len(t) + 1)
    pro = np.clip(pro_sum / region_count, 0, 1)
    fpr = fp / background_count
    dice = dice_sum / ng_count
    feasible = fp <= int(np.floor(budget * background_count))
    best_pro = pro[feasible].max()
    eligible = feasible & (pro >= best_pro - 1e-10)
    best_fp = fp[eligible].min()
    eligible &= fp == best_fp
    choices = list(zip(*np.where(eligible), strict=True))
    chosen = max(choices, key=lambda ij: (dice[ij], sum(ij)))
    pixels = [float(jt[k][chosen[k]]) for k in range(2)]
    baseline_masks = [(e >= baseline_pixels[0]) | (p >= baseline_pixels[1]) for e, p in zip(ea_maps, pc_maps, strict=True)]
    selected_masks = [(e >= pixels[0]) | (p >= pixels[1]) for e, p in zip(ea_maps, pc_maps, strict=True)]
    baseline = point_metrics(baseline_masks, truths)
    selected = point_metrics(selected_masks, truths)
    assert abs(selected['PRO'] - pro[chosen]) < 1e-8
    assert selected['FP_pixels'] == int(fp[chosen])
    assert selected['background_FPR'] <= budget
    baseline_index = tuple(int(np.flatnonzero(t == np.float32(old))[0]) for t, old in zip(jt, baseline_pixels, strict=True))
    assert abs(baseline['PRO'] - pro[baseline_index]) < 1e-8
    curves, quality = {}, {}
    for k, name in enumerate(['efficientad', 'patchcore']):
        x = (background_count - curve_bg[k].cumsum()[:-1])[::-1] / background_count
        y = np.clip((region_count - curve_region[k].cumsum()[:-1])[::-1] / region_count, 0, 1)
        curves[name] = {'thresholds': ct[k][::-1], 'FPR': x, 'PRO': y}
        quality[name] = {f'AU_PRO_{cap:.2f}': curve_integral(x, y, cap) for cap in sorted({budget, .05, limit}) if cap <= limit}
    return {'pixels': pixels, 'baseline': baseline, 'selected': selected, 'single_detector_AU_PRO': quality,
            'curves': curves, 'grid': {'thresholds': jt, 'PRO': pro, 'FPR': fpr, 'Dice': dice, 'chosen': chosen},
            'exact_operating_point_verified': True,
            'limitations': ['AU-PRO is a sweep statistic; changing one threshold cannot change a fixed continuous map AU-PRO.',
                            'Single-detector AU-PRO is densely sampled; monotone integration bounds are provided.',
                            'Independent OR threshold pairs do not define a single continuous fused score; no fused AU-PRO is claimed.']}
