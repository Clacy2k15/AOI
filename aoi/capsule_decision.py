"""Image-level OR decisions for an explicitly calibrated capsule model pair."""
import math


def decide_capsule(scores: dict, thresholds: dict) -> dict:
    flags = {}
    for name in ('efficientad', 'patchcore'):
        score, threshold = float(scores[name]), float(thresholds[name])
        if not math.isfinite(score) or not math.isfinite(threshold):
            raise ValueError(f'Non-finite {name} score or image threshold')
        flags[name] = score >= threshold
    return {'decision': 'NG' if any(flags.values()) else 'OK',
            'efficientad_ng': flags['efficientad'], 'patchcore_ng': flags['patchcore']}
