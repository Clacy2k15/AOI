# capsule 影像級雙模型 OR 判決的小工具。此處只處理單張影像分數，不產生像素遮罩。
# 門檻必須來自與這組 capsule 權重配對的校正結果，不能借用其他產品的數值。

"""Image-level OR decisions for an explicitly calibrated capsule model pair."""
import math


# 將兩模型的分數及門檻轉為 float，先拒絕 NaN／Infinity，再計算各自是否達標。
# 任何模型達門檻即 NG；兩者皆未達標才 OK，並回傳各模型旗標方便診斷。
def decide_capsule(scores: dict, thresholds: dict) -> dict:
    flags = {}
    for name in ('efficientad', 'patchcore'):
        score, threshold = float(scores[name]), float(thresholds[name])
        if not math.isfinite(score) or not math.isfinite(threshold):
            raise ValueError(f'Non-finite {name} score or image threshold')
        flags[name] = score >= threshold
    return {'decision': 'NG' if any(flags.values()) else 'OK',
            'efficientad_ng': flags['efficientad'], 'patchcore_ng': flags['patchcore']}
