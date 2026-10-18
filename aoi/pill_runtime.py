# pill 正式採用的 ROI PatchCore adapter；EfficientAD 繼續走一般 Detector。
# 部署 bundle 綁定 checkpoint、固定 ROI 參數、前處理程式、切分及門檻，防止不同實驗內容混用。
# 模型輸入使用 ImageNet 正規化；輸出保留 PatchCore 原生影像分數與原圖座標分數圖。

"""Deployment adapter for pill's ROI-aligned PatchCore; EfficientAD is unchanged."""
from __future__ import annotations
import gc
import hashlib
import time
from pathlib import Path
import numpy as np
from aoi.common import ROOT, read_json
from aoi.pill_roi import PillROI

VARIANT='PILL_ROI_ALIGN'


# 以串流方式計算檔案 SHA256。
# 不需一次將整份 checkpoint 讀入額外記憶體，用於確認部署來源未被替換。
def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


# 核對產品、ROI 組別、bundle 以及前處理程式／參數／切分的雜湊。
# 門檻、OR 決策及輸入尺寸也必須與校正紀錄一致；通過後回傳 bundle。
def validate_deployment(config,settings):
    if config['project']['category']!='pill' or settings.get('product')!='pill':
        raise ValueError('Pill PatchCore cannot be used by another product')
    deployment=settings['pill_patchcore']
    if not deployment.get('enabled') or settings['selected_patchcore']!=VARIANT:
        raise ValueError('Invalid pill detector variant')
    path=ROOT/deployment['bundle']
    if digest(path)!=deployment['bundle_sha256']:
        raise ValueError('Pill calibrated bundle changed')
    bundle=read_json(path)
    if bundle['product']!='pill' or bundle['variant']!=VARIANT:
        raise ValueError('Wrong pill detector bundle')
    # 這三個雜湊分別約束演算法程式、擬合參數及資料切分；只換其中一項也會被拒絕。
    for path,expected in [(ROOT/'aoi/pill_roi.py',bundle['preprocessing_code_sha256']),
                          (Path(bundle['preprocessing']),bundle['preprocessing_sha256']),
                          (ROOT/config['project']['splits'],bundle['split_sha256'])]:
        if digest(path)!=expected:
            raise ValueError('Pill calibration source changed: '+str(path))
    if settings['thresholds']!=bundle['thresholds'] or settings['fusion']['decision']!='OR':
        raise ValueError('Pill thresholds/decision differ from calibration')
    if config['project']['image_size']!=bundle['input_size_hw']:
        raise ValueError('Pill detector input size changed')
    return bundle


# 將 pill 的專用 ROI PatchCore 接入一般 pipeline 介面。
# 只取代 pill 的 PatchCore 槽位，不修改其他產品或 pill 的 EfficientAD 路徑。
class PillPatchcore:
    # 驗證部署與 checkpoint，建立 PillROI 並嚴格還原完整 PatchCore。
    # pre_trained=False 防止離線時嘗試取得初始 backbone；最後移到 CUDA 並切為 eval。
    def __init__(self,config,settings):
        import torch
        from anomalib.models import Patchcore
        bundle=validate_deployment(config,settings)
        self.path=Path(bundle['checkpoint'])
        self.model_sha256=digest(self.path)
        if self.model_sha256!=bundle['checkpoint_sha256']:
            raise ValueError('Pill checkpoint changed after calibration')
        self.recipe=PillROI(read_json(bundle['preprocessing']))
        self.model=Patchcore.load_from_checkpoint(str(self.path),map_location='cpu',pre_processor=False,
            post_processor=False,evaluator=False,visualizer=False,pre_trained=False,strict=True).cuda().eval()

    # 依固定 ROI 做姿態對齊與裁切，轉 CHW 浮點張量並正規化後推論。
    # 將異常圖回映到原圖 support，檢查有限數值，回傳原生影像分數、metadata 及毫秒耗時。
    def predict(self,rgb):
        import torch
        started=time.perf_counter()
        out,metadata,support=self.recipe.prepare(rgb)
        tensor=torch.from_numpy(out).permute(2,0,1).float()/255
        # 均值與標準差依 RGB 通道順序排列，[:,None,None] 用廣播套用到每個像素。
        tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        preprocessing_ms=(time.perf_counter()-started)*1000
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():prediction=self.model(tensor.unsqueeze(0).cuda())
        raw=prediction.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
        score=float(prediction.pred_score.flatten()[0].cpu())
        torch.cuda.synchronize();inference_ms=(time.perf_counter()-started)*1000
        if not np.isfinite(raw).all() or not np.isfinite(score):
            raise RuntimeError('Non-finite pill PatchCore output')
        started=time.perf_counter()
        original=self.recipe.restore_map(raw,metadata,support)
        metadata.update(model_checkpoint=str(self.path),model_sha256=self.model_sha256,
                        score_definition='native_weighted_nearest_neighbor')
        return {'score':score,'map':original,'metadata':metadata,'preprocessing_ms':preprocessing_ms,
                'inference_ms':inference_ms,'remap_ms':(time.perf_counter()-started)*1000}

    # 釋放模型引用並清理可釋放的 CUDA 記憶體。
    # pipeline 切換產品時會呼叫此方法，釋放目前模型資源。
    def close(self):
        import torch
        del self.model
        gc.collect();torch.cuda.empty_cache()
