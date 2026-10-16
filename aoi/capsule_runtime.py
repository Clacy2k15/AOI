# capsule 正式採用的 ROI 雙模型執行介面，負責權重／前處理／門檻配對及座標回映。
# EfficientAD 的影像分數定義為回映後 allowed 範圍內的最大值；PatchCore 保留模型原生影像分數。
# 背景填充值 gate_floor 必須小於像素門檻，否則背景可能被誤判成異常。

"""Capsule-only adapter for the adopted ROI/alignment detector checkpoints."""
from __future__ import annotations
import gc
import hashlib
import time
from pathlib import Path
import numpy as np
from aoi.common import ROOT,read_json
from aoi.capsule_roi import CapsuleROI

VARIANT='CAPSULE_ROI_MASK_ALIGN'


# 讀取指定檔案並計算 SHA256，供部署 bundle、程式、參數及 checkpoint 一致性核對。
# 此版本將檔案讀入記憶體；大模型的驗證在 adapter 建構時執行一次。
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 逐層核對 capsule 產品、專屬組別、bundle 檔案、前處理程式及參數的雜湊。
# 再核對採用的 image／pixel 門檻及 OR 決策；任一項不一致即拒絕載入，回傳通過的 bundle。
def validate_deployment(config,settings):
    deployment=settings['capsule_model_pair']
    if config['project']['category']!='capsule' or settings.get('product')!='capsule':
        raise ValueError('Capsule detector deployment cannot be used by another product')
    if not deployment.get('enabled') or settings['selected_patchcore']!=VARIANT:
        raise ValueError('Invalid capsule detector variant')
    path=ROOT/deployment['bundle']
    if digest(path)!=deployment['bundle_sha256']:
        raise ValueError('Capsule calibrated bundle changed')
    bundle=read_json(path)
    if bundle.get('product')!='capsule':raise ValueError('Incorrect detector bundle product')
    if digest(ROOT/'aoi/capsule_roi.py')!=bundle['preprocessing_code_sha256']:
        raise ValueError('Capsule preprocessing source changed after calibration')
    parameters=Path(bundle['preprocessing'])
    key=str(parameters.relative_to(ROOT))
    if digest(parameters)!=bundle['source_sha256'][key]:
        raise ValueError('Capsule preprocessing parameters changed after calibration')
    for name in ['efficientad','patchcore']:
        expected={'image':bundle['image_thresholds'][name],'pixel':bundle[name]['thresholds']['pixel']}
        if settings['thresholds'][name]!=expected:
            raise ValueError(f'{name} thresholds do not match the adopted capsule calibration')
    if settings.get('fusion',{}).get('decision')!='OR':raise ValueError('Capsule pair requires calibrated OR decisions')
    return bundle


# 將 capsule 專屬 EfficientAD 或 PatchCore 包裝成 pipeline 共用的 predict／close 介面。
# 回傳原圖分數圖、影像分數、幾何 metadata 及各階段毫秒耗時。
class CapsuleDetector:
    # 驗證部署內容及 checkpoint 雜湊，還原對應模型並建立 CapsuleROI。
    # 先在 CPU 嚴格還原，再移至 CUDA eval；PatchCore 禁止先下載另一份 backbone 初始權重。
    def __init__(self,config,name,settings):
        import torch
        from anomalib.models import Patchcore
        from aoi.detectors import ProjectEfficientAd
        bundle=validate_deployment(config,settings)
        self.name='efficientad' if name=='efficientad' else 'patchcore'
        if name not in ['efficientad',VARIANT]:raise ValueError('Unrecognized capsule detector slot')
        self.path=Path(bundle[self.name]['checkpoint'])
        # 權重必須與校正 bundle 的內容雜湊相同，不能只靠檔名判定是同一份模型。
        self.model_sha256=digest(self.path)
        if self.model_sha256!=bundle['source_sha256'][str(self.path.relative_to(ROOT))]:
            raise ValueError('Capsule checkpoint changed after calibration')
        self.recipe=CapsuleROI(read_json(bundle['preprocessing']))
        if self.recipe.parameters['input_size_hw']!=config['project']['image_size']:
            raise ValueError('Capsule model input size changed')
        self.gate_floor=bundle['efficientad']['gate_floor'] if self.name=='efficientad' else 0.
        if self.gate_floor>=settings['thresholds'][self.name]['pixel']:
            raise ValueError('Foreground gate fill must remain below the pixel threshold')
        cls=ProjectEfficientAd if self.name=='efficientad' else Patchcore
        options={} if self.name=='efficientad' else {'pre_trained':False}
        self.model=cls.load_from_checkpoint(str(self.path),map_location='cpu',strict=True,pre_processor=False,
            post_processor=False,evaluator=False,visualizer=False,**options).cuda().eval()

    # 前處理 RGB、依模型類型正規化、執行 CUDA 推論，再回映異常圖與套用背景門控。
    # 檢查 NaN／Infinity；輸出分數定義與校正時一致，並保留可追溯的模型雜湊及幾何資訊。
    def predict(self,rgb):
        import torch
        started=time.perf_counter()
        prepared,metadata,allowed=self.recipe.prepare(rgb)
        tensor=torch.from_numpy(prepared).permute(2,0,1).float()/255
        if self.name=='patchcore':
            tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        preprocessing_ms=(time.perf_counter()-started)*1000
        # GPU 預設非同步執行；量測推論時間前同步，避免把之前排隊工作算入或漏算。
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():prediction=self.model(tensor.unsqueeze(0).cuda())
        raw=prediction.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
        native_score=float(prediction.pred_score.flatten()[0].cpu())
        torch.cuda.synchronize();inference_ms=(time.perf_counter()-started)*1000
        if not np.isfinite(raw).all() or not np.isfinite(native_score):
            raise RuntimeError('Non-finite capsule detector output')
        started=time.perf_counter()
        original_map=self.recipe.restore_map(raw,metadata,allowed)
        # 此處的 EA 分數不是 native_score；改用原圖前景最大值是已採用配方的一部分，不能任意互換。
        score=float(original_map[allowed].max()) if self.name=='efficientad' else native_score
        original_map[~allowed]=self.gate_floor
        metadata.update(group=VARIANT,detector=self.name,model_checkpoint=str(self.path),
            model_sha256=self.model_sha256,foreground_gate_fill=self.gate_floor,
            score_definition='foreground_map_max' if self.name=='efficientad' else 'native_weighted_nearest_neighbor',
            original_size_hw=list(rgb.shape[:2]),resized_size_hw=self.recipe.parameters['input_size_hw'],
            alignment_failed=False,clahe=False,bilateral=False)
        return {'score':score,'map':original_map,'metadata':metadata,'preprocessing_ms':preprocessing_ms,
                'inference_ms':inference_ms,'remap_ms':(time.perf_counter()-started)*1000}

    # 釋放此 adapter 持有的模型、回收 Python 物件並清空可釋放的 CUDA 快取。
    # 只處理偵測器資源，不終止共用的原生 Qwen 程序。
    def close(self):
        import torch
        del self.model
        gc.collect();torch.cuda.empty_cache()
