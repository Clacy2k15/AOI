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


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


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


class PillPatchcore:
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

    def predict(self,rgb):
        import torch
        started=time.perf_counter()
        out,metadata,support=self.recipe.prepare(rgb)
        tensor=torch.from_numpy(out).permute(2,0,1).float()/255
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

    def close(self):
        import torch
        del self.model
        gc.collect();torch.cuda.empty_cache()
