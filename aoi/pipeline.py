"""共用偵測流程：只輸出 OK／NG、原圖尺寸 mask 與疊圖。"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import cv2
import numpy as np
from aoi.common import ROOT, configure_runtime, load_inference, output_path, write_json
from aoi.fusion import combine, contour_image


class Pipeline:
    def __init__(self, config: dict, inference_path: str | Path | None = None):
        from aoi.offline import enable_offline
        enable_offline()
        self.config = config
        self.product = config['project']['category']
        self.settings = load_inference(config, inference_path, require_calibrated=True)
        configure_runtime(config)
        import torch
        from aoi.inference import Detector
        torch.set_num_threads(config['project']['cpu_threads'])
        cv2.setNumThreads(1)
        self.efficientad = self.patchcore = None
        group = self.settings['selected_patchcore']
        if self.product == 'capsule' and self.settings.get('capsule_model_pair', {}).get('enabled'):
            from aoi.capsule_runtime import CapsuleDetector
            self.efficientad = CapsuleDetector(config, 'efficientad', self.settings)
            try:
                self.patchcore = CapsuleDetector(config, group, self.settings)
            except Exception:
                self.close()
                raise
        else:
            self.efficientad = Detector(config, 'efficientad')
            try:
                if self.product == 'pill' and self.settings.get('pill_patchcore', {}).get('enabled'):
                    from aoi.pill_runtime import PillPatchcore
                    self.patchcore = PillPatchcore(config, self.settings)
                else:
                    self.patchcore = Detector(config, group)
            except Exception:
                self.close()
                raise

    def predict(self, input_path: str | Path, output_dir: str | Path | None = None) -> dict:
        from aoi.inference import load_rgb
        input_path = Path(input_path).resolve()
        sample_id = hashlib.sha256(f'{self.product}:{input_path}'.encode()).hexdigest()[:16]
        directory = ROOT / (output_dir or output_path(self.config, 'predictions')) / sample_id
        directory.mkdir(parents=True, exist_ok=True)
        record = {'sample_id': sample_id, 'input_path': str(input_path), 'product': self.product,
                  'decision': None, 'artifacts': {}, 'error': None}
        try:
            rgb = load_rgb(input_path)
            ea = self.efficientad.predict(rgb)
            pc = self.patchcore.predict(rgb)
            # 影像與像素門檻各自做 OR；OK 也保留實際 mask，避免改變既有校準語意。
            fused = combine(ea, pc, self.settings['thresholds'])
            mask = fused['mask']
            if mask.shape != rgb.shape[:2]:
                raise ValueError('Mask dimensions must match the original image')
            mask_path, overlay_path = directory / 'mask.png', directory / 'overlay.png'
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise IOError('Could not save mask')
            overlay = contour_image(rgb, mask)
            if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
                raise IOError('Could not save overlay')
            record.update(decision=fused['decision'], artifacts={'mask': str(mask_path),
                                                               'overlay': str(overlay_path)})
        except Exception as error:
            # 執行失敗不偽裝成 OK／NG；CLI 用非零退出碼回報。
            record.update(decision=None, artifacts={}, error=f'{type(error).__name__}: {error}')
        record['artifacts']['json'] = str(directory / 'result.json')
        write_json(directory / 'result.json', record)
        return record

    def close(self):
        for name in ('efficientad', 'patchcore'):
            detector = getattr(self, name, None)
            if detector is not None:
                detector.close()
                setattr(self, name, None)


def save_records(records: list[dict], output_dir: str | Path) -> None:
    directory = ROOT / output_dir
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'results.json', {'results': records})
    with (directory / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['product', 'sample_id', 'input_path',
                                                  'decision', 'mask', 'overlay', 'error'])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in ('product', 'sample_id', 'input_path',
                                                              'decision', 'error')} |
                            {key: record['artifacts'].get(key) for key in ('mask', 'overlay')})
