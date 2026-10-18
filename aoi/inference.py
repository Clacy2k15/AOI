"""只載入已校準的完整 checkpoint，執行原圖偵測，不訓練或下載。"""
from __future__ import annotations
import gc
import time
from pathlib import Path
import numpy as np
import torch
from anomalib.data.utils import read_image
from anomalib.models import Patchcore
from aoi.common import ROOT, read_json, output_path, model_directory, artifact_identity, detector_config_sha256
from aoi.data import recipe_tensor
from aoi.detectors import ProjectEfficientAd
from aoi.preprocessing import OpenCVRecipe, inverse_anomaly_map


def load_rgb(path: str | Path) -> np.ndarray:
    image = np.asarray(read_image(str(path), as_tensor=False))
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(image * 255).clip(0, 255)
    return image.astype(np.uint8)


class Detector:
    # 核對訓練摘要、模型路徑及固定正常參考圖，再建立相同的前處理配方。
    # 完整 checkpoint 在 CPU 上 strict=True 還原後才移到 CUDA；不套用框架自動門檻或視覺化。
    def __init__(self, config: dict, name: str):
        summary = training_summary(config)
        self.name = name
        self.path = Path(summary["models"][name]["model_path"])
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        manifest = read_json(ROOT / config["project"]["splits"])
        # EfficientAD 固定使用 P0；PatchCore 依已選組別套用 P0／PG／PD，兩者不共用錯誤的輸入配方。
        self.recipe = OpenCVRecipe("P0" if name == "efficientad" else name, config,
                                   load_rgb(ROOT / manifest["normal_reference"]))
        cls = ProjectEfficientAd if name == "efficientad" else Patchcore
        # Full checkpoints contain the backbone. Do not fetch initial pretrained
        # weights before strictly restoring those same saved tensors offline.
        # checkpoint 已包含 backbone；pre_trained=False 避免還原前先觸發網路下載或套用不同權重。
        checkpoint_options = {} if name == 'efficientad' else {'pre_trained': False}
        self.model = cls.load_from_checkpoint(str(self.path), map_location="cpu", strict=True, pre_processor=False,
                                              post_processor=False, evaluator=False, visualizer=False,
                                              **checkpoint_options).cuda().eval()

    # 將 RGB 轉成單張 batch 張量，執行 inference_mode 推論並取出影像分數及連續異常圖。
    # GPU 同步後記錄毫秒耗時，拒絕非有限值，最後回映原圖尺寸，尚不在此二值化。
    def predict(self, rgb: np.ndarray) -> dict:
        start = time.perf_counter()
        tensor, metadata = recipe_tensor(rgb, self.recipe, self.name != "efficientad")
        preprocessing_ms = (time.perf_counter() - start) * 1000
        # 先同步 GPU 排程再計時，量到的是實際模型運算而非只量到 CPU 發送指令的時間。
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model(tensor.unsqueeze(0).cuda())
        score = float(prediction.pred_score.flatten()[0].cpu())
        anomaly_map = prediction.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000
        if not np.isfinite(score) or not np.isfinite(anomaly_map).all():
            raise RuntimeError(f"Non-finite prediction from {self.name}")
        start = time.perf_counter()
        original_map = inverse_anomaly_map(anomaly_map, metadata)
        return {"score": score, "map": original_map, "metadata": metadata,
                "preprocessing_ms": preprocessing_ms, "inference_ms": inference_ms,
                "remap_ms": (time.perf_counter() - start) * 1000}

    # 移除模型引用、執行垃圾回收並釋放可回收 CUDA 快取。
    # 多產品切換時主動釋放舊模型，可減少不必要的顯存占用。
    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def training_summary(config: dict) -> dict:
    summary = read_json(output_path(config, "training_summary.json"))
    if summary["status"] != "completed" or summary.get("product", "metal_nut") != config["project"]["category"]:
        raise RuntimeError("Product training incomplete/mismatched; do not silently retrain")
    if config["project"]["category"] != "metal_nut":
        if summary["splits_sha256"] != artifact_identity(config)["splits_sha256"]:
            raise ValueError("Trained model manifest changed")
        if summary["detector_config_sha256"] != detector_config_sha256(config):
            raise ValueError("Trained model settings changed")
    for name, record in summary["models"].items():
        if Path(record["model_path"]).resolve() != (model_directory(config, name) / "model.ckpt").resolve():
            raise ValueError("Model path belongs to a different product")
    return summary
