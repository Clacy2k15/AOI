# 一般 EfficientAD／PatchCore checkpoint 的載入、原圖推論與歷史 dev 快取工具。
# Detector 對單張新圖只載入正式權重，不訓練、不建庫；ROI 專屬產品由 pipeline 分流到各自 adapter。
# dev 快取相關函式需要完整開發資料與歷史紀錄，推論精簡包未包含這些資料。

"""Load existing Anomalib checkpoints once; reuse stage A input recipes."""
from __future__ import annotations
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.utils import read_image
from anomalib.models import Patchcore

from aoi.common import ROOT, read_json, write_json, output_path, model_directory, artifact_identity, load_config, detector_config_sha256
from aoi.data import recipe_tensor
from aoi.detectors import ProjectEfficientAd
from aoi.preprocessing import OpenCVRecipe, inverse_anomaly_map


# 依固定 dev 清單從 Anomalib 資料集篩出正常與異常樣本。
# 核對產品、清單張數及去重後張數，避免缺圖或重複清單造成評估母體不一致。
def dev_dataset(config: dict):
    manifest = read_json(ROOT / config["project"]["splits"])
    paths = manifest["dev"]["normal"] + [p for group in manifest["dev"]["anomalous"].values() for p in group]
    requested = {str((ROOT / p).resolve()).casefold() for p in paths}
    dataset = MVTecADDataset(root=ROOT / config["project"]["data_root"], category=config["project"]["category"])
    indices = [i for i, p in enumerate(dataset.samples.image_path) if str(Path(p).resolve()).casefold() in requested]
    if manifest["category"] != config["project"]["category"] or len(indices) != len(paths) or len(requested) != len(paths):
        raise ValueError("Dev manifest/product does not match Anomalib dataset")
    return dataset.subsample(indices)


# 把資料集樣本的 CHW 浮點影像還原為 HWC uint8 RGB。
# 先乘 255、四捨五入再裁切，供與一般讀圖相同的前處理使用。
def item_rgb(item) -> np.ndarray:
    return np.rint(item.image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


# 用 Anomalib 讀取影像並統一輸出 uint8 RGB。
# 若讀圖工具回傳浮點 0～1 影像，先還原到 0～255；不在此猜測產品或瑕疵類別。
def load_rgb(path: str | Path) -> np.ndarray:
    image = np.asarray(read_image(str(path), as_tensor=False))
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(image * 255).clip(0, 255)
    return image.astype(np.uint8)


# 一般模型的統一推論介面，建構時載入一次，predict 可重複處理多張圖。
# 每次回傳原始影像分數、原圖座標異常圖及實際前處理／推論／回映耗時。
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


# 開發用的快取入口：身分一致時讀取保存結果，否則才對固定 dev 集推論並保存。
# 保留每張影像 GT、分數、metadata 與實測耗時；本函式不是任意新圖的快取捷徑。
def cached_dev_predictions(config: dict, name: str) -> tuple[dict, dict]:
    dataset = dev_dataset(config)
    summary = training_summary(config)
    model_path = Path(summary["models"][name]["model_path"])
    manifest = read_json(ROOT / config["project"]["splits"])
    identity = artifact_identity(config)
    fingerprint = prediction_fingerprint(config, name, model_path, manifest, list(dataset.samples.image_path))
    directory = output_path(config, "dev_predictions") / name
    metadata_path = directory / "metadata.json"
    archive_path = directory / "predictions.npz"
    if metadata_path.exists() and archive_path.exists():
        metadata = read_json(metadata_path)
        if metadata["fingerprint"] == fingerprint or legacy_cache_compatible(config, name, metadata_path, model_path):
            return read_prediction_archive(archive_path, metadata), {**metadata, "product": identity["product"], "identity": identity}
    detector = Detector(config, name)
    rows, maps, scores, masks = [], [], [], []
    try:
        for item, row in zip(dataset, dataset.samples.itertuples(), strict=True):
            result = detector.predict(item_rgb(item))
            maps.append(result.pop("map"))
            scores.append(result["score"])
            masks.append(item.gt_mask.numpy().astype(bool))
            rows.append({"path": str(Path(item.image_path).resolve()), "label": int(item.gt_label),
                         "defect_class": row.label, **result})
    finally:
        detector.close()
    # npz 禁用 pickle，因此不同原圖大小不能存成 object 陣列，改以 map_0、map_1 等鍵分別保存。
    uniform = len({m.shape for m in maps}) == 1
    arrays = {"maps": np.stack(maps) if uniform else maps, "scores": np.asarray(scores),
              "gt_masks": np.stack(masks) if uniform else masks,
              "labels": np.asarray([row["label"] for row in rows], dtype=np.uint8)}
    directory.mkdir(parents=True, exist_ok=True)
    saved = arrays if uniform else {"scores": arrays["scores"], "labels": arrays["labels"],
            **{f"map_{i}": m for i, m in enumerate(maps)}, **{f"gt_mask_{i}": m for i, m in enumerate(masks)}}
    np.savez_compressed(archive_path, **saved)
    metadata = {"fingerprint": fingerprint, "identity": identity, "product": identity["product"],
                "detector": name, "split": "dev", "rows": rows,
                "timings_source": "actual CUDA inference; cached reads are not inference timings"}
    write_json(metadata_path, metadata)
    return arrays, metadata


# 將產品、切分、模型設定、checkpoint 路徑／大小／mtime、前處理及影像路徑組合成指紋。
# 也納入 adapter／前處理原始碼位元組；即使只加註釋，也會使舊 dev 快取指紋失效。
def prediction_fingerprint(config, name, model_path, manifest, paths):
    identity = artifact_identity(config)
    return hashlib.sha256(json.dumps({
        "product": identity["product"], "split": "dev", "splits_sha256": identity["splits_sha256"], "detector": name,
        "model_settings": config["efficientad" if name == "efficientad" else "patchcore"],
        "checkpoint": str(model_path), "size": model_path.stat().st_size,
        "mtime": model_path.stat().st_mtime_ns, "preprocessing": config["preprocessing"],
        "image_size": config["project"]["image_size"], "reference": manifest["normal_reference"],
        "paths": paths,
        "adapter_source": hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "aoi/preprocessing.py").read_bytes()
                                         + (ROOT / "aoi/data.py").read_bytes()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()


# 讀取 npz 快取並複製陣列，避免 archive 關閉後仍依賴其資源。
# 等尺寸資料可直接 stack；不等尺寸資料依 metadata 的 rows 順序還原逐張 map／gt_mask 清單。
def read_prediction_archive(path: Path, metadata: dict) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if "maps" not in arrays:
        arrays["maps"] = [arrays.pop(f"map_{i}") for i in range(len(metadata["rows"]))]
        arrays["gt_masks"] = [arrays.pop(f"gt_mask_{i}") for i in range(len(metadata["rows"]))]
    return arrays


# 只讀既有 dev 快取，核對當前指紋或明確允許的 metal_nut 舊快取。
# 缺檔或身分不一致就報錯，不偷偷執行新推論；供分類開發的 cache-only 路徑使用。
def existing_dev_predictions(config: dict, name: str) -> tuple[dict, dict]:
    """Cache-only consumer for Qwen; missing/stale predictions never trigger inference."""
    dataset = dev_dataset(config)
    summary = training_summary(config)
    model_path = Path(summary["models"][name]["model_path"])
    folder = output_path(config, "dev_predictions") / name
    metadata = read_json(folder / "metadata.json")
    manifest = read_json(ROOT / config["project"]["splits"])
    expected = prediction_fingerprint(config, name, model_path, manifest, list(dataset.samples.image_path))
    if metadata["fingerprint"] != expected and not legacy_cache_compatible(config, name, folder / "metadata.json", model_path):
        raise ValueError("Product dev predictions are stale/mismatched; regenerate only in authorized dev stage")
    return read_prediction_archive(folder / "predictions.npz", metadata), metadata


# 確認訓練完成、產品相符、切分與偵測器配方未變，並驗證每個模型槽位的預期路徑。
# 回傳摘要供權重載入使用；不會因檔案缺失就自動重訓或借用別的產品。
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


# 僅在 metal_nut 舊紀錄明確存在時，核對設定、切分、metadata 及模型檔案身分。
# 全部符合歷史清單才允許沿用；其他產品或來源不完整時一律 False。
def legacy_cache_compatible(config: dict, name: str, metadata_path: Path, model_path: Path) -> bool:
    """Reuse verified old nut predictions without editing original metadata."""
    inventory_path = ROOT / "outputs/expansion/legacy_inventory.json"
    if config["project"]["category"] != "metal_nut" or not inventory_path.is_file():
        return False
    inventory = read_json(inventory_path)
    if config != load_config(ROOT / "configs/aoi.yaml"):
        return False
    for path in [ROOT / "configs/aoi.yaml", ROOT / config["project"]["splits"], metadata_path]:
        entry = inventory["files"].get(path.relative_to(ROOT).as_posix())
        if not entry or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            return False
    old = inventory["models"].get(name, {})
    return (str(model_path) == old.get("path") and model_path.stat().st_size == old.get("size")
            and model_path.stat().st_mtime_ns == old.get("mtime_ns"))
