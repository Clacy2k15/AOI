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


def dev_dataset(config: dict):
    manifest = read_json(ROOT / config["project"]["splits"])
    paths = manifest["dev"]["normal"] + [p for group in manifest["dev"]["anomalous"].values() for p in group]
    requested = {str((ROOT / p).resolve()).casefold() for p in paths}
    dataset = MVTecADDataset(root=ROOT / config["project"]["data_root"], category=config["project"]["category"])
    indices = [i for i, p in enumerate(dataset.samples.image_path) if str(Path(p).resolve()).casefold() in requested]
    if manifest["category"] != config["project"]["category"] or len(indices) != len(paths) or len(requested) != len(paths):
        raise ValueError("Dev manifest/product does not match Anomalib dataset")
    return dataset.subsample(indices)


def item_rgb(item) -> np.ndarray:
    return np.rint(item.image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def load_rgb(path: str | Path) -> np.ndarray:
    image = np.asarray(read_image(str(path), as_tensor=False))
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(image * 255).clip(0, 255)
    return image.astype(np.uint8)


class Detector:
    def __init__(self, config: dict, name: str):
        summary = training_summary(config)
        self.name = name
        self.path = Path(summary["models"][name]["model_path"])
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        manifest = read_json(ROOT / config["project"]["splits"])
        self.recipe = OpenCVRecipe("P0" if name == "efficientad" else name, config,
                                   load_rgb(ROOT / manifest["normal_reference"]))
        cls = ProjectEfficientAd if name == "efficientad" else Patchcore
        # Full checkpoints contain the backbone. Do not fetch initial pretrained
        # weights before strictly restoring those same saved tensors offline.
        checkpoint_options = {} if name == 'efficientad' else {'pre_trained': False}
        self.model = cls.load_from_checkpoint(str(self.path), map_location="cpu", strict=True, pre_processor=False,
                                              post_processor=False, evaluator=False, visualizer=False,
                                              **checkpoint_options).cuda().eval()

    def predict(self, rgb: np.ndarray) -> dict:
        start = time.perf_counter()
        tensor, metadata = recipe_tensor(rgb, self.recipe, self.name != "efficientad")
        preprocessing_ms = (time.perf_counter() - start) * 1000
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

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


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


def read_prediction_archive(path: Path, metadata: dict) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if "maps" not in arrays:
        arrays["maps"] = [arrays.pop(f"map_{i}") for i in range(len(metadata["rows"]))]
        arrays["gt_masks"] = [arrays.pop(f"gt_mask_{i}") for i in range(len(metadata["rows"]))]
    return arrays


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
