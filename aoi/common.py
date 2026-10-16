from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path, _parents: tuple = ()) -> dict:
    path = (ROOT / path).resolve()
    if path in _parents:
        raise ValueError("Circular config inheritance: " + str(path))
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    parents = config.pop("inherits", [])
    if isinstance(parents, str):
        parents = [parents]
    base = {}
    for parent in parents:
        base = _merge(base, load_config(ROOT / parent, (*_parents, path)))
    return _merge(base, config)


def output_path(config: dict, suffix: str = "") -> Path:
    return ROOT / config["project"]["output_root"] / suffix


def model_directory(config: dict, name: str) -> Path:
    base = ROOT / config["project"].get("model_root", "models")
    return base / ("efficientad" if name == "efficientad" else f"patchcore/{name}")


def inference_path(config: dict) -> Path:
    return ROOT / config["project"].get("inference_config", "configs/inference.yaml")


def artifact_identity(config: dict, split: str = "dev") -> dict:
    return {"product": config["project"]["category"], "split": split,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "splits_sha256": hashlib.sha256((ROOT / config["project"]["splits"]).read_bytes()).hexdigest()}


def detector_config_sha256(config: dict) -> str:
    # Prompt edits must not invalidate identical detector weights/maps.
    project = {k:v for k,v in config["project"].items() if k != "inference_config"}
    settings = {"project": project, **{k:config[k] for k in ["efficientad","patchcore","preprocessing","loader"]}}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def load_inference(config: dict, path: str | Path | None = None, require_calibrated: bool = False) -> dict:
    from aoi.velm import product_options
    settings = load_config(path or inference_path(config))
    product = config["project"]["category"]
    if settings.get("product", "metal_nut") != product:
        raise ValueError("AOI and inference configs refer to different products")
    manifest = read_json(ROOT / config["project"]["splits"])
    if manifest["category"] != product or settings["normal_reference"] != manifest["normal_reference"]:
        raise ValueError("Product/reference conflicts with the fixed manifest")
    classification = settings.get('classification', config.get('classification'))
    if product != 'metal_nut' and classification.get('labels') != config['classification']['labels']:
        raise ValueError('Inference descriptions may change; product class IDs must stay fixed')
    settings["qwen"] = product_options(settings["qwen"], product, classification)
    allowed_groups = list(config["patchcore"]["groups"])
    if product == 'capsule' and settings.get('capsule_model_pair', {}).get('enabled'):
        allowed_groups.append('CAPSULE_ROI_MASK_ALIGN')
    if product == 'pill' and settings.get('pill_patchcore', {}).get('enabled'):
        allowed_groups.append('PILL_ROI_ALIGN')
    if require_calibrated and (settings.get("selected_patchcore") not in allowed_groups
            or not settings.get("thresholds")
            or any(settings["thresholds"].get(n, {}).get(k) is None
                   for n in ["efficientad", "patchcore"] for k in ["image", "pixel"])):
        raise RuntimeError(f"{product}: dev calibration is not ready; E0 does not supply thresholds")
    return settings


def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def configure_runtime(config: dict) -> None:
    # Keep all pretrained downloads/caches inside the project.
    os.environ["TORCH_HOME"] = str(ROOT / "models" / "pretrained" / "torch")
    os.environ["HF_HOME"] = str(ROOT / "models" / "pretrained" / "huggingface")
    os.environ["OMP_NUM_THREADS"] = str(config["project"]["cpu_threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["project"]["cpu_threads"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
