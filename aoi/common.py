# 共用設定、路徑及成果身分管理。所有專案相對路徑以本檔位置推導的 ROOT 為基準。
# YAML 決定產品、模型配方、前處理與門檻；SHA256 用來辨認設定或資料切分是否已改變。
# 設定讀取與模型執行分離：這些工具不會自行訓練或下載模型。

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import yaml

# 從原始碼位置計算根目錄，讓使用者換資料夾名稱後仍可解析專案相對路徑。
ROOT = Path(__file__).resolve().parents[1]


# 遞迴合併父設定與覆寫設定，回傳新的最外層字典。
# 兩邊同為字典才繼續往下合併；清單、數值及字串由 override 整個取代。
def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


# 讀取 UTF-8 YAML，解析 inherits 並依宣告順序套用父設定。
# _parents 記錄目前繼承鏈，遇到循環即失敗；inherits 以 ROOT 解析，而非子 YAML 所在目錄。
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


# 依產品的 project.output_root 組出輸出路徑。
# suffix 可指定檔案或子目錄；本函式只回傳 Path，不建立目錄。
def output_path(config: dict, suffix: str = "") -> Path:
    return ROOT / config["project"]["output_root"] / suffix


# 取得特定偵測器的模型目錄。
# EfficientAD 使用 efficientad/；PatchCore 使用 patchcore/<組別>/，metal_nut 可沿用預設 models/。
def model_directory(config: dict, name: str) -> Path:
    base = ROOT / config["project"].get("model_root", "models")
    return base / ("efficientad" if name == "efficientad" else f"patchcore/{name}")


# 取得產品實際採用的推論 YAML。
# 未覆寫時回到 configs/inference.yaml；其他產品可在 AOI 設定指定自己的檔案。
def inference_path(config: dict) -> Path:
    return ROOT / config["project"].get("inference_config", "configs/inference.yaml")


# 建立產品、split、完整設定及切分檔內容的身分標記。
# 設定使用排序後的 JSON 計算雜湊；切分直接雜湊原始檔案位元組，供成果一致性核對。
def artifact_identity(config: dict, split: str = "dev") -> dict:
    return {"product": config["project"]["category"], "split": split,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "splits_sha256": hashlib.sha256((ROOT / config["project"]["splits"]).read_bytes()).hexdigest()}


# 只對影響偵測器的配方計算 SHA256。
# 排除 inference_config 路徑，讓僅修改 Qwen 提示的操作不會誤判偵測器訓練設定已改變。
def detector_config_sha256(config: dict) -> str:
    # Prompt edits must not invalidate identical detector weights/maps.
    project = {k:v for k,v in config["project"].items() if k != "inference_config"}
    settings = {"project": project, **{k:config[k] for k in ["efficientad","patchcore","preprocessing","loader"]}}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


# 載入推論門檻及 Qwen 設定，核對產品、正常參考圖、類別 ID 是否與固定清單一致。
# 再由 product_options 補齊提示及輸出文法；require_calibrated=True 時拒絕缺少門檻或未採用的組別。
def load_inference(config: dict, path: str | Path | None = None, require_calibrated: bool = False) -> dict:
    from aoi.velm import product_options
    settings = load_config(path or inference_path(config))
    product = config["project"]["category"]
    if settings.get("product", "metal_nut") != product:
        raise ValueError("AOI and inference configs refer to different products")
    manifest = read_json(ROOT / config["project"]["splits"])
    if manifest["category"] != product or settings["normal_reference"] != manifest["normal_reference"]:
        raise ValueError("Product/reference conflicts with the fixed manifest")
    # 優先採推論 YAML 的描述覆寫，但類別 ID 不可任意重排，否則模型答案將對到錯誤瑕疵。
    classification = settings.get('classification', config.get('classification'))
    if product != 'metal_nut' and classification.get('labels') != config['classification']['labels']:
        raise ValueError('Inference descriptions may change; product class IDs must stay fixed')
    settings["qwen"] = product_options(settings["qwen"], product, classification)
    # 一般組別與已啟用的 capsule／pill 專屬 ROI 組別分開列舉，避免跨產品誤用模型。
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


# 以 UTF-8 讀取 JSON，交由呼叫端解讀其結構。
# 缺檔或格式錯誤會直接拋出例外，不自行填入可能錯誤的預設模型資訊。
def read_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


# 以可讀的 UTF-8 JSON 保存資料，先寫同目錄的 .tmp 再替換目的檔。
# ensure_ascii=False 保留中文；替換方式可減少讀取端取得半份 JSON 的機會。
def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# 在載入模型前設定套件快取位置及 CPU 執行緒上限。
# 快取集中在專案內；TOKENIZERS_PARALLELISM 關閉 tokenizer 額外平行工作，離線限制另由 offline 管理。
def configure_runtime(config: dict) -> None:
    # Keep all pretrained downloads/caches inside the project.
    os.environ["TORCH_HOME"] = str(ROOT / "models" / "pretrained" / "torch")
    os.environ["HF_HOME"] = str(ROOT / "models" / "pretrained" / "huggingface")
    os.environ["OMP_NUM_THREADS"] = str(config["project"]["cpu_threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["project"]["cpu_threads"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
