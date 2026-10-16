# 資料讀取與模型輸入張量轉換。一般影像為 RGB、H×W×3、uint8，模型輸入為 C×H×W、float。
# 保留訓練與 dev 清單的資料介面；本推論發佈包只帶正常參考圖，未帶完整訓練／dev 影像。
# 幾何前處理只作用於輸入影像；GT 遮罩保留原圖座標，評估前必須把預測圖回映。

"""Manifest filtering and an image-only recipe around Anomalib's MVTec loader."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import normalize

from aoi.common import ROOT, read_json
from aoi.preprocessing import OpenCVRecipe


# 先執行指定 OpenCV 配方，再把 HWC 的 0～255 RGB 轉成 CHW 的 0～1 張量。
# patchcore=True 時套用 ImageNet 均值與標準差；回傳 tensor 與供異常圖回映使用的 metadata。
def recipe_tensor(rgb: np.ndarray, recipe: OpenCVRecipe, patchcore: bool):
    image, metadata = recipe(rgb)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
    if patchcore:
        tensor = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor, metadata


# 沿用 Anomalib 的影像／遮罩解析，只替換交給模型的影像前處理。
# 不要在此改動 GT 遮罩，否則原圖座標的定位評估會失去一致性。
class RecipeDataset(MVTecADDataset):
    """Keep Anomalib image/mask parsing; only adapt the image fed to the model.

    Ground-truth masks retain original coordinates. PG predictions must use
    inverse_anomaly_map before comparison to a mask in later stages.
    """

    # 依產品資料根目錄建立資料集，從固定正常參考圖建立前處理配方。
    # OpenCV 讀入是 BGR，需轉成 RGB 才能與推論及 Qwen 使用的顏色順序一致。
    def __init__(self, config: dict, manifest: dict, group: str, patchcore: bool):
        super().__init__(root=ROOT / config["project"]["data_root"],
                         category=config["project"]["category"], split=None)
        reference = cv2.imread(str(ROOT / manifest["normal_reference"]), cv2.IMREAD_COLOR)
        if reference is None:
            raise ValueError("Normal reference could not be read")
        self.recipe = OpenCVRecipe(group, config, cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
        self.patchcore = patchcore

    # 取出 Anomalib 樣本，將其張量轉回 uint8 RGB，再套用同一組輸入配方。
    # 浮點影像以四捨五入及裁切還原到 0～255；只更新 item.image，其他樣本資訊保留。
    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        tensor = item.image.detach().cpu()
        array = tensor.permute(1, 2, 0).numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = np.rint(array * 255).clip(0, 255).astype(np.uint8)
        item.image, _ = recipe_tensor(array, self.recipe, self.patchcore)
        return item


# 把固定清單內的相對路徑對應到資料集索引。
# Windows 路徑以 resolve＋casefold 比對；缺少影像時拒絕繼續，避免悄悄縮小資料母體。
def subset_from_paths(dataset: RecipeDataset, paths: list[str]) -> RecipeDataset:
    requested = {str((ROOT / path).resolve()).casefold() for path in paths}
    indices = [index for index, path in enumerate(dataset.samples.image_path)
               if str(Path(path).resolve()).casefold() in requested]
    if len(indices) != len(requested):
        raise ValueError("Manifest and Anomalib dataset paths do not match")
    return dataset.subsample(indices)


# 以明確清單取代框架的自動切分，保存訓練配方的資料載入方式。
# 驗證載入器只讀 dev 正常圖供內部校正；此類別禁止以 test_dataloader 執行 final_test。
class ManifestDataModule(AnomalibDataModule):
    """No automatic split; calibration uses only manifest dev normals."""

    # 依 EfficientAD 或 PatchCore 配方設定 batch、workers 與 seed。
    # val_split_mode/test_split_mode 設為 none，並建立空的 loader 快取以重用工作程序。
    def __init__(self, config: dict, group: str = "P0", patchcore: bool = False,
                 batch_size: int | None = None):
        options = config["patchcore" if patchcore else "efficientad"]
        super().__init__(train_batch_size=batch_size or options["batch_size"],
                         eval_batch_size=1, num_workers=options["num_workers"],
                         val_split_mode="none", test_split_mode="none",
                         seed=config["project"]["seed"])
        self.config, self.group, self.patchcore = config, group, patchcore
        self.category = config["project"]["category"]
        self.manifest = read_json(ROOT / config["project"]["splits"])
        self._loaders: dict[str, DataLoader] = {}

    # 建立同一個 RecipeDataset 後，分別套用 train_normal 與 dev.normal 的路徑清單。
    # test_data 指向 val_data 只是滿足 Anomalib 介面，並不代表開放最終測試。
    def _setup(self, _stage: str | None = None) -> None:
        dataset = RecipeDataset(self.config, self.manifest, self.group, self.patchcore)
        self.train_data = subset_from_paths(dataset, self.manifest["train_normal"])
        self.val_data = subset_from_paths(dataset, self.manifest["dev"]["normal"])
        # Anomalib's setup expects this attribute. It contains calibration normals,
        # and stage A explicitly disallows calling test_dataloader.
        self.test_data = self.val_data
        self._is_setup = True

    # 按 train／val 名稱建立並快取 DataLoader。
    # 使用資料集專屬 collate_fn；只有 workers>0 才啟用 persistent_workers 與 prefetch，避免無效參數。
    def _loader(self, name: str, shuffle: bool, batch_size: int) -> DataLoader:
        if name not in self._loaders:
            dataset = getattr(self, name + "_data")
            options = self.config["loader"]
            self._loaders[name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers,
                collate_fn=dataset.collate_fn, pin_memory=options["pin_memory"],
                persistent_workers=options["persistent_workers"] and self.num_workers > 0,
                prefetch_factor=options["prefetch_factor"] if self.num_workers else None,
            )
        return self._loaders[name]

    # 回傳會 shuffle 的正常訓練資料載入器。
    # 訓練 batch 大小沿用初始化時的設定，不在此重新決定切分。
    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True, self.train_batch_size)

    # 回傳不 shuffle 的 dev 正常圖載入器。
    # 固定順序有助於校正紀錄與影像清單核對。
    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False, self.eval_batch_size)

    # 明確阻止訓練資料模組執行最終測試。
    # final_test 需要另走固定評估流程，不能被訓練入口順帶讀取。
    def test_dataloader(self):
        raise RuntimeError("Stage A does not run final_test or threshold evaluation")
