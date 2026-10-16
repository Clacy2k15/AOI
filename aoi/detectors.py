# Anomalib 偵測器的薄封裝，模型架構、loss、coreset 與最近鄰實作仍由 Anomalib 負責。
# 保留配方建構函式及 EfficientAD 相容性鉤子，以便載入含自訂類別的既有 checkpoint。
# 正常的新圖推論由 inference.Detector 載入已完成的權重，不會呼叫這裡的訓練準備流程。

"""Compatibility hooks only. All model/loss/coreset/NN code stays in Anomalib."""
from __future__ import annotations

from unittest.mock import patch

from anomalib.models import EfficientAd, Patchcore
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from torch.utils.data import DataLoader

from aoi.common import ROOT


# 擴充 EfficientAd 的排程器、teacher 快取位置及 Imagenette workers 控制。
# 此類別名稱也存在 checkpoint 的還原路徑中，因此推論發佈包仍需保留。
class ProjectEfficientAd(EfficientAd):
    # 先讓 Anomalib 建立模型，再保存專案額外的 imagenette_workers 超參數。
    # 其他模型參數由 kwargs 原樣傳給上游，避免重新實作網路。
    def __init__(self, imagenette_workers: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.imagenette_workers = imagenette_workers
        self.save_hyperparameters("imagenette_workers")

    # 沿用上游 optimizer，但讓學習率排程器每個 step 更新一次。
    # 兼容上游回傳排程器物件或字典的兩種形式，與固定更新步數配方保持一致。
    def configure_optimizers(self):
        result = super().configure_optimizers()
        scheduler = result["lr_scheduler"]
        if isinstance(scheduler, dict):
            scheduler = {**scheduler, "interval": "step", "frequency": 1}
        else:
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        result["lr_scheduler"] = scheduler
        return result

    # 將上游 teacher 權重目錄暫時導向專案內的 models/pretrained/anomalib。
    # patch 僅在 with 區塊內生效；下載、驗證與載入細節仍由 Anomalib 執行。
    def prepare_pretrained_model(self) -> None:
        # Version 2.7 has no public teacher-cache argument. Redirect only its path
        # provider, while reusing its download, verification and weight loading.
        directory = ROOT / "models" / "pretrained" / "anomalib"
        directory.mkdir(parents=True, exist_ok=True)
        with patch("anomalib.models.image.efficient_ad.lightning_model.get_pretrained_weights_dir",
                   return_value=directory):
            super().prepare_pretrained_model()

    # 沿用上游建立的 Imagenette dataset，重新設定背景資料的 DataLoader。
    # 保留 batch 與 shuffle，使用指定 workers、pinned memory 及預取，最後重建資料迭代器。
    def prepare_imagenette_data(self, image_size) -> None:
        super().prepare_imagenette_data(image_size)
        self.imagenet_loader = DataLoader(
            self.imagenet_loader.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.imagenette_workers, pin_memory=True,
            persistent_workers=self.imagenette_workers > 0,
            prefetch_factor=2 if self.imagenette_workers else None,
        )
        self.imagenet_iterator = iter(self.imagenet_loader)


# 依 YAML 建立 small EfficientAD 的訓練配方。
# 關閉框架額外的前／後處理、評估器與視覺化，讓專案自行控制原始分數及影像處理。
def make_efficientad(config: dict) -> ProjectEfficientAd:
    options = config["efficientad"]
    return ProjectEfficientAd(
        imagenet_dir=str(ROOT / options["imagenette_dir"]),
        teacher_out_channels=options["teacher_out_channels"], model_size=EfficientAdModelSize.S,
        lr=options["lr"], weight_decay=options["weight_decay"],
        padding=options["padding"], pad_maps=options["pad_maps"],
        imagenette_workers=options["imagenette_workers"],
        pre_processor=False, post_processor=False, evaluator=False, visualizer=False,
    )


# 依 YAML 建立 PatchCore，設定 backbone、取特徵層、coreset 比例及最近鄰數。
# 此建構函式保留 pre_trained 配方；離線 checkpoint 還原則會在 inference 中顯式關閉預訓練下載。
def make_patchcore(config: dict) -> Patchcore:
    options = config["patchcore"]
    return Patchcore(
        backbone=options["backbone"], layers=options["layers"],
        pre_trained=options["pre_trained"],
        coreset_sampling_ratio=options["coreset_sampling_ratio"],
        num_neighbors=options["num_neighbors"], precision="float32",
        pre_processor=False, post_processor=False, evaluator=False, visualizer=False,
    )
