"""Compatibility hooks only. All model/loss/coreset/NN code stays in Anomalib."""
from __future__ import annotations

from unittest.mock import patch

from anomalib.models import EfficientAd, Patchcore
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from torch.utils.data import DataLoader

from aoi.common import ROOT


class ProjectEfficientAd(EfficientAd):
    def __init__(self, imagenette_workers: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.imagenette_workers = imagenette_workers
        self.save_hyperparameters("imagenette_workers")

    def configure_optimizers(self):
        result = super().configure_optimizers()
        scheduler = result["lr_scheduler"]
        if isinstance(scheduler, dict):
            scheduler = {**scheduler, "interval": "step", "frequency": 1}
        else:
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        result["lr_scheduler"] = scheduler
        return result

    def prepare_pretrained_model(self) -> None:
        # Version 2.7 has no public teacher-cache argument. Redirect only its path
        # provider, while reusing its download, verification and weight loading.
        directory = ROOT / "models" / "pretrained" / "anomalib"
        directory.mkdir(parents=True, exist_ok=True)
        with patch("anomalib.models.image.efficient_ad.lightning_model.get_pretrained_weights_dir",
                   return_value=directory):
            super().prepare_pretrained_model()

    def prepare_imagenette_data(self, image_size) -> None:
        super().prepare_imagenette_data(image_size)
        self.imagenet_loader = DataLoader(
            self.imagenet_loader.dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.imagenette_workers, pin_memory=True,
            persistent_workers=self.imagenette_workers > 0,
            prefetch_factor=2 if self.imagenette_workers else None,
        )
        self.imagenet_iterator = iter(self.imagenet_loader)


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


def make_patchcore(config: dict) -> Patchcore:
    options = config["patchcore"]
    return Patchcore(
        backbone=options["backbone"], layers=options["layers"],
        pre_trained=options["pre_trained"],
        coreset_sampling_ratio=options["coreset_sampling_ratio"],
        num_neighbors=options["num_neighbors"], precision="float32",
        pre_processor=False, post_processor=False, evaluator=False, visualizer=False,
    )
