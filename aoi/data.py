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


def recipe_tensor(rgb: np.ndarray, recipe: OpenCVRecipe, patchcore: bool):
    image, metadata = recipe(rgb)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
    if patchcore:
        tensor = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor, metadata


class RecipeDataset(MVTecADDataset):
    """Keep Anomalib image/mask parsing; only adapt the image fed to the model.

    Ground-truth masks retain original coordinates. PG predictions must use
    inverse_anomaly_map before comparison to a mask in later stages.
    """

    def __init__(self, config: dict, manifest: dict, group: str, patchcore: bool):
        super().__init__(root=ROOT / config["project"]["data_root"],
                         category=config["project"]["category"], split=None)
        reference = cv2.imread(str(ROOT / manifest["normal_reference"]), cv2.IMREAD_COLOR)
        if reference is None:
            raise ValueError("Normal reference could not be read")
        self.recipe = OpenCVRecipe(group, config, cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
        self.patchcore = patchcore

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        tensor = item.image.detach().cpu()
        array = tensor.permute(1, 2, 0).numpy()
        if np.issubdtype(array.dtype, np.floating):
            array = np.rint(array * 255).clip(0, 255).astype(np.uint8)
        item.image, _ = recipe_tensor(array, self.recipe, self.patchcore)
        return item


def subset_from_paths(dataset: RecipeDataset, paths: list[str]) -> RecipeDataset:
    requested = {str((ROOT / path).resolve()).casefold() for path in paths}
    indices = [index for index, path in enumerate(dataset.samples.image_path)
               if str(Path(path).resolve()).casefold() in requested]
    if len(indices) != len(requested):
        raise ValueError("Manifest and Anomalib dataset paths do not match")
    return dataset.subsample(indices)


class ManifestDataModule(AnomalibDataModule):
    """No automatic split; calibration uses only manifest dev normals."""

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

    def _setup(self, _stage: str | None = None) -> None:
        dataset = RecipeDataset(self.config, self.manifest, self.group, self.patchcore)
        self.train_data = subset_from_paths(dataset, self.manifest["train_normal"])
        self.val_data = subset_from_paths(dataset, self.manifest["dev"]["normal"])
        # Anomalib's setup expects this attribute. It contains calibration normals,
        # and stage A explicitly disallows calling test_dataloader.
        self.test_data = self.val_data
        self._is_setup = True

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

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", True, self.train_batch_size)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", False, self.eval_batch_size)

    def test_dataloader(self):
        raise RuntimeError("Stage A does not run final_test or threshold evaluation")
