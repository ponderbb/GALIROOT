from typing import List, Tuple, Optional

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

from src import utils
from src.features.generate_transforms import generate_transform


class RGBDDataset(Dataset):
    def __init__(self, input_path: str, transform: A.Compose = None):
        self.input_list = utils.list_files(input_path, ".npy")
        self.transform = transform
        self.item_length = len(self.input_list)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        data = np.load(self.input_list[idx])
        rgb, depth, mask1 = self._disassemble(data.astype("float32"))
        if self.transform:
            transformed = self.transform(image=rgb, masks=[depth, mask1])
            rgb, depth, mask1 = self._disassemble_transform(transformed)
        return (
            rgb.transpose(2, 0, 1),
            depth.transpose(2, 0, 1),
            mask1.transpose(2, 0, 1),
        )
        # return rgb, torch.from_numpy(depth.transpose(2, 0, 1)), torch.from_numpy(mask1.transpose(2, 0, 1)) # FIXME: issue ith ToTensorV2

    @staticmethod
    def _disassemble(data) -> Tuple:
        rgb = data[:, :, 0:3]
        depth = np.expand_dims(data[:, :, 3], axis=-1)
        mask1 = np.expand_dims(data[:, :, 4], axis=-1)
        return rgb, depth, mask1

    @staticmethod
    def _disassemble_transform(transformed) -> List:
        return transformed["image"], transformed["masks"][0], transformed["masks"][1]

    def calculate_splits(self, train_percentage: float = 0.8) -> Tuple:
        train_size = round(self.item_length * train_percentage)
        if (train_size % 2) != (self.item_length % 2):
            train_size += 1
        valid_size = round((self.item_length - train_size) / 2)
        test_size = round((self.item_length - train_size) / 2)
        return [train_size, valid_size, test_size]


class MyLittleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/processed",
        batch_size: int = 16,
        train_split: float = 0.8,
        transform_config: str = "src/features/transform_config.yml",
    ):  # FIXME: remove hard defined config from there
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.transform = generate_transform(transform_config)

    def setup(self, stage: Optional[str] = None):
        self.data_full = RGBDDataset(self.data_dir, self.transform)
        self.data_train, self.data_valid, self.data_test = random_split(
            self.data_full,
            self.data_full.calculate_splits(train_percentage=self.train_split),
        )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_valid, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)


def main():

    # TODO: testing delete

    utils.set_seed(42)
    dm = MyLittleDataModule()
    dm.setup()

    for rgb, depth, mask1 in dm.train_dataloader():
        rgb_np = rgb[0, :, :, :].numpy()
        rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min())
        plt.imsave("src/visualization/image_tests/rgb.png", rgb_np)
        plt.imsave("src/visualization/image_tests/depth.png", depth[0, :, :].numpy())
        plt.imsave("src/visualization/image_tests/label.png", mask1[0, :, :].numpy())


if __name__ == "__main__":

    main()
