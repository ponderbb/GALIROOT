import os
import pickle
from typing import Dict, Tuple

import albumentations as A
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from src import utils


class MeanStdDataset(Dataset):
    def __init__(self, input_path: str, config_path: str) -> Dict:
        self.config = utils.load_config(config_path)
        self.transforms = A.Compose(
            [
                A.Crop(
                    self.config["x_min"],
                    self.config["y_min"],
                    self.config["x_max"],
                    self.config["y_max"],
                )
            ]
        )
        self.input_path = input_path
        self.input_list = utils.list_files(input_path, ".npy")

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):

        data = np.load(self.input_list[idx])

        image, depth = self._disassemble(data)

        # NOTE: here can the depth be segmented
        # depth = self._remap(depth, 300, 500)
        depth = self._clip(
            depth,
            self.config["z_min"],
            self.config["z_max"],
        )

        # NOTE: might be redundant to remap before normalizing
        # image = self._remap(image, image.min(), image.max())

        data_processed = self._assemble(image, depth)

        return self.transforms(image=data_processed)

    @staticmethod
    def _remap(array, min: int, max: int):
        return (array - min) / (max - min)

    @staticmethod
    def _clip(array, min: int, max: int):
        return array.clip(min, max)

    @staticmethod
    def _disassemble(data) -> Tuple:
        image = data[:, :, 0:3]
        depth = data[:, :, 3]
        return image, depth

    @staticmethod
    def _assemble(image, depth):
        depth = np.expand_dims(depth, axis=-1)
        return np.concatenate((image, depth), axis=-1)

    def calculate_mean_std(self, dataset) -> Dict:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["ms_batch_size"],
            num_workers=0,
            shuffle=False,
        )

        rgb_mean, rgb_std, depth_mean, depth_std, nb_samples = 0, 0, 0, 0, 0
        for data in tqdm(loader):
            batch_samples = data["image"].size(0)

            rgb = data["image"][:, :, :, 0:3]
            depth = data["image"][:, :, :, 3].unsqueeze(-1)

            rgb = rgb.reshape(batch_samples, rgb.size(-1), -1)
            rgb_float = rgb.type(torch.FloatTensor)
            depth = depth.reshape(batch_samples, -1)
            depth_float = depth.type(torch.FloatTensor)

            rgb_mean += rgb_float.mean(2).sum(0)
            rgb_std += rgb_float.std(2).sum(0)

            depth_mean += depth_float.mean(1).sum(0)
            depth_std += depth_float.std(1).sum(0)

            nb_samples += batch_samples

        rgb_mean /= nb_samples
        rgb_std /= nb_samples
        depth_mean /= nb_samples
        depth_std /= nb_samples

        out_dict = {
            "rgb_mean": list(rgb_mean.numpy()),
            "rgb_std": list(rgb_std.numpy()),
            "depth_mean": round(float(depth_mean.numpy()), 8),
            "depth_std": round(float(depth_std.numpy()), 8),
        }

        with open(os.path.join(self.input_path, "mean_std.pickle"), "wb") as outfile:
            pickle.dump(out_dict, outfile)

        return out_dict


def load_mean_std(input_path: str) -> Dict:
    with open(os.path.join(input_path, "mean_std.pickle"), "rb") as infile:
        return pickle.load(infile)


def main():

    msd = MeanStdDataset("data/interim", "src/data/data_config.yml")
    msd.calculate_mean_std(msd)

    print("done")


if __name__ == "__main__":
    main()
