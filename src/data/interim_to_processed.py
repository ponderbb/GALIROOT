import os
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from tqdm import tqdm

from src import utils
from src.data.mean_std_dataset import load_mean_std


class InterimToProcessed:
    def __init__(self, input_path: str, output_path: str, config_path: str):
        self.mean_std_dict = load_mean_std(input_path)
        self.input_list = utils.list_files(input_path, ".npy")
        self.output_path = output_path
        self.rgb_norm = self._generate_rgb_norm()
        self.depth_norm = self._generate_depth_norm()
        self.config = utils.load_config(config_path)
        self.transforms = A.Compose(
            [
                A.Crop(
                    self.config["x_min"],
                    self.config["y_min"],
                    self.config["x_max"],
                    self.config["y_max"],
                ),
                A.Resize(self.config["size"], self.config["size"]),
            ]
        )

    def write_to_npy(self):

        for data_path in tqdm(self.input_list):

            data = np.load(data_path)

            # normalization for rgb and depth (tensor)
            rgb, depth, mask = self._normalize(data)

            # cropping to ROI and convert to (tensor -> numpy)
            cropped = self.transforms(image=rgb, masks=[depth, mask])

            combined = np.concatenate(
                (
                    cropped["image"],
                    np.expand_dims(cropped["masks"][0], axis=-1),
                    np.expand_dims(cropped["masks"][1], axis=-1),
                ),
                axis=-1,
            )

            np.save(
                os.path.join(self.output_path, f"{Path(data_path).stem}.npy"),
                combined,
            )

    def _clip_depth(self, data: Any):
        return data[:, :, 3].clip(self.config["z_min"], self.config["z_max"])

    def _normalize(self, data: Any):
        rgb = self.rgb_norm(image=data[:, :, 0:3])
        depth = self.depth_norm(image=self._clip_depth(data))
        mask = data[:, :, 4]
        return rgb["image"], depth["image"], mask

    def _generate_rgb_norm(self):
        return A.Compose(
            [
                A.Normalize(
                    mean=self.mean_std_dict["rgb_mean"],
                    std=self.mean_std_dict["rgb_std"],
                    max_pixel_value=1.0,
                )
            ]
        )

    def _generate_depth_norm(self):
        return A.Compose(
            [
                A.Normalize(
                    mean=self.mean_std_dict["depth_mean"],
                    std=self.mean_std_dict["depth_std"],
                    max_pixel_value=1.0,
                )
            ]
        )


def main():

    i2p = InterimToProcessed("data/interim", "data/processed", "data_config.yml")
    i2p.write_to_npy()


if __name__ == "__main__":
    main()
