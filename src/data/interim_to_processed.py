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
            ],
            keypoint_params=A.KeypointParams(format="xy"),
        )

    def write_to_npy(self):

        for data_path in tqdm(self.input_list):

            data = np.load(data_path)

            # normalization for rgb and depth (tensor)
            rgb, depth, mask = self._normalize(data)

            keypoints = self.mask_to_keypoint(mask)

            # cropping to ROI and convert to (tensor -> numpy)
            cropped = self.transforms(image=rgb, mask=depth, keypoints=keypoints)

            mask = self._keypoint_to_mask(cropped["keypoints"])

            combined = np.concatenate(
                (
                    cropped["image"],
                    np.expand_dims(cropped["mask"], axis=-1),
                    np.expand_dims(mask, axis=-1),
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

    @staticmethod
    def mask_to_keypoint(mask: Any):
        y_list = np.where(mask == 1)[0]
        x_list = np.where(mask == 1)[1]

        keypoints = []

        for y, x in zip(y_list, x_list):
            keypoints.append((x.astype('float32'), y.astype('float32')))

        return keypoints

    def _keypoint_to_mask(self, keypoints: list):
        keypoint_mask = np.zeros(
            (self.config["size"], self.config["size"]), dtype="float32"
        )
        for keypoint in keypoints:
            keypoint_mask[
                round(keypoint[1]), round(keypoint[0])
            ] = 1  # NOTE: losing accuracy on the keypoint
        return keypoint_mask


def main():

    i2p = InterimToProcessed("data/interim", "data/processed", "data_config.yml")
    i2p.write_to_npy()


if __name__ == "__main__":
    main()
