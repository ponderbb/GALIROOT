# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from src import utils


class RawToInterim:
    """
    Create an intermediate set of data from the images, depth information and annotations in the raw folder.
    Structure them in .npy objects in an (1920x1080x5) array:
    [R channel,
     B channel,
     G channel,
     depth channel,
     annotation]
    """

    def __init__(
        self, input_filepath: str, output_filepath: str, annotation_filepath: str
    ):
        self.input = input_filepath
        self.output = output_filepath
        self.annotation = annotation_filepath
        self.img_format = ".png"
        self.ann_format = ".json"
        self.path_dict = {}  # TODO: naming convention change?
        self.length = 0

    def _image_collector(self) -> Tuple:
        img_list = utils.list_files(self.input, self.img_format)
        depth_list = []
        rgb_list = []

        for path in img_list:
            if "rgb" in Path(path).stem:
                rgb_list.append(path)
            elif "depth" in Path(path).stem:
                depth_list.append(path)
            else:
                raise NameError(
                    "No identifier in image name. Neither <depth> nor <rgb> found in name"
                )

        self.path_dict.update(rgb=rgb_list, depth=depth_list)

    def _annotation_collector(self):
        ann_list = utils.list_files(self.annotation, self.ann_format)
        self.path_dict.update(annotation=ann_list)

    def _path_list_cleaner(self):
        """
        The image names contain a unique 19 digit code for each image.

        TODO: change annotation selection method

        The json file are only included in case the image is annotated (stem is definable).

        Therefore both the depth and rgb image's ID is checked against the annotations ID.
        """

        print("Creating annotation list")
        ann_stem_list = []
        for annotation in tqdm(self.path_dict["annotation"]):
            ann_stem_list.append(str(Path(annotation).stem)[-19:])

        print("Cleaning rgb list")
        for rgb in tqdm(self.path_dict["rgb"]):
            if str(Path(rgb).stem)[-19:] not in ann_stem_list:
                self.path_dict["rgb"].remove(rgb)

        print("Cleaning depth list")
        for depth in tqdm(self.path_dict["depth"]):
            if str(Path(depth).stem)[-19:] not in ann_stem_list:
                self.path_dict["depth"].remove(depth)

        self.length = len(ann_stem_list)

    @staticmethod
    def _get_rgb(rgb_path: str) -> Any:

        rgb_img = np.asarray(Image.open(rgb_path))

        return rgb_img  # HxWxC

    @staticmethod
    def _get_depth(depth_path: str) -> Any:

        depth_img = np.asarray(Image.open(depth_path))  # HxW

        depth_img = np.expand_dims(depth_img, axis=-1)

        return depth_img  # HxWxC

    @staticmethod
    def _get_annotation(annotation_path: str) -> Any:

        keypoint_mask = np.zeros((1080, 1920), dtype="uint8")

        with open(annotation_path, "rb") as j:
            annotation = json.load(j)
        if not annotation["objects"]:
            pass
        else:
            keypoint_list = annotation["objects"][0]["points"]["exterior"]

            for keypoint in keypoint_list:
                # TODO: possibility to di ot with a wider kernel?
                keypoint_mask[keypoint[0], keypoint[1]] = 1

            keypoint_mask = np.expand_dims(keypoint_mask, axis=-1)
            # plt.imsave("keypoint_mask.png", keypoint_mask)
            return keypoint_mask

    def _write_to_npy(self):
        """
        Channels in the .npy:
        0:2 -> RGB image
        3 -> depth image
        4 -> keypoint annotation

        FIXME: for some reason, the combining and saving is super slow
        """

        for idx in tqdm(range(self.length)):
            rgb = self._get_rgb(self.path_dict["rgb"][idx])
            depth = self._get_depth(self.path_dict["depth"][idx])
            annotation = self._get_annotation(self.path_dict["annotation"][idx])

            combined = np.concatenate((rgb, depth, annotation), axis=-1)

            np.save(
                os.path.join(self.output, f"{idx}.npy"), combined
            )  # FIXME: better naming convention
