from numpy.core.fromnumeric import transpose
import torch 
import json
from typing import Optional
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import cv2
import os
from albumentations.pytorch import ToTensorV2

from preprocessing import preProcessing


class KeypointsDataset(Dataset):
    def __init__(self):
        self.settings_location = '/home/bbejczy/repos/GALIROOT/config/dataset.json'
        self.settings = self.open_settings()
        self.annotations = self.list_files(self.settings['annotations'], self.settings['annotation_format']) #might not be needed
        self.images = self.list_files(self.settings['images'], self.settings['image_format'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # read ROI from json
        x = self.settings['ROI']['x']
        y = self.settings['ROI']['y']

        # define augmentation pipeline
        transform = A.Compose([
            A.Normalize(mean=0, std=1),
            A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]),
            A.Resize(height = self.settings['size']['height'], width = self.settings['size']['width']),
            # ToTensorV2() # might not be needed?
        ],
        keypoint_params=A.KeypointParams(format='xy')
        )

        with open(self.annotations[idx],"r") as j:
            ann_file = json.load(j)
        if not ann_file['objects']:
            raise Exception
        else:
            kp_raw = ann_file['objects'][0]['points']['exterior'][0]
            keypoints = [(kp_raw[0],kp_raw[1])] # convert from list to tuple
            print(keypoints)
            image = cv2.imread(self.images[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image, keypoints=keypoints)

            return transformed

    def open_settings(self):
        with open(self.settings_location) as j:
            settings = json.load(j)
        return settings

    @staticmethod
    def list_files(directory, fileformat):
        img_list = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                if name.endswith(fileformat):
                    img_list.append(os.path.join(root, name))
        return img_list


def main():

    dataset = KeypointsDataset()

    data_load = torch.utils.data.DataLoader(
        dataset
    )

    for idx, data in enumerate(data_load):
        images = data['image']
        key_pts = data['keypoints']

        print(images.shape, key_pts)


if __name__ == "__main__":
    main()