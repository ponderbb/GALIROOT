from numpy.core.fromnumeric import transpose

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

from pre_processing import preProcessing

# TODO:clean out imports

import json
import utils
import albumentations as A
import cv2
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset




class KeypointsDataset(Dataset):

    '''
    Loader for supervisely based dataset
    '''

    def __init__(self, config_file, transform=None):
        self.config = utils.open_config(config_file)
        self.iftransform = transform
        self.transform = A.load(self.config['aug_pipeline'])
        self.annotations_list, __ = utils.list_files(self.config['annotations'], self.config['annotation_format'])
        self.images_list, __ = utils.list_files(self.config['images'], self.config['image_format'])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        with open(self.annotations_list[idx],"r") as j:
            annotation = json.load(j)
        if not annotation['objects']:
            raise Exception
        else:
            kp_raw = annotation['objects'][0]['points']['exterior'][0]
            keypoints = [(kp_raw[0],kp_raw[1])] # tuple wrapped in a list
            image = cv2.imread(self.images_list[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # print("Pre-transform keypoint {} and image shape: {}".format(keypoints, image.shape))
            data = {"image":image, "keypoints":keypoints}

            if self.iftransform:
                data = self.transform(image=data['image'], keypoints=data['keypoints'])
                # print("Post-transform keypoint {} and image shape: {}".format(data['keypoints'], data['image'].shape))

            return data


def generate_transform_json(config):

    '''
    Generate albumentations pipeline json file.
    TODO: implement functionality for specifying the albumentations added to the pipeline
    '''

    config_json = utils.open_config(config)

    # read ROI from json
    x = config_json['ROI']['x']
    y = config_json['ROI']['y']

    # define augmentation pipeline
    transform = A.Compose([
        A.Normalize(mean=0, std=1),
        A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]),
        A.Resize(height = config_json['size']['height'], width = config_json['size']['height']),
    ],
    keypoint_params=A.KeypointParams(format='xy')
    )

    A.save(transform, config_json['aug_pipeline'])

def vis_keypoints(image, keypoints, color=(0, 255, 0), diameter=10):

    '''
    Visualizing keypoints on images.
    '''

    image = image.detach().numpy().squeeze().copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
        
    plt.figure(figsize=(32, 32))
    plt.axis('off')
    plt.imshow((image*256).astype('uint8'))
    plt.show()

def main():

    ### UNIT TEST

    config = '/home/bbejczy/repos/GALIROOT/config/dataset.json'

    generate_transform_json(config)

    dataset = KeypointsDataset(config, transform=True)

    data_load = torch.utils.data.DataLoader(
        dataset
    )

    # For visualization

    for idx, data in enumerate(data_load):
        image = data['image']
        keypoints = data['keypoints']
        vis_keypoints(image, keypoints)

if __name__ == "__main__":
    main()