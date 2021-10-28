import json
import os
from albumentations import augmentations

from torch import random
from train_pipeline import train
import utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold



class KeypointsDataset(Dataset):
    '''
    Loader for supervisely based dataset
    '''

    def __init__(self, config, annotations, images, transform):
        self.config = config
        # self.annotations = 
        # self.images = 


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        with open(self.annotations_list[idx],"r") as j:
            annotation = json.load(j)
        if not annotation['objects']:
            raise Exception
        else:
            kp_raw = annotation['objects'][0]['points']['exterior'][0] # FIXME: change here to get mupltipoints in
            keypoints = [(kp_raw[0],kp_raw[1])] # tuple wrapped in a list -> weird format by albumentations
            image = cv2.imread(self.images_list[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            data = {"image":image, "keypoints":keypoints}

            return image, keypoints


def augmentation_pipeline(config, data):

    config = utils.open_config(config)

    # split to training and validation sets
    train_set, valid_set = train_valid_split(config,data)

    if config['processing']['reuse']:
        print("Reusing previously generated transforms.")
        transform = A.load(os.path.join(config['folders']['augmentations'],config['processing']['load_aug']))
        valid_transform = A.load(os.path.join(config['folders']['augmentations'],"v"+config['processing']['load_aug']))
    else:
        print("Generating new transforms.")
        transform, valid_transform = generate_transform()

    # carry out transformations
    train_set.dataset.transform = transform
    valid_set.dataset.transform = valid_transform

    # normalize training keypoints
    train_set['keypoints'] = utils.normalize_keypoints(train_set['keypoints'],config['processing']['size']['height'])

    return train_set, valid_set


def generate_transform(config):

    '''
    Generate albumentations pipeline json file.
    Static augmentations:
        -crop
        -resize
        -totensor
    Dynamic augmentations:
        -normalization
    '''

    train_transform = []
    valid_transform = []

    

    processing = config['processing']
    add_augmentation = processing['add_augmentation']

    # read ROI from json
    x = processing['ROI']['x']
    y = processing['ROI']['y']
    train_transform.append(A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]))
    valid_transform.append(A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]))

    if add_augmentation['normalization']:
        train_transform.append(A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044]))

    # resize images
    train_transform.append(A.Resize(height = processing['size']['height'], width = processing['size']['height']))
    valid_transform.append(A.Resize(height = processing['size']['height'], width = processing['size']['height']))

    # convert to tensors
    train_transform.append(ToTensorV2())

    # define augmentation pipeline
    train_transform_composed = A.Compose(train_transform, keypoint_params=A.KeypointParams(format='xy'))
    valid_transform_composed = A.Compose(valid_transform, keypoint_params=A.KeypointParams(format='xy'))

    if config['processing']['generate_aug_json']:
        print("Generating augmentation json.\n")
        A.save(train_transform_composed, os.path.join(config['folders']['augmentations'],processing['aug_json_name']))
        A.save(valid_transform_composed, os.path.join(config['folders']['augmentations'],"v"+processing['aug_json_name']))

    return train_transform_composed, valid_transform_composed

def main():

    ### UNIT TEST

    config = utils.open_config('../config/example.json')
    train_transform, valid_transform = generate_transform(config) # FIXME: do transforms within the loop?

    ann_list, __ = utils.list_files(config['folders']['annotations'], config['processing']['format_ann'])
    img_list, __ = utils.list_files(config['folders']['images'], config['processing']['format_img'])

    kf = KFold(n_splits=config['training']['kfold'],shuffle=True)
    print(kf)

    for train_index, test_index in kf.split(img_list):

        train_ann = utils.index_with_list(ann_list,train_index)
        
        print(train_ann)



if __name__ == "__main__":
    main()