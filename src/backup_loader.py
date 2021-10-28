import json

from torch import random
import utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, random_split



class KeypointsDataset(Dataset):
    '''
    Loader for supervisely based dataset
    '''

    def __init__(self, config_file, transform=False, generate_pickle=False):
        self.config = utils.open_config(config_file)
        self.iftransform = transform
        self.transform = A.load(self.config['aug_pipeline'])
        self.annotations_list, __ = utils.list_files(self.config['annotations'], self.config['annotation_format'])
        self.images_list, __ = utils.list_files(self.config['images'], self.config['image_format'])
        if generate_pickle:
            self.generate_pickle(config_file)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        with open(self.annotations_list[idx],"r") as j:
            annotation = json.load(j)
        if not annotation['objects']:
            raise Exception
        else:
            kp_raw = annotation['objects'][0]['points']['exterior'][0] # FIXME: change here to get mupltipoints in
            keypoints = [(kp_raw[0],kp_raw[1])] # tuple wrapped in a list # FIXME: albumentations weird in format
            image = cv2.imread(self.images_list[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # print("Pre-transform keypoint {} and image shape: {}".format(keypoints, image.shape))
            data = {"image":image, "keypoints":keypoints}

            if self.iftransform:
                data = self.transform(image=data['image'], keypoints=data['keypoints'])
                # print("Post-transform keypoint {} and image shape: {}".format(data['keypoints'], data['image'].shape))
                data['keypoints']= torch.div(torch.from_numpy(np.asarray(data['keypoints'])),256)
            return data

    @staticmethod
    def train_valid_split(config_file, train_ratio=0.81):

        '''
        Create a validation and a test set from the processed data.
        TODO: might me more effitient way of sharing variables such as config
        '''

        dataset = KeypointsDataset(config_file, transform=True)

        data_length = len(dataset.images_list)
        train_length = round(data_length*train_ratio)

        train_set, val_set = random_split(dataset,[train_length, data_length-train_length])

        print("Training set: {}\nValidation set: {}".format(len(train_set), len(val_set)))

        return train_set, val_set

    @staticmethod
    def generate_pickle(config_file):

        config = utils.open_config(config_file)

        print("Loading the dataset.")
        dataset = KeypointsDataset(config_file, transform=True)
        print("Dataset loaded.")
        train_set, val_set = KeypointsDataset.train_valid_split(config_file)

        # Dump the data into a pickle file
        with open(config['pickle_pipeline'] + 'train_set.pickle', 'wb') as f:
            pickle.dump(train_set, f)
        with open(config['pickle_pipeline'] + 'valid_set.pickle', 'wb') as f:
            pickle.dump(val_set, f)


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
        A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]),
        # A.Normalize(mean=0, std=1),
        A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044]),
        A.Resize(height = config_json['size']['height'], width = config_json['size']['height']),
        ToTensorV2(always_apply=True)
    ],
    keypoint_params=A.KeypointParams(format='xy')
    )

    A.save(transform, config_json['aug_pipeline'])

def vis_keypoints(image, keypoints, prediction=None, plot=True):

    '''
    Visualizing keypoints on images.
    
    # FIXME: might not work due to converting to tensors before the dataloader

    '''
    image_copy = image.squeeze().permute(1,2,0).numpy().copy()*256 # get it back from the normalized state

    keypoints = keypoints.detach().numpy()
    prediction = prediction.detach().numpy()

    for idx in keypoints[0]:
        cv2.circle(image_copy, (int(idx[0]), int(idx[1])), 5, (255,0,0), -1)
    if prediction is not None:
        for idx in prediction:
            cv2.circle(image_copy, (int(idx[0]), int(idx[1])), 5, (0,0,0), -1)
    
    if plot:
        plt.figure(figsize=(16, 16))
        plt.axis('off')
        plt.imshow((image_copy).astype('uint8'))
        plt.show()

    return image_copy.astype('uint8')

def main():

    ### UNIT TEST

    config = '/zhome/3b/d/154066/repos/GALIROOT/config/gbar_1_dataset.json'
    # generate_transform_json(config)
    dataset = KeypointsDataset(config, transform=False)
    data_load = torch.utils.data.DataLoader(
        dataset
    )
    # For visualization
    for idx, data in enumerate(data_load):
        image = data['image']
        keypoints = data['keypoints']
    #     vis_keypoints(image, keypoints)
    # train_set, val_set = KeypointsDataset.train_valid_split(config)

if __name__ == "__main__":
    main()