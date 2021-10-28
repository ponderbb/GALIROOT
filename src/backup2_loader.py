import json
import os
from albumentations import augmentations

from torch import random
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



class KeypointsDataset(Dataset):
    '''
    Loader for supervisely based dataset
    '''

    def __init__(self, config_file):
        self.config = utils.open_config(config_file)

        if self.config['processing']['reuse']:
            print("Reusing previously generated transforms.")
            self.transform = A.load(os.path.join(self.config['folders']['augmentations'],self.config['processing']['load_aug']))
            self.valid_transform = A.load(os.path.join(self.config['folders']['augmentations'],"v"+self.config['processing']['load_aug']))
        else:
            print("Generating new transforms.")
            self.transform, self.valid_transform = self.generate_transform()

        self.annotations_list, __ = utils.list_files(self.config['folders']['annotations'], self.config['processing']['format_ann'])
        self.images_list, __ = utils.list_files(self.config['folders']['images'], self.config['processing']['format_img'])


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

            print(data)

            # # split to training and validation sets
            # train_set, valid_set = self.train_valid_split(data)

            # # carry out transformations
            # train_set = self.transform(image=train_set['image'], keypoints=train_set['keypoints'])
            # valid_set = self.valid_transform(image=valid_set['image'], keypoints=valid_set['keypoints'])

            # # normalize training keypoints

            # train_set['keypoints'] = utils.normalize_keypoints(train_set['keypoints'],self.config['processing']['size']['height'])

            # return train_set, valid_set

    def train_valid_split(self, dataset):

        '''
        Create a validation and a test set from the processed data.
        TODO: might me more effitient way of sharing variables such as config
        '''

        data_length = len(self.images_list)
        train_length = round(data_length*self.config['processing']['train_ratio'])

        train_set, val_set = random_split(dataset,[train_length, data_length-train_length], generator=torch.Generator().manual_seed(self.config['processing']['seeding']))

        print("Training set: {}\nValidation set: {}".format(len(train_set), len(val_set)))

        return train_set, val_set

    # @staticmethod
    # def generate_pickle(config_file):

    #     config = utils.open_config(config_file)

    #     print("Loading the dataset.")
    #     dataset = KeypointsDataset(config_file, transform=True)
    #     print("Dataset loaded.")
    #     train_set, val_set = KeypointsDataset.train_valid_split(config_file)

    #     # Dump the data into a pickle file
    #     with open(config['pickle_pipeline'] + 'train_set.pickle', 'wb') as f:
    #         pickle.dump(train_set, f)
    #     with open(config['pickle_pipeline'] + 'valid_set.pickle', 'wb') as f:
    #         pickle.dump(val_set, f)

    def generate_transform(self):

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

        config = self.config['processing']
        add_augmentation = config['add_augmentation']

        # read ROI from json
        x = config['ROI']['x']
        y = config['ROI']['y']
        train_transform.append(A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]))
        valid_transform.append(A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]))

        if add_augmentation['normalization']:
            train_transform.append(A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044]))

        # resize images
        train_transform.append(A.Resize(height = config['size']['height'], width = config['size']['height']))
        valid_transform.append(A.Resize(height = config['size']['height'], width = config['size']['height']))

        # convert to tensors
        train_transform.append(ToTensorV2())

        # define augmentation pipeline
        train_transform_composed = A.Compose(train_transform, keypoint_params=A.KeypointParams(format='xy'))
        valid_transform_composed = A.Compose(valid_transform, keypoint_params=A.KeypointParams(format='xy'))

        if self.config['processing']['generate_aug_json']:
            print("Generating augmentation json.\n")
            A.save(train_transform_composed, os.path.join(self.config['folders']['augmentations'],config['aug_json_name']))
            A.save(valid_transform_composed, os.path.join(self.config['folders']['augmentations'],"v"+config['aug_json_name']))

        return train_transform_composed, valid_transform_composed

# def vis_keypoints(image, keypoints, prediction=None, plot=True):

#     '''
#     Visualizing keypoints on images.
    
#     # FIXME: might not work due to converting to tensors before the dataloader

#     '''
#     image_copy = image.squeeze().permute(1,2,0).numpy().copy()*256 # get it back from the normalized state

#     keypoints = keypoints.detach().numpy()
#     prediction = prediction.detach().numpy()

#     for idx in keypoints[0]:
#         cv2.circle(image_copy, (int(idx[0]), int(idx[1])), 5, (255,0,0), -1)
#     if prediction is not None:
#         for idx in prediction:
#             cv2.circle(image_copy, (int(idx[0]), int(idx[1])), 5, (0,0,0), -1)
    
#     if plot:
#         plt.figure(figsize=(16, 16))
#         plt.axis('off')
#         plt.imshow((image_copy).astype('uint8'))
#         plt.show()

#     return image_copy.astype('uint8')

def main():

    ### UNIT TEST

    config = '../config/example.json'
    # generate_transform_json(config)
    train_set = KeypointsDataset(config)
    data_load = torch.utils.data.DataLoader(
        train_set
    )
    # print(train_load)
    # # For visualization
    for idx, data in enumerate(data_load):
        image = data['image']
        keypoints = data['keypoints']
    #     vis_keypoints(image, keypoints)
    # train_set, val_set = dataset.train_valid_split(config)
    print(len(data_load))
if __name__ == "__main__":
    main()