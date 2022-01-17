import json
import os
from albumentations.augmentations.transforms import ColorJitter
from torch.utils.data.dataset import T
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
import torch

import utils



class KeypointsDataset(Dataset):
    '''
    Loader for supervisely based dataset
    '''

    def __init__(self, config, annotations, images, masks, transform, depth=False):
        self.config = config
        self.annotations_list = annotations
        self.images_list = images
        self.mask_list = masks
        self.transform = transform
        self.img_norm = A.Compose([A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044],max_pixel_value=1.0), A.ColorJitter()])
        self.mask_norm = A.Compose([A.Normalize(mean=[0.7188], std=[0.1109],max_pixel_value=1.0)])
        self.depth_channel = depth


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        with open(self.annotations_list[idx],"r") as j:
            annotation = json.load(j)
        if not annotation['objects']:
            raise Exception
        else:
            # Read in values
            kp_raw = annotation['objects'][0]['points']['exterior'][0] # TODO: add option for reading in multipoint 
            keypoints = [(kp_raw[0],kp_raw[1])] # tuple wrapped in a list

            image = cv2.imread(self.images_list[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # NOTE: this is the newly added remapping
            image = np.divide(image,255) 
            
            # print("img_b:", image.min(), image.max())
            norm_img = self.img_norm(image=image)
            # print("img_a:", norm_img['image'][0].min(),norm_img['image'][0].max())

            if self.depth_channel:
                mask = np.asarray(Image.open(self.mask_list[idx]))

                # NOTE: this is the newly added remapping
                mask = (mask-300)/(500-300)
                mask = mask.clip(min=0, max=1) 
                # print("mask_b:", mask.min(), mask.max())

                norm_mask = self.mask_norm(image=mask)
                # print("mask_a: {:.5f} {:.5f}".format(norm_mask['image'].min(),norm_mask['image'].max()))
                norm_mask = np.expand_dims(norm_mask['image'], axis=-1)

                # Concatenate into a 4 channel input
                combined_image = np.concatenate((norm_img['image'], norm_mask), axis=2)

                data = self.transform(image=combined_image, keypoints=keypoints)
            else:
                data = self.transform(image=norm_img['image'], keypoints=keypoints)

            # print("other transforms image", data['image'][0].min(),data['image'][0].max())
            # print("other transforms image", data['image'][3].min(),data['image'][3].max())

            if not data['keypoints']:
                print('Empty keypoint')
                pass
            data['keypoints'] = _normalize_keypoints(data['keypoints'],self.config['processing']['size']['height']) # normalize keypoints on the rescaled image
            return data


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

    if config['processing']['reuse']:
        print("Reusing previously generated transforms.")
        train_transform_composed = A.load(os.path.join(config['folders']['augmentations'],config['processing']['aug_json_name']))
        valid_transform_composed = A.load(os.path.join(config['folders']['augmentations'],"v"+config['processing']['aug_json_name']))
    else:
        print("Generating new transforms.")

        train_transform = []
        valid_transform = []

        processing = config['processing']
        add_augmentation = processing['add_augmentation']

        # read ROI from json
        x = processing['ROI']['x']
        y = processing['ROI']['y']
        train_transform.append(A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]))
        valid_transform.append(A.Crop(x_min=x[0], y_min=y[0], x_max=x[1], y_max=y[1]))

        # add augmentations if specified in the .json
        if add_augmentation['normalization']: # TODO: remove this from config
            # train_transform.append(A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044]))
            # valid_transform.append(A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044]))
            pass

        if add_augmentation['gaussian_blur']:
            train_transform.append(A.GaussianBlur())

        if add_augmentation['channel_dropout']:
            train_transform.append(A.ChannelDropout())

        if add_augmentation['color_jitter']:
            # train_transform.append(A.ColorJitter())
            print('add colorjitter') # FIXME: remove this

        if add_augmentation['vertical_flip']:
            train_transform.append(A.VerticalFlip())
            # valid_transform.append(A.VerticalFlip())

        if add_augmentation['horizontal_flip']:
            train_transform.append(A.HorizontalFlip())
            # valid_transform.append(A.HorizontalFlip())

        if add_augmentation['safe_rotate']:
            train_transform.append(A.Rotate(p=0.75))
            # valid_transform.append(A.SafeRotate(border_mode=0))

        # resize images
        train_transform.append(A.Resize(height = processing['size']['height'], width = processing['size']['height']))
        valid_transform.append(A.Resize(height = processing['size']['height'], width = processing['size']['height']))

        # convert to tensors
        train_transform.append(ToTensorV2())
        valid_transform.append(ToTensorV2())

        # define augmentation pipeline
        train_transform_composed = A.Compose(train_transform, keypoint_params=A.KeypointParams(format='xy'))
        valid_transform_composed = A.Compose(valid_transform, keypoint_params=A.KeypointParams(format='xy'))

        # save the generated augmentation
        print("Generating augmentation json.\n")
        A.save(train_transform_composed, os.path.join(config['folders']['augmentations'],processing['aug_json_name']))
        A.save(valid_transform_composed, os.path.join(config['folders']['augmentations'],"v"+processing['aug_json_name']))

    return train_transform_composed, valid_transform_composed

def _normalize_keypoints(keypoints, scale):
    return torch.div(torch.from_numpy(np.asarray(keypoints)),scale)

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
            return tensor


def main():

    ### UNIT TEST

    from torch.utils.data import DataLoader
    from sklearn.model_selection import KFold

    config = utils.open_config('../config/example.json')
    train_transform, valid_transform = generate_transform(config)

    ann_list, __ = utils.list_files(config['folders']['annota   tions'], config['processing']['format_ann'])
    img_list, __ = utils.list_files(config['folders']['images'], config['processing']['format_img'])

    kf = KFold(n_splits=config['training']['kfold'],shuffle=True)

    for train_index, valid_index in kf.split(img_list):

        train_img, train_ann = utils.index_with_list(img_list, ann_list, train_index)
        valid_img, valid_ann = utils.index_with_list(img_list,ann_list,valid_index)

        train_set = KeypointsDataset(config, train_ann, train_img, train_transform)
        valid_set = KeypointsDataset(config, valid_ann, valid_img, valid_transform)

        train_loader = DataLoader(train_set, batch_size=6)
        # for idx, data in enumerate(train_loader):
        #     image = data['image']
        #     keypoints = data['keypoints']
        valid_loader = DataLoader(valid_set, batch_size=6)
        # for idx, data in enumerate(valid_loader):
        #     image = data['image']
        #     keypoints = data['keypoints']



if __name__ == "__main__":
    main()