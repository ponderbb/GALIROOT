import json
import os
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from torchvision import transforms

import utils



class KeypointsDataset(Dataset):
    '''
    Loader for supervisely based dataset
    '''

    def __init__(self, config, annotations, images, transform):
        self.config = config
        self.annotations_list = annotations
        self.images_list = images
        self.transform = transform


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        with open(self.annotations_list[idx],"r") as j:
            annotation = json.load(j)
        if not annotation['objects']:
            raise Exception
        else:
            kp_raw = annotation['objects'][0]['points']['exterior'][0] # FIXME: change here to get mupltipoints in
            keypoints = [(kp_raw[0],kp_raw[1])] # tuple wrapped in a list
            image = cv2.imread(self.images_list[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data = self.transform(image=image, keypoints=keypoints)
            data['keypoints'] = utils.normalize_keypoints(data['keypoints'],self.config['processing']['size']['height']) # normalize keypoints on the rescaled image
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
        train_transform_composed = A.load(os.path.join(config['folders']['augmentations'],config['processing']['load_aug']))
        valid_transform_composed = A.load(os.path.join(config['folders']['augmentations'],"v"+config['processing']['load_aug']))
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
        if add_augmentation['normalization']:
            train_transform.append(A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044]))
            valid_transform.append(A.Normalize(mean=[0.3399, 0.3449, 0.1555], std=[0.1296, 0.1372, 0.1044]))

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
        if config['processing']['generate_aug_json']:
            print("Generating augmentation json.\n")
            A.save(train_transform_composed, os.path.join(config['folders']['augmentations'],processing['aug_json_name']))
            A.save(valid_transform_composed, os.path.join(config['folders']['augmentations'],"v"+processing['aug_json_name']))

    return train_transform_composed, valid_transform_composed


# def inverse_normalize(image, mean, std):
#     inv_norm = transforms.Normalize(mean=[-x/y for x, y in zip(mean,std)],std=[1/x for x in std])
#     return inv_norm(image)

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

    ann_list, __ = utils.list_files(config['folders']['annotations'], config['processing']['format_ann'])
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