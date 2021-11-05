import loader
import utils
from models import models_list, loss_list

import random
from matplotlib import pyplot as plt
import cv2
import os
import albumentations as A

from models import SelfNet, SimpleNet

import torch



def main(): # TODO: clean up inference and creat config

    # loading configurations 
    config = utils.open_config("../config/example.json")
    folders, processing, training = utils.first_layer_keys(config)
    ann_list, __ = utils.list_files("/zhome/3b/d/154066/repos/GALIROOT/data/l515_lab_1410_test/ann", processing['format_ann'])
    img_list, __ = utils.list_files("/zhome/3b/d/154066/repos/GALIROOT/data/l515_lab_1410_test/img", processing['format_img'])
    valid_transform = A.load(os.path.join(config['folders']['augmentations'],"v"+config['processing']['load_aug']))
    dataset = loader.KeypointsDataset(config, ann_list, img_list, valid_transform)
    index = random.sample(range(20),10)
    subset1 = torch.utils.data.Subset(dataset, index)
    data_load = torch.utils.data.DataLoader(subset1)
    
    # load model
    model = models_list[training['model']]
    model.eval()
    model.load_state_dict(torch.load('../models/checkpoint/simplenet_0511_1539.pt',map_location='cpu'))
    
    sum_kp = torch.zeros(2)
    name = 'simplenet_0511_1539_100epoch'
    plt.figure(figsize=(64,32))
    plt.title(name)
    for idx, data in enumerate(data_load):
        plt.subplot(2,5,idx+1)
        plt.axis('off')
        image = data['image']
        keypoints = torch.mul(data['keypoints'],256)
        plt.text(0,270, f"Ground truth {str(keypoints.detach().numpy().astype('uint8')[0][0])}", color = (0,0,0), fontsize = 'xx-large')
        sum_kp = sum_kp.add(keypoints)
        prediction = torch.mul(model(image), 256)
        plt.text(0,280, f"Prediction {str(prediction.detach().numpy().astype('uint8')[0])}", color = (0,0,0), fontsize = 'xx-large')
        print(prediction)
        image_copy = utils.vis_keypoints(image, keypoints, prediction, plot=False)
        plt.imshow(image_copy)



    plt.savefig('../models/inference/{}.png'.format(name)) #TODO: this into a config

if __name__ == "__main__":
    main()