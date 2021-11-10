import loader
import utils
from models import models_list, loss_list

import random
from matplotlib import pyplot as plt
import cv2
import os
import albumentations as A
import numpy as np
import argparse

from models import SelfNet, SimpleNet

import torch

parser = argparse.ArgumentParser(description='configuration_file')
parser.add_argument('-c', '--config', default='/zhome/3b/d/154066/repos/GALIROOT/config/example.json', type=str, help='Configuration .json file')
args = parser.parse_args()

config = utils.open_config(args.config) 
folders, processing, training = utils.first_layer_keys(config)

ann_list, __ = utils.list_files("../data/l515_lab_1410_test/ann", processing['format_ann'])
img_list, __ = utils.list_files("../data/l515_lab_1410_test/img", processing['format_img'])

valid_transform = A.load(os.path.join(config['folders']['augmentations'],"v"+config['processing']['aug_json_name']))
index = random.sample(range(len(img_list)),5)
# index = [1,2,3,4,5]
dataset = loader.KeypointsDataset(config, ann_list, img_list, valid_transform)
subset1 = torch.utils.data.Subset(dataset, index)
data_load = torch.utils.data.DataLoader(subset1)

best_model = models_list[training['model']]
best_model.eval()
best_model.load_state_dict(torch.load(f"../models/{training['checkpoint_name']}/bv_{training['checkpoint_name']}.pt",map_location='cpu'))


plt.figure(figsize=(40,20))
plt.title(f"Best validation loss {training['checkpoint_name']}")
for idx, data in enumerate(data_load):

    eucledian_dist = 0
    plt.subplot(2,5,idx+1)
    plt.axis('off')
    image = data['image']
    keypoints = torch.mul(data['keypoints'],256)
    kp_np = [int(x) for x in keypoints.detach().numpy()[0][0]]
    plt.text(0,270, f"Ground truth: {str(kp_np)}", color = (1,0,0), fontsize = 'xx-large')

    prediction = torch.mul(best_model(image), 256)
    print(prediction)
    pred_np = [int(x) for x in prediction.detach().numpy()[0]]
    plt.text(0,290, f"Prediction: {str(pred_np)}", color = (0,0,1), fontsize = 'xx-large')

    eucledian_dist = np.sqrt(np.power(pred_np[0]-kp_np[0],2)+np.power(pred_np[1]-kp_np[1],2))
    plt.text(130,290, f"Eucl. dist: {int(eucledian_dist)} pixels", color = (0,0,0), fontsize = 'xx-large')
    image_copy = utils.vis_keypoints(image, keypoints, prediction, plot=False)
    plt.imshow(image_copy)
    plt.show()

# plt.savefig(f"../models/{training['checkpoint_name']}/bv__inf_{training['checkpoint_name']}.png")

last_model = models_list[training['model']]
last_model.eval()
last_model.load_state_dict(torch.load(f"../models/{training['checkpoint_name']}/{training['checkpoint_name']}.pt",map_location='cpu'))

# plt.figure(figsize=(64,16))
# plt.title(f"General model {training['checkpoint_name']}")
for idx, data in enumerate(data_load):

    eucledian_dist = 0
    plt.subplot(2,5,idx+6)
    plt.axis('off')
    image = data['image']
    keypoints = torch.mul(data['keypoints'],256)
    kp_np = [int(x) for x in keypoints.detach().numpy()[0][0]]
    plt.text(0,270, f"Ground truth: {str(kp_np)}", color = (1,0,0), fontsize = 'xx-large')

    prediction = torch.mul(last_model(image), 256)
    print(prediction)
    pred_np = [int(x) for x in prediction.detach().numpy()[0]]
    plt.text(0,290, f"Prediction: {str(pred_np)}", color = (0,0,1), fontsize = 'xx-large')

    eucledian_dist = np.sqrt(np.power(pred_np[0]-kp_np[0],2)+np.power(pred_np[1]-kp_np[1],2))
    plt.text(130,290, f"Eucl. dist: {int(eucledian_dist)} pixels", color = (0,0,0), fontsize = 'xx-large')
    image_copy = utils.vis_keypoints(image, keypoints, prediction, plot=False)
    plt.imshow(image_copy)
    plt.show()

plt.title(f"Best validation loss {training['checkpoint_name']}")
plt.savefig(f"../models/{training['checkpoint_name']}/inf_{training['checkpoint_name']}.png")
