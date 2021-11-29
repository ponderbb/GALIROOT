import json
import math
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch._C import device

import loader

''' GENERAL FUNCTIONS '''

def open_config(path):
    with open(path) as j:
        config_json = json.load(j)
    return config_json

def first_layer_keys(config):
    return config.values()

def list_files(directory, fileformat):
    path_list = []
    name_list = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(fileformat):
                path_list.append(os.path.join(root, name))
                name_list.append(name)
    return sorted(path_list), sorted(name_list)

def create_file(path):
    open(path, 'w').close()

def create_folder(path):
    os.makedirs(path, exist_ok=True)

def plot_epoch_losses(config, loss_dictionary):
    plt.figure(figsize=(10,5))
    plt.title(f"Average loss per epoch for {config['training']['checkpoint_name']}")
    plt.plot(range(1,config['training']['epochs']+1), loss_dictionary['train_epochs'],label="train")
    plt.plot(range(1,config['training']['epochs']+1), loss_dictionary['valid_epochs'],label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config['folders']['out_folder']+f"epoch_loss_{config['training']['checkpoint_name']}.png")

def plot_kfold_losses(config, loss_dictionary):
    plt.figure(figsize=(10,5))
    plt.title(f"Average loss per fold for {config['training']['checkpoint_name']}")
    plt.bar(np.arange(1,config['training']['kfold']+1)-0.1, loss_dictionary['train_folds'],width = 0.2, label="train")
    plt.bar(np.arange(1,config['training']['kfold']+1)+0.1, loss_dictionary['valid_folds'],width = 0.2, label="validation")
    plt.xlabel("Folds")
    plt.xticks(range(1,config['training']['kfold']+1))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config['folders']['out_folder']+f"fold_loss_{config['training']['checkpoint_name']}.png") # FIXME: from config



def index_with_list(img_list, ann_list, mask_list, index):
    return [img_list[i] for i in index], [ann_list[i] for i in index], [mask_list[i] for i in index]

def dump_to_json(file, output):
    with open(output, 'w') as output_file:
        json.dump(file, output_file)

def vis_keypoints(image, keypoints, prediction, distance, mean):

    '''
    Visualizing keypoints on images.
    '''
    image_denorm = loader.inverse_normalize(image,(0.3399, 0.3449, 0.1555),(0.1296, 0.1372, 0.1044))
    image_denorm = image_denorm.mul_(255)
    image_copy = image_denorm.cpu().squeeze().permute(1,2,0).numpy().copy() # get it back from the normalized state

    border_image = make_border(image_copy,distance,mean)

    keypoints = keypoints.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()

    for kp, pred in zip(keypoints[0].astype('uint8'),prediction.astype('uint8')):
        cv2.circle(border_image, (int(kp[0]), int(kp[1])), 5, (255,0,0), -1)
        cv2.circle(border_image, (int(pred[0]), int(pred[1])), 5, (0,0,255), -1)
        cv2.line(border_image, kp, pred, (255, 255, 255),thickness=1, lineType=1)


    return border_image.astype('uint8')

def normal_dist(x , mean , sd, device):
    # x = x.detach().cpu()
    # mean = mean.detach().cpu()
    pi = torch.Tensor([math.pi]).to(device)
    prob_density = 1/(sd*torch.sqrt(2*pi))* torch.exp(-0.5*((x-mean)/(2*sd))**2)
    return torch.mean(prob_density)

def closest_to_average(epoch_average, epochs: list):
    diff = [abs(e-epoch_average) for e in epochs]
    closest_value = diff.index(min(diff))
    return closest_value+1

def cleanup_models(loss_dictionary: dict, training: dict, folders: dict):
    temp_folder = os.path.join(folders['out_folder'], 'temp')
    old_path = os.path.join(temp_folder, "f{}_{}.pt".format(loss_dictionary['average'],training['checkpoint_name']))
    new_path = os.path.join(folders['out_folder'], "{}.pt".format(training['checkpoint_name']))
    print(f"Moving model {training['checkpoint_name']}")
    shutil.move(old_path,new_path)
    print(f'Deleting {temp_folder}')
    shutil.rmtree(temp_folder)

def eucledian_dist(prediction, ground_truth):
    return np.sqrt(np.power(prediction[0]-ground_truth[0],2)+np.power(prediction[1]-ground_truth[1],2))

def make_border(image, distance, mean):
    bordersize = 10
    above = (255,0,0)
    below = (0,255,0)
    if distance>40:
        color = above
    else:
        color = below
    border_image = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=color)
    return border_image