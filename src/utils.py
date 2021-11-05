import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch

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
    try:
        open(path, 'w').close()
    except OSError:
        print('Failed creating the file')
    else:
        print('File created')

def plot_losses(train_losses, val_losses, epoch, figure, name):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss for {}".format(name))
    plt.plot(range(epoch), val_losses,label="val")
    plt.plot(range(epoch), train_losses,label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(figure)

def plot_losses_kfold(loss_dictionary, epoch, name):
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss for {}".format(name))
    plt.plot(range(epoch), loss_dictionary['train_folds'],label="train")
    plt.plot(range(epoch), loss_dictionary['valid_folds'],label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("/zhome/3b/d/154066/repos/GALIROOT/models/losses/{}.png".format(name)) # FIXME: from config

def normalize_keypoints(keypoints, scale):
    return torch.div(torch.from_numpy(np.asarray(keypoints)),scale)

def index_with_list(img_list, ann_list, index):
    return [img_list[i] for i in index], [ann_list[i] for i in index]

def dump_to_json(file, output):
    with open(output, 'w') as output_file:
        json.dump(file, output_file)

def vis_keypoints(image, keypoints, prediction=None, plot=True):

    '''
    Visualizing keypoints on images.
    
    # FIXME: might not work due to converting to tensors before the dataloader

    '''
    image_denorm = loader.inverse_normalize(image,(0.3399, 0.3449, 0.1555),(0.1296, 0.1372, 0.1044))
    image_denorm = image_denorm.mul_(255)
    image_copy = image_denorm.squeeze().permute(1,2,0).numpy().copy() # get it back from the normalized state
    # image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
    keypoints = keypoints.detach().numpy()
    prediction = prediction.detach().numpy()

    # for idx in keypoints[0]:
    #     cv2.circle(image_copy, (int(idx[0]), int(idx[1])), 5, (255,0,0), -1)
    # if prediction is not None:
    #     for idx in prediction:
    #         cv2.circle(image_copy, (int(idx[0]), int(idx[1])), 5, (255,255,0), -1)

    for kp, pred in zip(keypoints[0].astype('uint8'),prediction.astype('uint8')):
        cv2.circle(image_copy, (int(kp[0]), int(kp[1])), 5, (255,0,0), -1)
        cv2.circle(image_copy, (int(pred[0]), int(pred[1])), 5, (255,255,0), -1)
        cv2.line(image_copy, kp, pred, (255, 255, 255),thickness=1, lineType=1)

    if plot:
        plt.figure(figsize=(16, 16))
        plt.axis('off')

        plt.imshow((image_copy).astype('uint8'))
        plt.show()

    return image_copy.astype('uint8')

def main():

    losses_dict = open_config("/zhome/3b/d/154066/repos/GALIROOT/models/losses/simplenet_0511_1539.json")
    plot_losses_kfold(losses_dict,5, "simplenet_0511_1539")

if __name__ == "__main__":
    main()
