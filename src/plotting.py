import utils
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import cv2
import albumentations as A
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import json
import loader
import torch
from models import models_list

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    print("Running GPU.") if use_cuda else print("No GPU available.")

    boxplot_name_list =[]
    boxplot_list = []
    # configs_list = ['selfnet_1_2','selfnet_ch4_1_3','resnet18_4_2','resnet18_ch4_1_4']
    # configs_name_list = [ 'selfnet_RGB','selfnet_RGB-D', 'resnet18_RGB', 'resnet18_RGB-D']

    configs_list = ['resnet18_4_4', 'resnet18_4_3_np', 'resnet18_ch4_1_4']
    configs_name_list = ['resnet18_RGB','resnet18_RGB_npt', 'resnet18_RGB-D']

    for name, config_name in zip(configs_name_list, configs_list):

        config = utils.open_config(f'/zhome/3b/d/154066/repos/GALIROOT/config/{config_name}.json')

        folders, processing, training = utils.first_layer_keys(config)

        ann_list, __ = utils.list_files("../data/l515_lab_1410_test/ann", processing['format_ann'])
        img_list, __ = utils.list_files("../data/l515_lab_1410_test/img", processing['format_img'])
        mask_list, __ = utils.list_files("../data/l515_lab_1410_test/depth", processing['format_img'])

        train_transform, valid_transform = loader.generate_transform(config)

        dataset = loader.KeypointsDataset(config, ann_list, img_list, mask_list, valid_transform, training['depth'])
        data_load = torch.utils.data.DataLoader(dataset)

        model = models_list[training['model']]
        model.eval()
        model.load_state_dict(torch.load(os.path.join(folders['out_folder'],f"{training['checkpoint_name']}.pt"),map_location=device))


        # Boxplot for distance offsets

        distance_list = []
        prediction_list = []
        ground_truth_list = []
        for data in data_load:
            image = data['image'] # BUG: throws an error when run individually (comment it out)
            keypoints = torch.mul(data['keypoints'],255)
            prediction = torch.mul(model(image), 255)
            keypoints_np = [int(x) for x in keypoints.cpu().detach().numpy()[0][0]]
            prediction_np = [int(x) for x in prediction.cpu().detach().numpy()[0]]
            distance_list.append(utils.eucledian_dist(prediction_np, keypoints_np))


        boxplot_name_list.append(name)
        boxplot_list.append(distance_list)

        print("\n",name)
        print(distance_list)
        print(f'MEAN: {np.mean(distance_list)}')
        print(f'MEDIAN: {np.median(distance_list)}\n')

    plt.title("Model comparison")
    plt.boxplot(boxplot_list)
    # plt.axhline(y=40, color='r', linestyle='-')
    plt.ylabel("Prediction error in pixels")
    plt.xticks([1,2,3],boxplot_name_list)
    plt.savefig(f"../box_comparison.png")

if __name__ == "__main__":
    main()