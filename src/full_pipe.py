import utils
import loader
from models import models_list, loss_list
from train import training_loop

import argparse

import torch

parser = argparse.ArgumentParser(description='configuration_file')
parser.add_argument('-c', '--config', default='/zhome/3b/d/154066/repos/GALIROOT/config/example.json', type=str, help='Configuration .json file')
args = parser.parse_args()
print(args.config)

# check for CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
print("Running GPU.") if use_cuda else print("No GPU available.")

# loading configurations 
config = utils.open_config(args.config) 
folders, processing, training = utils.first_layer_keys(config)

# create checkpoint file 
utils.create_folder(folders['out_folder'])
utils.create_file(folders['out_folder']+training['checkpoint_name']+".pt")

# loading the dataset
train_transform, valid_transform = loader.generate_transform(config)

ann_list, __ = utils.list_files(folders['annotations'], processing['format_ann'])
img_list, __ = utils.list_files(folders['images'], processing['format_img'])

# Initialize network model

print('\nTraining loop has started.\n')

loss_dictionary = training_loop(config, device, img_list, ann_list, train_transform, valid_transform)

# plot losses

print("\nPlotting losses.")

utils.plot_epoch_losses(config, loss_dictionary)
utils.plot_kfold_losses(config, loss_dictionary)








