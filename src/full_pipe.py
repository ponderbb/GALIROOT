import utils
import os
import loader
from models import models_list, loss_list
from inference import inference
from train import training_loop

import argparse

import torch

os.system("python3 -m wandb login 80b496a88f7d314f833c5f7f62963b9c3d58a6d1")
import wandb

parser = argparse.ArgumentParser(description='configuration_file')
parser.add_argument('-c', '--config', default='/zhome/3b/d/154066/repos/GALIROOT/config/example.json', type=str, help='Configuration .json file')
args = parser.parse_args()

# loading configurations 
config = utils.open_config(args.config) 
folders, processing, training = utils.first_layer_keys(config)

# initialize Weights & Biases
wandb.init(project="GALIROOT", name = training['checkpoint_name'], config=config)

# check for CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
print("Running GPU.") if use_cuda else print("No GPU available.")

# create checkpoint file 
utils.create_folder(folders['out_folder']+'temp/')
# utils.create_file(folders['out_folder']+training['checkpoint_name']+".pt")

# loading the dataset
train_transform, valid_transform = loader.generate_transform(config)

ann_list, __ = utils.list_files(folders['annotations'], processing['format_ann'])
img_list, __ = utils.list_files(folders['images'], processing['format_img'])

# Initialize network model

print('\nTraining loop has started.\n')

loss_dictionary = training_loop(config, device, img_list, ann_list, train_transform, valid_transform)

# plot losses

print("\nPlotting losses.") 

# utils.plot_epoch_losses(config, loss_dictionary)
# utils.plot_kfold_losses(config, loss_dictionary)

utils.cleanup_models(loss_dictionary, training, folders)

inference(config, loss_dictionary, device)








