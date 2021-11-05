import utils
import loader
from models import models_list, loss_list
from train import training_loop

import numpy as np
from pathlib import Path

import torch



# check for CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
print("Running GPU.") if use_cuda else print("No GPU available.")

# loading configurations 
config = utils.open_config("../config/example.json")
folders, processing, training = utils.first_layer_keys(config)

# create checkpoint file 
utils.create_file(folders['save_checkpoint']+training['checkpoint_name']+".pt") # FIXME: is it checkpoint or save_dict?

# loading the dataset
train_transform, valid_transform = loader.generate_transform(config)

ann_list, __ = utils.list_files(folders['annotations'], processing['format_ann'])
img_list, __ = utils.list_files(folders['images'], processing['format_img'])

# Initialize network model

print('\nTraining loop has started.\n')

loss_dictionary = training_loop(config, device, img_list, ann_list, train_transform, valid_transform)

# utils.plot_losses(loss_dictionary['train_loss'], loss_dictionary['valid_loss'], training['epochs'], folders['save_graph']+training['checkpoint_name']+".png", training['checkpoint_name'])








