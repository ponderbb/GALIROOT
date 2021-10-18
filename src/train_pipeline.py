import utils
import loader
from models import SelfNet
# import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset


# check for CUDA
use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")

# loading configurations 
train_config = utils.open_config('/zhome/3b/d/154066/repos/GALIROOT/config/train_config.json')
data_config = '/zhome/3b/d/154066/repos/GALIROOT/config/gbar_1_dataset.json'
# create model file 
utils.create_file(train_config['model_folder']+train_config['model_name'])
# config Tensorboard
writer = SummaryWriter()
print("Configurations loaded.")

# loading the dataset
print("Loading the dataset.")
dataset = loader.KeypointsDataset(data_config, transform=True)
print("Dataset loaded.")
train_set, val_set = loader.KeypointsDataset.train_valid_split(data_config)

train_load = torch.utils.data.DataLoader(train_set, batch_size = 4, shuffle=True) # TODO: figure out how to use the batch size
valid_load = torch.utils.data.DataLoader(val_set, batch_size = 4, shuffle=True)
print("Train and validation split loaded.")

# # verifying the loading
# for idx, samples in enumerate(train_load):
#     print(samples['image'].shape)

# Initialize network model
net = SelfNet()
torch.cuda.empty_cache() 
if use_cuda:
    net.cuda()
print(net)
print("\nNetwork is loaded")

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-4, weight_decay=1e-6)

print('\nTraining loop has started.\n')

epochs = 50

def train(epochs, net, train_load, valid_load, criterion, optimizer): # FIXME: what to include or make the whole setup as a class/function

    train_loss_list = []
    valid_loss_list = []
    
    min_valid_loss = np.inf

    for e in range(epochs):
        train_loss = 0.0
        net.train()

        for data in train_load:
            if use_cuda:
                image = data['image'].cuda()
                keypoints = data['keypoints'].squeeze().cuda()
            else:
                image = data['image']
                keypoints = data['keypoints'].squeeze()

            
            optimizer.zero_grad()
            target = net(image)
            loss = criterion(target, keypoints)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        net.eval()
        for data in valid_load:
            if use_cuda:
                image = data['image'].cuda()
                keypoints = data['keypoints'].squeeze().cuda()
            else:
                image = data['image']
                keypoints = data['keypoints'].squeeze()

            
            target = net(image)
            loss = criterion(target, keypoints)
            valid_loss = loss.item()*image.size(0)
    
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_load)} \t\t Validation Loss: {valid_loss / len(valid_load)}')

        train_loss_list.append(train_loss / len(train_load))
        valid_loss_list.append(valid_loss / len(valid_load))

        writer.add_scalar('Loss/train', train_loss / len(train_load), e)
        writer.add_scalar('Loss/valid', valid_loss / len(valid_load), e)
        writer.close()

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), train_config['model_folder']+train_config['model_name'])

    return train_loss_list, valid_loss_list

train_loss, valid_loss = train(epochs, net, train_load, valid_load, criterion, optimizer)

utils.plot_losses(train_loss, valid_loss, epochs, train_config['plot'], Path(train_config['model_name']).stem)








