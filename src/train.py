from numpy.core.fromnumeric import transpose
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import albumentations as albu

import loader
from models import Net

# import utilities to keep workspaces alive during model training
# from workspace_utils import active_session


# directory definition
# img = '/home/bbejczy/repos/GALIROOT/data/20200809_skaevinge_pruned/img/'
# ann = '/home/bbejczy/repos/GALIROOT/data/20200809_skaevinge_pruned/ann/'
model_dir = '/home/bbejczy/repos/GALIROOT/models/'
model_name = 'baseline_v3.pt'
img_name = Path(model_name).stem+".png"


# loading the dataset
Dataset = loader.KeypointsDataset()

data_load = torch.utils.data.DataLoader(
    Dataset
)

print("Dataset is loaded")

# for idx, samples in enumerate(data_load): # verifying the loading
    # print(samples['image'].shape)

net = Net()

print("Network is loaded")

net.eval()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

n_epochs = 1



def train_net(n_epochs):

    # prepare the net for training
    net.train()
    training_loss = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for idx, data in enumerate(data_load):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoint']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()

            if idx % 10 == 9:    # print every 10 batches
                print('Epoch: {}, idx: {}, Avg. Loss: {}'.format(epoch + 1, idx+1, running_loss/10))
                running_loss = 0.0

            training_loss.append(running_loss)


    print('Finished Training')
    return training_loss


# with active_session():
print("Starting training session")
training_loss = train_net(n_epochs)
print(training_loss)
print("Len: ",len(training_loss))

torch.save(net.state_dict(), model_dir+model_name)

# visualize the loss as the network trained
index = np.arange(len(data_load))
plt.figure()
plt.plot(index, training_loss)
plt.grid()
plt.xlabel('Index')
plt.ylabel('Loss')
plt.savefig(model_dir+img_name)




