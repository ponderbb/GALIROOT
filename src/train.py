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

use_cuda = torch.cuda.is_available()
print("Running GPU.") if use_cuda else print("No GPU available.")


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


# directory definition
# img = '/home/bbejczy/repos/GALIROOT/data/20200809_skaevinge_pruned/img/'
# ann = '/home/bbejczy/repos/GALIROOT/data/20200809_skaevinge_pruned/ann/'
model_dir = '/zhome/3b/d/154066/repositories/GALIROOT/models/'
model_name = 'baseline_v6.pt'
img_name = Path(model_name).stem+".png"

print("Loading dataset \n")
# loading the dataset
Dataset = loader.KeypointsDataset()

data_load = torch.utils.data.DataLoader(
    Dataset,
    batch_size=1,
    shuffle = True
)

print("Dataset is loaded")

# for idx, samples in enumerate(data_load): # verifying the loading
    # print(samples['image'].shape)

net = Net()
torch.cuda.empty_cache() 
if use_cuda:
    net.cuda()
print(net)

print("Network is loaded")

net.eval()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-4)

n_epochs = 10



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
            key_pts = [data['keypoints'][0][0]/256, data['keypoints'][0][0]/256]

            # print('Key points', key_pts)

            # flatten pts
            key_pts = torch.FloatTensor(key_pts)
            key_pts = key_pts.view(key_pts.size(0), -1)

            # key_pts = torch.flatten(key_pts)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            if use_cuda:
                key_pts = key_pts.cuda()
                images = images.cuda()

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
index = np.arange(len(data_load)*n_epochs)
plt.figure()
plt.plot(index, training_loss)
plt.grid()
plt.xlabel('Index')
plt.ylabel('Loss')
plt.savefig(model_dir+img_name)




