import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I # for weight initialization
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

import utils

class SelfNet(nn.Module):

    def __init__(self):
        super(SelfNet, self).__init__()

        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding (by default 0)
        # W - the width/height (square) of the previous layer

        # output = (W-F+2P)/S+1

        self.conv1 = nn.Conv2d(3, 32, 5) # 315 385
        self.conv2 = nn.Conv2d(32, 64, 3) # 156 191
        self.conv3 = nn.Conv2d(64, 128, 3) # 77 94
        self.conv4 = nn.Conv2d(128, 256, 3) # 37 46
        self.conv5 = nn.Conv2d(256, 512, 1) # 18 23

        self.norm1 = nn.BatchNorm2d(num_features=32)
        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.norm3 = nn.BatchNorm2d(num_features=128)
        self.norm4 = nn.BatchNorm2d(num_features=256)
        self.norm5 = nn.BatchNorm2d(num_features=512)

        self.pool = nn.MaxPool2d(2,2)

        # Fully connected layers
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 8)
        self.fc9 = nn.Linear(8, 2)
        # output should correspond to desired amount of keypoints (x,y)

        self.dropout = nn.Dropout(p=0.25)
        self.normf1 = nn.BatchNorm1d(num_features=1024)
        self.normf2 = nn.BatchNorm1d(num_features=512)
        self.normf3 = nn.BatchNorm1d(num_features=256)
        self.normf4 = nn.BatchNorm1d(num_features=128)
        self.normf5 = nn.BatchNorm1d(num_features=32)

    def forward(self, x):
        # x = x.type(torch.FloatTensor)
        x = x.float()
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = self.pool(F.relu(self.norm4(self.conv4(x))))
        x = self.pool(F.relu(self.norm5(self.conv5(x))))

        # Prep for linear layer / Flatten
        x = x.reshape(x.shape[0], -1)

        # linear layers with dropout in between 

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)

        return x.type(torch.float64)

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()

        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - the width/height (square) of the previous layer

        # Since there are F*F*D weights per filter
        # The total number of weights in the convolutional layer is K*F*F*D

        # 224 by 224 pixels

        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (1, 220, 220)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(3, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # 220/2 = 110
        # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        # the output Tensor for one image, will have the dimensions: (32, 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        # the output Tensor for one image, will have the dimensions: (64, 54, 54)
        self.conv3 = nn.Conv2d(64, 128, 3)

        # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        # the output Tensor for one image, will have the dimensions: (128, 26, 26)
        self.conv4 = nn.Conv2d(128, 256, 3)

        # output size = (W-F)/S +1 = (12-3)/1 + 1 = 10
        # the output Tensor for one image, will have the dimensions: (256, 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1)

        # output size = (W-F)/S +1 = (6-1)/1 + 1 = 6
        # the output Tensor for one image, will have the dimensions: (512, 6, 6)

        # Fully-connected (linear) layers
        self.fc1 = nn.Linear(25088, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, 2)
        # Dropout
        self.dropout = nn.Dropout(p=0.25)


    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        # 5 conv/relu + pool layers
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)

        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x.type(torch.float64)    

class FaceKeypointResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(FaceKeypointResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        self.l0 = nn.Linear(2048, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64,32)
        self.l6 = nn.Linear(32,16)
        self.l7 = nn.Linear(16,8)
        self.l8 = nn.Linear(8,4)
        self.l9 = nn.Linear(4,2)
        self.dropout = nn.Dropout(p=0.25)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        # x = x.permute(0,3,1,2)
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        return x.type(torch.float64)

class FaceKeypointResNet50_dropout(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(FaceKeypointResNet50_dropout, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        self.l0 = nn.Linear(2048, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64,2)
        self.dropout = nn.Dropout(p=0.25)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        # x = x.permute(0,3,1,2)
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(F.relu(self.l0(x)))
        x = self.dropout(F.relu(self.l1(x)))
        x = self.dropout(F.relu(self.l2(x)))
        x = self.dropout(F.relu(self.l3(x)))
        x = self.l4(x)
        x = self.l5(x)

        return x.type(torch.float64)

class ResNet18(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(ResNet18, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64,32)
        self.l6 = nn.Linear(32,16)
        self.l7 = nn.Linear(16,2)

        self.dropout = nn.Dropout(p=0.25)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        # x = x.permute(0,3,1,2)
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        return x.type(torch.float64)


class ResNet18_v2(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(ResNet18_v2, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet18'](pretrained=None)
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')
        # change the final layer
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64,32)
        self.l6 = nn.Linear(32,16)
        self.l7 = nn.Linear(16,8)
        self.l8 = nn.Linear(8,4)
        self.l9 = nn.Linear(4,2)
        
        self.dropout = nn.Dropout(p=0.25)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(64)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        # x = x.permute(0,3,1,2)
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        return x.type(torch.float64)

def EucledianLoss(prediction, target, device): # FIXME: how to pass this a method (block?)
    ceiling = 10/np.sqrt(2*np.pi)
    distances = []
    for idx in range(len(prediction)):
        eucledian_dist=torch.cdist(prediction[idx].view(1, -1),target[idx].view(1, -1))
        distances.append(eucledian_dist)
    # print(distances)
    distance_tensor = torch.cat(distances, dim=1)
    loss = utils.normal_dist(distance_tensor,0,0.1, device)
    return torch.sub(ceiling,loss)

models_list = [SelfNet(),
               SimpleNet(),
               FaceKeypointResNet50(pretrained=True, requires_grad=True),
               FaceKeypointResNet50_dropout(pretrained=True, requires_grad=True),
               ResNet18(pretrained=True, requires_grad=True),
               ResNet18_v2(pretrained=True, requires_grad=True)]
loss_list = [nn.MSELoss()]



if __name__ == "__main__":

    x = torch.randn(4,4,256,256) # testing the output

    net = models_list[0]

    # print(type(loss_list[0]))
    # print(type(loss_list[1]))

    output = net.forward(x)
    print(output.detach().numpy())


