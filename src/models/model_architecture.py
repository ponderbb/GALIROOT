import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F

class ResNet18_4ch(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(ResNet18_4ch, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        # add own layer
        pretrained_weights = self.model.conv1.weight.clone()
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:,:3,:,:] = torch.nn.Parameter(pretrained_weights)
            self.model.conv1.weight[:,3,:,:] = torch.nn.Parameter(pretrained_weights[:,1,:,:])#

        self.new_model = nn.Sequential(*list(self.model.children())[:-2]) # remove the last two layers 

        # change the final layer
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64,32)
        self.l6 = nn.Linear(32,16)
        self.l7 = nn.Linear(16,8)
        self.l8 = nn.Linear(8,4)
        self.l9 = nn.Linear(4,2)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.new_model(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        return x.type(torch.float32)
