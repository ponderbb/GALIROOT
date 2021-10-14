import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I # for weight initialization

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding (by default 0)
        # W - the width/height (square) of the previous layer

        # output = (W-F+2P)/S+1


        # [3,635,775]
        # x = torch.randn(1,3,635,775) # testing the output

        self.conv1 = nn.Conv2d(3, 32, 5) # 315 385
        self.conv2 = nn.Conv2d(32, 64, 3) # 156 191
        self.conv3 = nn.Conv2d(64, 128, 3) # 77 94
        self.conv4 = nn.Conv2d(128, 256, 3) # 37 46
        self.conv5 = nn.Conv2d(256, 512, 1) # 18 23

        self.pool = nn.MaxPool2d(2,2)

        # Fully connected layers
        self.fc1 = nn.Linear(512*18*23, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1*2)
        # output should correspond to desired amount of keypoints (x,y)

        self.dropout = nn.Dropout(p=0.30)

    def forward(self, x):
        x = x.permute(0,3,1,2)
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
        x = F.relu(self.fc3(x))

        return x



if __name__ == "__main__":

    x = torch.randn(1,3,256,256) # testing the output

    net = Net()

    output = net(x)
    print(output)


