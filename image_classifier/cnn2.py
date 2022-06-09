from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, numChannels, classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(numChannels,12, 7)
        self.pool = nn.MaxPool2d(2,2)
        self.conv_layer2 = nn.Conv2d(12, 30, 7)
        self.fc1 = nn.Linear(30*20*20, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, classes)

        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # conv layers
        x = self.conv_layer1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv_layer2(x)
        x = F.relu(x)
        x = self.pool(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        out = self.logSoftmax(x)
        return out

