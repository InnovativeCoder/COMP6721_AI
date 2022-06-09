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
        self.conv_layer1 = nn.Conv2d(numChannels,10, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv_layer2 = nn.Conv2d(10, 25, 5)
        self.fc1 = nn.Linear(25*22*22, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, classes)

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

