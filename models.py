"""
models.py

All Pytorch model definitions
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A standard CNN model
        2 convolutional layers
        1 max-pool layer
        3 fully-connected layers
        Output is a scalar (sent to BCEWithLogitsLoss to 
                            calculate value in [0,1])
    """
    
    def __init__(self, dim=128):
        super().__init__()
        final_dim = int((math.floor(((dim-4)-2)/2+1)-4)/2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * final_dim * final_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x