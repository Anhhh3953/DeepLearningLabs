import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=5*5*16, out_features=120)

        self.fc2 = nn.Linear(in_features=120, out_features=84)

        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        "Define the forward pass of the LeNet model"
        # 1. Conv1 -> Pool1
        x = self.conv1(x)
        x = self.pool1(x)

        # 2. Conv2 -> Pool2
        x = self.conv2(x)
        x = self.pool2(x)

        # 3. Flatten before putting into Fully Connected layer
        x = x.view(x.size(0), -1)

        # 4. FC1 -> fc2 -> fc3
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)

        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities
    
