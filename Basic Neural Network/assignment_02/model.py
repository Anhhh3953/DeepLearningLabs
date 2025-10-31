# assignment_02/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size_1=512, hidden_size_2=256, num_classes=10):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_1, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probabilities = F.softmax(logits)
        return logits, probabilities