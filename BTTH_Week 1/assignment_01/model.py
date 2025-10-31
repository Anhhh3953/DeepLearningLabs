# model.py
"""
Functions:
    - Define the structure of NN
        + Input Layer: Take vector flatten image (784)
        + Hidden Layer (MLP): 1 linear layer (fully connected) with 512 neuron, with active function ReLU
        + Output Layer: 1 linear layer with 10 neuron (10 digits: 0-9) with Softmax to convert into possibility
    - MLP (Multi-Layer Perceptron) with 1 hidden layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OneLayerMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(OneLayerMLP, self).__init__()
        # nn.Linear: Fully Connected Layer
        # Ánh xạ dữ liệu dầu vào (input_size) -> không gian của lớp ẩn (hidden_size) (784 -> 512)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # No nn.Softmax here because CrossEntropyLoss has already had Softmax

    def forward(self, x):
        """
        Define the forward pass of model, which is called when model is executed
        Args:
            x (torch.Tensor): The input tensor to model. Expected shape: (batch_size, input_size)
        Returns:
            torch.Tensor: output tensor (logits before Softmax)
            torch.Tensor: probilities after applying Softmax
        """
        # 1. Move data through the fist layer
        x = self.fc1(x)

        # 2. Use ReLU
        x = F.relu(x)

        # 3. Move through second layer
        logits = self.fc2(x)

        # 4. Apply Softmax to convert logits into probabilities
        probabilities = F.softmax(logits, dim=1)

        return logits, probabilities