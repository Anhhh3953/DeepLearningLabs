# trainer.py
"""
Functions:
    - Manage training loop through epoch
    - Execute forward/ backward pass, update parameters, save best models periodically
"""
import torch
import torch.nn as nn
import torch.optim as optim
from  tqdm import tqdm
import os
import glob

from model import ThreeLayerMLP
from ..data_loader import MNISTDataLoader
from ..evaluate import evaluate_model

class Trainer:
    "Class to handle the training process of the ThreeLayerMLP model"
    def __init__(self, model, train_loader, test_loader, device, learning_rate=0.01, epochs=10, save_dir='./checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_dir = save_dir
        self.best_accuracy = 0.0 # varible for check the best accuracy in testing section

        # Define loss function (CrossEntropyLoss for multi-label classification)
        # nn.CrossEntropyLoss has already had Softmax and Logarithm -> model output (logits) DO NOT have to softmax before go into this loss function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.model.parameters(): input all models'parameters (trọng số và bias) that optimizer needs to update
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """
        Perform 1 training epoch
        Args:
            epoch: The current epoch number

        Returns: the average training loss for the epoch
        """
        self.model.train() # training mode will activate Dropout/ BatchNorm -> will be different from evaluating mode
        total_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"{epoch}/{self.epochs} training")):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad() # 1. Reset all gradiant as 0 to avoid sum from previous batch

            # 2. Forward pass: Truyền dữ liệu qua model
            logits, _ = self.model(data) # just need logits for loss

            loss = self.criterion(logits, target)

            # 3. Backward pass: Tính toán gradient
            loss.backward() # Compute all gradient of loss function of all operates which have requires_grad=True

            # 4. Update parameters
            self.optimizer.step() # base on the calculated gradient and learning rate -> optimizer update

            # 5.
            total_loss += loss.item() # Sum loss of each batch
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} with training loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        "Execute full training process"
        print(f"Training on {self.device} for {self.epochs} with learning rate = {self.learning_rate}")
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)

            # Evaluate in testing after each epoch
            metrics = evaluate_model(self.model, self.test_loader, self.device)
            current_accuracy = metrics['overall']['accuracy']
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                model_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"New best model saved at: {model_path} with accuracy = {self.best_accuracy:.4f}")
        print("Training finished")
        print(f"Best accuracy: {self.best_accuracy:.4f}")




