import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from evaluate import evaluate_model

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, learning_rate=0.01, epochs=10, save_dir='./checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_dir = save_dir

        self.best_accuracy = 0.0

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"{epoch}/{len(self.epochs)} training")):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # 2. Forward pass
            logits, _ = self.model(data)
            loss = self.criterion(logits, target)

            # 3. Backward pass
            loss.backward()

            # 4. Update parameters
            self.optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} with training loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        print(f"Training on {self.device} for {self.epochs} with learning rate = {self.learning_rate}")
        for epoch in range(1, self.epochs + 1):
            # 1. Train
            self.train_epoch(epoch)

            # 2. Evaluate
            metrics = evaluate_model(self.model, self.test_loader, self.device)
            current_accuracy = metrics['overall']['accuracy']
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                model_path = os.path.join(self.save_dir, 'best_model_assignment_01.pt')
                torch.save(self.model.state_dict(), model_path)
                print(f"New best model saved at {model_path} with accuracy = {self.best_accuracy}")
        print("Training finished")
        print(f"Best accuracy: {self.best_accuracy:.4f}")


