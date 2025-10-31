import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNistDataLoader:
    def __init__(self, batch_size = 64, num_workers = 0, data_dir = "../data"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Convert into PyTorch Tensor
            transforms.Normalize((0.1307,), (0.3081,)), # 2. Normalization
            # transforms.Lambda(lambda x: x.view(-1)) # Flatten tensor
        ])
        if not os.path.exists(self.data_dir):
            print(f"Could not found data in {self.data_dir}")
            raise

    def get_train_loader(self):
        train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=False, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print(f"Loaded {len(train_dataset)} training samples")
        return train_loader

    def get_test_loader(self):
        test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=False, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        print(f"Loaded {len(test_dataset)} test samples")
        return test_loader