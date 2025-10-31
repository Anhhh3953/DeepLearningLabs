# data_loader.py
"""
Function:
- Prepare input data for training and evaluating process
- To be more specific, load data, preprocess image (convert to tensor, normalize, flatten,...)
- Package into DataLoader of PyTorch
- Make it easier for loop data batch by batch during training section
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class MNISTDataLoader:
    "Class to load and preprocess the MNist Dataset"
    def __init__(self, batch_size=64, num_workers=0, data_dir="./mnist_dataset"):
        """
        Initialize the data loader with batch size, numberr of workers and data directory
        Args:
            batch_size: Number of samples per batch
            num_workers: Numberr of subprocesses for data loading (0 here -> data will be load in just 1 main process)
            data_dir: Directory where MNist dataset is stored
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        # Define transformations for image in MNist
        # Compose allow to concate multiple transformations together
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 1. Convert PIL image or NumPy array into PyTorch tensor
            # Automaticallt divide value of pixel in [0, 255] -> [0.0, 1.0]
            # Also size [28, 28] -> ToTensor -> [1, 28, 28] (kernel, height, width)
            transforms.Normalize((0.1307,), (0.3081,)),  # 2. Normalization
            # This is mean and std devivation of MNist Dataset
            # Formular for normalization: pixel = (pixel - mean)/ std_dev
            # (0.1307,), (0.3081,) is for grayscale (just 1 kernal)
            transforms.Lambda(lambda x: x.view(-1)) # 3. Flatten tensor
            # Image 28x28x1 -> a vector 28x28=784
            # x.view(-1): compute the remained size base on batch size -> necessary to put data into Fully Connected layer (Linear) of MLP
        ])
        if not os.path.exists(self.data_dir):
            print(f"Could not found data in {self.data_dir}")
            raise

    def get_train_loader(self):
        "Return the training data loader"
        # Load train dataset
        train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=False, transform=self.transform)

        # Create DataLoader from dataset
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # shuffle = True: let the dataset be mixed up each epoch to avoid model learning in orders and patterns
        print(f"Loaded {len(train_dataset)} training samples")
        return train_loader

    def get_test_loader(self):
        "Return the testing data loader"
        test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=False, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        print(f"Loaded {len(test_dataset)} test samples")
        return test_loader
