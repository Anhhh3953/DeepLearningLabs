# assignment_2/main.py
import torch
import argparse
import os
import glob

from ..data_loader import MNISTDataLoader
from .model import ThreeLayerMLP
from .trainer import Trainer
from ..evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Exercise: 3-Layer MLP for MNIST")
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility (default: 42)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_assignment_02', help='directory to save model checkpoints for assignment 02')
    parser.add_argument('--data_dir', type=str, default='../mnist_dataset', help='directory where MNIST dataset is located')
    parser.add_argument('--input_size', type=int, default=784, help='input feature size (default: 784 for 28x28 images)')
    parser.add_argument('--hidden_size_1', type=int, default=512, help='number of neurons in the first hidden layer (default: 512)')
    parser.add_argument('--hidden_size_2', type=int, default=256, help='number of neurons in the second hidden layer (default: 256)')
    parser.add_argument('--num_classes', type=int, default=10, help='number of output classes (default: 10)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    # 1. Load and preprocess data
    print("Load data")
    data_loader = MNISTDataLoader(batch_size=args.batch_size, data_dir=args.data_dir)
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # 2. Build model
    print("Build model")
    model = ThreeLayerMLP(
        input_size=args.input_size,
        hidden_size_1=args.hidden_size_1,
        hidden_size_2=args.hidden_size,
        num_classes=args.num_classes
    ).to(device)
    print(model)

    # 3. Train model
    print(f"Train model")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        save_dir=args.save_dir
    )
    trainer.train()

    # 4. Evaluate
    print("Final evaluate with best model")
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    final_model = ThreeLayerMLP(input_size=args.input_size, hidden_size=args.hidden_size, num_classes=args.num_classes).to(device) # (*)
    # Load trained parameters from file into (*)
    final_model.load_state_dict(torch.load(best_model_path), map_location=device)
    # Evaluate in test dataset
    final_metrics = evaluate_model(final_model, test_loader, device)

    print(f"Final overall metrics")
    for k, v in final_metrics["overall"].items():
        print(f"{k}: {v:.4f}")

    print(f"Final per-class metrics")
    for digit, metrics in final_metrics['per_class'].items():
        print(f"Digit {digit}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

if __name__ == '__main__':
    main()

