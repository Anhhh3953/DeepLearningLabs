# main.py
"""
Flow:
    1. Read operators from command line
    2. Prepare environment (device, seed)
    3. Load data through data_loader.py
    4. Initailize model through model.py
    5. Initialize and run training process through trainer.py
    5. Save best model and evaluate by evaluate.py
"""
import torch
import argparse
import os
import glob

from data_loader import MNISTDataLoader
from .model import OneLayerMLP
from .trainer import Trainer
from evaluate import evaluate_model


def main():
    # Config command by argparse to rrun by terminal
    parser = argparse.ArgumentParser(description="DL Week 01: 1-Layer MLP for MNist")

    # Defind đối số
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default = 128)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="directory to save model checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./mnist_dataset",
        help="directory where MNIST dataset is located",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=784,
        help="input feature size (default: 784 for 28x28 images)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="number of neurons in the hidden layer (default: 512)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="number of output classes (default: 10)",
    )

    args = parser.parse_args()

    # 1. Create seed (just for reproductivity) -> same results
    torch.manual_seed(args.seed)  # for cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # 2. Load and preprocess data
    print(f"Loading data")
    data_loader = MNISTDataLoader(batch_size=args.batch_size, data_dir=args.data_dir)
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # 3. Build model
    print(f"Building model")
    model = OneLayerMLP(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
    ).to(device)
    print(model)

    # 4. Train model
    print("Start traing")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )
    trainer.train()

    # 6. Evaluate best model
    print("Final evaluate with best model")
    _path = os.path.join(args.save_dir, "best_model.pth")
    final_model = OneLayerMLP(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=args.num_classes,
    ).to(
        device
    )  # (*)
    # Load trained parameters from file into (*)
    final_model.load_state_dict(torch.load(best_model_path, map_location=device))
    # Evaluate in test dataset
    final_metrics = evaluate_model(final_model, test_loader, device)

    print(f"Final overall metrics")
    for k, v in final_metrics["overall"].items():
        print(f"{k}: {v:.4f}")

    print(f"Final per-class metrics")
    for digit, metrics in final_metrics["per_class"].items():
        print(
            f"Digit {digit}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}"
        )


if __name__ == "__main__":
    main()
