import os
import torch

from data_loader import MNistDataLoader
from model import LeNet
from train import Trainer
from evaluate import evaluate_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # 1. Load and preprocessing data
    print("Loading data")
    data_loader = MNistDataLoader(batch_size=64, num_workers=0, data_dir="../data")
    train_loader =  data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # 2. Build model
    print("Building model")
    model = LeNet(num_classes=10).to(device)
    print(model)

    # 3. Train model
    print("Training model")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.01,
        epochs=10,
        save_dir="./checkpoints"
    )
    trainer.train()

    # 4. Evaluate
    print("Final evaluate with best model")
    best_model_path = os.path.join('best_model_assignment_01.pt')
    final_model = LeNet()
    final_model.load_state_dict(torch.load(best_model_path, map_location=device))
    # Evaluate in test dataset
    final_metrics = evaluate_model(final_model, test_loader, device)

    print(f"Final overall metrics")
    for k, v in final_metrics['overall'].items():
        print(f"{k}: {v:.4f}")
    for digit, metrics in final_metrics['per_class'].items():
        print(f"Digit {digit}: Acc = {metrics['accuracy']:.4f}, Pre = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}, F1 = {metrics['f1']:.4f}")

if __name__ == '__main__':
    main()