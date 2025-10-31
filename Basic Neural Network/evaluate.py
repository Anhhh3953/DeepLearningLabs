# evaluate.py
"""
Functions: Evaluate in testing dataset
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on given data loader with various classification metrics
    Args:
        model (torch.nn.Module): the trained model to evaluate
        data_loader(torch.utils.data.Dataloader): testing set
        device: cpu here
    Returns:
        dict: a dict contains overal and per-class evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        # Loop through each batch in data loader
        for data, target in tqdm(data_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            # Forward pass: fetch probabilities only
            _, probabilities = model(data)
            predictions = probabilities.argmax(dim=1) # Lấy lớp dự đoán có xác suất cao nhất (chỉ số của giá trị lớn nhất)
            # Convert from tensor -> numpy array
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print("Overall evaluation metrics")
    overall_accuracy = accuracy_score(all_targets, all_preds)
    print(f"Accuracy: {overall_accuracy:.4f}")

    overall_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"Precision (macro): {overall_precision:.4f}")

    overall_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"Recall (macro): {overall_recall:.4f}")

    overall_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"F1-score: {overall_f1:.4f}")

    num_classes = len(np.unique(all_targets))
    per_class_results = {}
    print("Per-class evaluation metrics")
    for i in range(num_classes):
        class_targets = (all_targets == i).astype(int)
        class_preds = (all_preds == i).astype(int)

        precision = precision_score(class_targets, class_preds, zero_division=0)
        recall = recall_score(class_targets, class_preds, zero_division=0)
        f1 = f1_score(class_targets, class_targets, zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)

        per_class_results[i] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }
        print(f"Class {i}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")
    return {
        'overall': {
            'accuracy': overall_accuracy,
            'precision_macro': overall_precision,
            'recall_macro': overall_recall,
            'f1_macro': overall_f1
        },
        'per_class': per_class_results
    }
