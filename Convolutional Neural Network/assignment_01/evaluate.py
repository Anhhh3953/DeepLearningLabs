import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            # Forward pass: fetch probabilities only
            _, probabilities = model(data)
            predictions = probabilities.argmax(dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print("Overall evaluation metrics")
    overall_accuracy = accuracy_score(all_targets, all_preds)
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    overall_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"Overall F1: {overall_f1:.4f}")

    overall_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"Overall recall: {overall_recall:.4f}")

    overall_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    print(f"Overall precision: {overall_precision:.4f}")

    num_classes = len(np.unique(all_targets))
    per_class_results = {}
    print("Per-class evaluation metrics")
    for i in range(num_classes):
        class_targets = (all_targets == i).astype(int)
        class_preds = (all_preds == i). astype(int)

        precision = precision_score(class_targets, class_preds, zero_division=0)
        recall = recall_score(class_targets, class_preds, zero_division=0)
        f1 = f1_score(class_targets, class_preds, zero_division=0)
        accuracy = accuracy_score(class_targets, class_preds)

        per_class_results[i] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print(f"Class {i}: \n Accuracy = {accuracy}\n Precision = {precision} \n Recall = {recall} \n F1 = {f1} \n")
    return{
        'overall': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        },
        'per_class': per_class_results
    }