# =============================================
# TRAINING / EVALUATION:
# Evaluate final model on test set and report:
# - Overall Accuracy + Macro F1
# - Per-class Accuracy (TP / total samples of that class)
# - Confusion Matrix
# =============================================

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# --- Configuration ---
MODEL_PATH = 'office_item_classifier.pth'
TEST_DIR = 'Classification/dataset/Final_Dataset/test'
BATCH_SIZE = 32

def evaluate_model():
    print("--- Starting Final Model Evaluation on Test Set ---")

    # --- 1. Load Data ---
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # --- 2. Re-create model & load weights ---
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Run Predictions ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("Predictions complete.")

    # --- 4. Metrics ---
    print("\n--- Performance Metrics ---")
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro-F1 Score : {macro_f1:.4f}")

    # --- 4.1 Per-class Accuracy (TP / total samples of that class) ---
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    # cm[i, i] = TP for class i; sum over row i = total true samples for class i
    per_class_acc = (np.diag(cm) / cm.sum(axis=1).clip(min=1)).astype(float)

    print("\nPer-class Accuracy:")
    for idx, cls in enumerate(class_names):
        print(f"  {cls:>20s}: {per_class_acc[idx]:.4f}  (support={cm.sum(axis=1)[idx]})")

    # --- 4.2 (Optional) Detailed per-class precision/recall/F1 ---
    print("\nDetailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # --- 5. Confusion Matrix Plot ---
    print("\nGenerating confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_model()
