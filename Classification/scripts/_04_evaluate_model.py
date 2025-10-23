# =============================================
# TRAINING / EVALUATION:
# Evaluate final model on test set and report:
# - Overall Accuracy + Macro F1
# - Per-class Accuracy + F1 + Support
# - Confusion Matrix
# =============================================

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# --- Configuration ---
MODEL_PATH = 'office_item_classifier.pth'
TEST_DIR = 'Classification/dataset/Final_Dataset/test'
BATCH_SIZE = 32

def evaluate_model():
    print("\n--- Starting Final Model Evaluation on Test Set ---\n")

    # --- 1. Load Data ---
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # --- 2. Load Model ---
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.\n")

    # --- 3. Run Predictions ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("Predictions complete.\n")

    # --- 4. Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print("=== OVERALL PERFORMANCE ===")
    print(f"{'Accuracy':<15}: {acc:.4f}")
    print(f"{'Macro F1':<15}: {macro_f1:.4f}\n")

    # --- 5. Per-class Metrics ---
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    per_class_f1 = f1_score(all_labels, all_preds, labels=range(num_classes), average=None)
    per_class_prec = precision_score(all_labels, all_preds, labels=range(num_classes), average=None, zero_division=0)
    per_class_rec = recall_score(all_labels, all_preds, labels=range(num_classes), average=None, zero_division=0)

    print("=== PER-CLASS METRICS ===")
    print(f"{'Class':<20}{'Acc':>8}{'Prec':>10}{'Rec':>10}{'F1':>10}{'Support':>10}")
    print("-" * 60)
    for i, cls in enumerate(class_names):
        support = cm.sum(axis=1)[i]
        print(f"{cls:<20}{per_class_acc[i]:>8.3f}{per_class_prec[i]:>10.3f}{per_class_rec[i]:>10.3f}{per_class_f1[i]:>10.3f}{support:>10}")
    print("-" * 60)

    # --- 6. Confusion Matrix ---
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
