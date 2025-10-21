# =============================================
#TRAINING:

#Normally training would have been done on local CPU => too much time + ressources

#Solution: Train on Google Collab to get access to virtual GPU

#Reason: Increase efficiency 
# ===============================================


import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# --- Configuration ---
MODEL_PATH = 'models/office_item_classifier.pth'
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

    # Load the test dataset
    test_dataset = datasets.ImageFolder(TEST_DIR, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # Re-create the model architecture
    model = models.resnet50() 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    print(" Model loaded successfully.")

    # --- 3. Run Predictions ---
    all_preds = []
    all_labels = []

    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(" Predictions complete.")

    # --- 4. Calculate and Display Metrics ---
    print("\n--- Performance Metrics ---")
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    # Macro-F1 Score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Macro-F1 Score: {macro_f1:.4f}")

    # --- 5. Generate and Plot Confusion Matrix ---
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    evaluate_model()