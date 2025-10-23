# ==============================================================
# 📂 SPLIT DATASET INTO TRAIN, VALIDATION, AND TEST SETS
# --------------------------------------------------------------
# Purpose:
#   Splits the processed dataset into train (70%), validation (15%),
#   and test (15%) subsets for model training and evaluation.
#   Maintains class subfolder structure for each split.
#
# Input  : Classification/dataset/Processed_Dataset
# Output : Classification/dataset/Final_Dataset
# ==============================================================

import os
import shutil
import random
from tqdm import tqdm

# --- Configuration Variables ---
SOURCE_DIR = 'Classification/dataset/Processed_Dataset'  # Input dataset (normalized images)
OUTPUT_DIR = 'Classification/dataset/Final_Dataset'      # Output directory for split data
SPLIT_RATIO = [0.70, 0.15, 0.15]                         # Train, Validation, Test ratios


def split_dataset():
    """
    Splits the dataset into training, validation, and test sets.
    """

    # --- Validate Split Ratio ---
    if sum(SPLIT_RATIO) != 1.0:
        print(f"Error: Split ratio must sum to 1.0, but it sums to {sum(SPLIT_RATIO)}")
        return

    # --- Remove Old Output Directory (if exists) ---
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # --- Create New Output Directory Structure ---
    print(f"Creating new directory structure at: {OUTPUT_DIR}")
    train_path = os.path.join(OUTPUT_DIR, 'train')
    val_path = os.path.join(OUTPUT_DIR, 'validation')
    test_path = os.path.join(OUTPUT_DIR, 'test')
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # --- Get All Class Folders from Source Directory ---
    class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not class_names:
        print(f"No class folders found in {SOURCE_DIR}. Please check your input directory.")
        return

    print(f"Found {len(class_names)} classes. Starting the split...")

    # --- Main Loop: Split Each Class Folder ---
    for class_name in tqdm(class_names, desc="Processing classes"):
        # Create class folders for each split
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

        class_source_path = os.path.join(SOURCE_DIR, class_name)
        files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        
        # Shuffle files for random distribution
        random.shuffle(files)

        # Calculate index boundaries for each split
        train_end = int(len(files) * SPLIT_RATIO[0])
        val_end = train_end + int(len(files) * SPLIT_RATIO[1])

        # Split files by subset
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # --- Copy Files to Corresponding Folders ---
        for f in train_files:
            shutil.copy2(os.path.join(class_source_path, f), os.path.join(train_path, class_name, f))
        for f in val_files:
            shutil.copy2(os.path.join(class_source_path, f), os.path.join(val_path, class_name, f))
        for f in test_files:
            shutil.copy2(os.path.join(class_source_path, f), os.path.join(test_path, class_name, f))

    # --- Completion Message ---
    print("\n Dataset split successfully!")
    print(f"Final dataset is ready in the '{OUTPUT_DIR}' folder.")


# --- Main Execution Entry Point ---
if __name__ == '__main__':
    split_dataset()

# NEXT STEP:
# Zip Final_Dataset folder 
# Upload to Google Drive for easy access
