# ======= Split Dataset into Train, Validation, and Test Sets ==========
# Splits the Processed_Dataset into Final_Dataset
# Creates train (70%), validation (15%), and test (15%) directories
# Maintains class subdirectories in each split

import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = 'Classification/dataset/Processed_Dataset'
OUTPUT_DIR = 'Classification/dataset/Final_Dataset'
SPLIT_RATIO = [0.70, 0.15, 0.15]


def split_dataset():
    """
    Splits the dataset into training, validation, and test sets.
    """
    if sum(SPLIT_RATIO) != 1.0:
        print(f"Error: Split ratio must sum to 1.0, but it sums to {sum(SPLIT_RATIO)}")
        return

    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print(f"Creating new directory structure at: {OUTPUT_DIR}")
    train_path = os.path.join(OUTPUT_DIR, 'train')
    val_path = os.path.join(OUTPUT_DIR, 'validation')
    test_path = os.path.join(OUTPUT_DIR, 'test')
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    class_names = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not class_names:
        print(f"No class folders found in {SOURCE_DIR}. Please check your input directory.")
        return

    print(f"Found {len(class_names)} classes. Starting the split...")

    # Process each class
    for class_name in tqdm(class_names, desc="Processing classes"):
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

        class_source_path = os.path.join(SOURCE_DIR, class_name)
        files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        
        random.shuffle(files)

        train_end = int(len(files) * SPLIT_RATIO[0])
        val_end = train_end + int(len(files) * SPLIT_RATIO[1])

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        for f in train_files:
            shutil.copy2(os.path.join(class_source_path, f), os.path.join(train_path, class_name, f))
        for f in val_files:
            shutil.copy2(os.path.join(class_source_path, f), os.path.join(val_path, class_name, f))
        for f in test_files:
            shutil.copy2(os.path.join(class_source_path, f), os.path.join(test_path, class_name, f))

    print("\n Dataset split successfully!")
    print(f"Final dataset is ready in the '{OUTPUT_DIR}' folder.")

if __name__ == '__main__':
    split_dataset()

# NEXT STEP:
# Zip Final_Dataset folder 
# Upload to Google Drive for easy access
# 