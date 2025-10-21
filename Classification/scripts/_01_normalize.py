# ======= Normaliza Dataset Images ==========
# Creates Processed_Dataset directory 
# Resizes images to 224x224 pixels
# Converts images to JPEG format with quality 95
# Uses LANCZOS filter for high-quality resizing

import os
from PIL import Image
from tqdm import tqdm # for progress bar

INPUT_DIR = 'Main_Dataset'
OUTPUT_DIR = 'Classification/dataset/Processed_Dataset'
SIZE = (224, 224)

def resize_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    images = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                images.append(os.path.join(root, file))
    
    for img_path in tqdm(images):
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        out_path = os.path.join(OUTPUT_DIR, rel_path)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        try:
            img = Image.open(img_path)
            img = img.resize(SIZE, Image.Resampling.LANCZOS) # Pixel enhancement through resizing => find out how
            img = img.convert('RGB')
            img.save(out_path, 'JPEG', quality=95)
        except Exception as e:
            print(f"Failed: {img_path} - {e}")

if __name__ == '__main__':
    resize_images()
