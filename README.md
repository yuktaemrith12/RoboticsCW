
# 🧠 Office Item Classification – AI in Robotics (PDE3802)

## 📘 Overview

This repository contains the **Classification Module** for the *AI in Robotics (PDE3802)* coursework 
The goal is to design and implement a **deep-learning-based classifier** that recognizes common office items from single images or live webcam feeds.

---

## 🧩 Project Structure

```
ROBOTICSCW/
│
├── Classification/
│   ├── dataset/
│   │   ├── Main_Dataset/           # Raw dataset – 10 office item classes
│   │   ├── Processed_Dataset/      # Normalized images (224×224 RGB)
│   │   └── Final_Dataset/          # Train / Validation / Test split
│   │
│   └── scripts/
│       ├── _01_normalize.py        # Image cleaning & resizing
│       ├── _02_split_script.py     # Train/Val/Test split script
│       ├── _03_evaluate_model.py   # Metrics & confusion matrix
│       └── Training_Classification_Model.ipynb
│
├── app.py                          # Flask app for image + webcam classification
├── office_item_classifier.pth      # Saved ResNet-50 model weights
│
├── static/
│   ├── style.css                   # UI design
│   └── logo.png
│
├── templates/
│   └── index.html                  # Web interface (upload + webcam tabs)
│
└── README.md                       # You are here
```

---

## 🧠 Recognized Classes

| Class      | Example               |
| ---------- | --------------------- |
| chair      | office chairs         |
| desk lamp  | table/LED lamps       |
| headphones | wired/wireless        |
| keyboard   | mechanical & membrane |
| monitor    | LCD/LED monitors      |
| mouse      | wired/wireless        |
| mug        | ceramic/cup           |
| notepad    | notebooks             |
| pen        | writing pens          |
| table      | office tables         |

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/<your-repo-name>.git
cd ROBOTICSCW
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Access

* The dataset (≈ 10 classes × several hundred images each) is too large for direct upload.
* Download the zipped dataset here → [**[Classification Dataset](https://drive.google.com/file/d/18K4xG9XFKQ2DGNMg43CZ8u7B-xJFcFJg/view?usp=drive_link)**](#)
* Unzip into:

```
Classification/dataset/Final_Dataset/
```

Each subset maintains class-specific folders:

```
train/
validation/
test/
```

---

## 🚀 Running the Application

### Option A – Run Locally

```bash
python app.py
```

Then open **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

### Option B – Google Colab Training

To retrain the model, open
`Classification/scripts/Training_Classification_Model.ipynb` in Google Colab.
Ensure GPU runtime is enabled for faster processing.

---

## 🧮 Model Details

| Feature                  | Description                                               |
| ------------------------ | --------------------------------------------------------- |
| **Architecture**         | ResNet-50 (transfer learning)                             |
| **Framework**            | PyTorch                                                   |
| **Optimizer**            | Adam                                                      |
| **Loss**                 | Cross-Entropy                                             |
| **Input Size**           | 224 × 224 RGB                                             |
| **Normalization**        | Mean = [0.485, 0.456, 0.406]; Std = [0.229, 0.224, 0.225] |
| **Data Augmentation**    | Random flip, rotation, crop, normalization                |
| **Train/Val/Test Split** | 70 / 15 / 15                                              |
| **Validation Accuracy**  | ≈ 95 %                                                    |
| **Macro F1 Score**       | ≈ 0.90 +                                                  |
| **Saved Weights**        | `office_item_classifier.pth`                              |

---

## 🧱 Dataset Card

* **Name:** Office-Goods Dataset
* **Version:** 1.0
* **Source:** Kaggle / Roboflow 
* **Classes:** 10 (chair, desk lamp, headphones, keyboard, monitor, mouse, mug, notepad, pen, table)
* **Size:** ≈ 4,500 images
* **Splits:** Train (70 %), Validation (15 %), Test (15 %)
* **License:** Public, educational use
* **Pre-processing:** Resized 224×224 px, normalized, converted to JPEG RGB
* **Usage:** For classification 

---

## 📹 Demonstration

A 2-minute code-walkthrough video explaining the installation, dataset structure, and live demo is available at:
🎥 [YouTube Demo Link](#)

---

## 👩‍💻 Team Members

| Name                      | Student ID | 
| ------------------------- | ---------- |
| **Yukta Emrith**          | M00977987  |
| **Rohaj Gokool Oopadhya** | M00955505  | 
| **Kevan Chinapul**        | M00963905  | 

