
# 🧠 AI in Robotics (PDE3802) — Office Item Classification (YOLOv8)

## 📘 Overview

This repository contains the **Classification Module** for the *AI in Robotics (PDE3802)* coursework.
It forms the **Perception** component of the desk-organising robotic arm — enabling the robot to **recognise and classify common office items** from static images or a live webcam feed.

This version replaces the earlier ResNet-50 model with a **YOLOv8-Classification** pipeline for improved training efficiency, higher accuracy, and seamless integration with the future detection module.

---

## 🧩 Project Structure

```
RoboticsCW/
│
├── Classification/
│   ├── dataset/
│   │   ├── Main_Dataset/         # Raw dataset – 10 office item classes
│   │   ├── Processed_Dataset/    # Normalised 224×224 RGB images
│   │   └── Final_Dataset/        # Train / Validation / Test split (70/15/15)
│   │
│   └── scripts/
│       ├── _01_normalize.py      # Image resizing and cleaning (LANCZOS filter)
│       ├── _02_split_script.py   # Automated train/val/test splitting
│       ├── _03_Classification_YOLO_Training.ipynb  # YOLOv8 classification training
│       ├── _04_evaluate_model.py # Evaluation: accuracy, F1, confusion matrix
│
├── app.py                        # Flask app (image upload + webcam classification)
├── office_item_classifier_yolov8cls.pt  # Trained YOLOv8 classification weights
│
├── static/
│   ├── style.css
│   └── logo.png
│
├── templates/
│   └── index.html                # Web interface (two-tab UI)
│
└── README.md
```

---

## 🧠 Recognised Classes

| Class      | 
| :--------- | 
| Chair      | 
| Desk Lamp  | 
| Headphones | 
| Keyboard   | 
| Monitor    |
| Mouse      | 
| Mug        | 
| Notepad    | 
| Pen        | 
| Table      | 

---

## ⚙️ Installation Guide

### 1. Clone Repository

```bash
git clone https://github.com/<your-repo-name>.git
cd RoboticsCW
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

The dataset is hosted externally due to size limits:
📦 [Classification Dataset (Google Drive)](https://drive.google.com/file/d/18K4xG9XFKQ2DGNMg43CZ8u7B-xJFcFJg/view?usp=drive_link)

Unzip into:

```
Classification/dataset/Final_Dataset/
    ├── train/
    ├── validation/
    └── test/
```

---

## ▶️ How to Run

### Run the Web Application Locally

1. Activate your virtual environment.
2. Launch the Flask app:

   ```bash
   python app.py
   ```
3. Open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

**Features:**

* Upload an image for classification
* Capture a webcam frame for instant prediction
* See the predicted class + confidence score

**Expected Output:**
Example → “Monitor – 96 %” with top-3 alternatives displayed.

---


## 🧪 Evaluation Results

**Overall Performance**

```
Accuracy : 0.9579
Macro F1 : 0.9551
```

**Per-Class Metrics (excerpt)**

| Class     | Acc   | Prec  | Rec   | F1    |
| :-------- | :---- | :---- | :---- | :---- |
| Chair     | 0.909 | 0.955 | 0.909 | 0.931 |
| Desk Lamp | 0.993 | 0.997 | 0.993 | 0.995 |
| Keyboard  | 0.985 | 0.973 | 0.985 | 0.979 |
| Mug       | 0.997 | 0.997 | 0.997 | 0.997 |
| Notepad   | 0.993 | 0.993 | 0.993 | 0.993 |
| Table     | 0.888 | 0.820 | 0.888 | 0.853 |

**Key Insight:**
The confusion matrix shows predictions tightly clustered along the diagonal — meaning minimal misclassifications. Slight confusion occurs between *chair* and *table*, which share similar textures in some samples.

---

## 🎯 Object Detection (YOLOv8)

In addition to classification, a **YOLOv8 detection module** was developed to locate multiple desk items in real time.
It provides **spatial awareness** for robotic manipulation.

| Feature            | Description                                                                             |
| :----------------- | :---------------------------------------------------------------------------------------|
| **Framework**      | Ultralytics YOLOv8                                                                      |
| **Goal**           | Detect and localise multiple objects                                                    |
| **Dataset Source** | Roboflow (YOLO format)                                                                  |
| **Model**          | Fine-tuned YOLOv8n                                                                      |
| **Output**         | `office_item_classifier_yolov8cls.pt` — integrated into Flask for live webcam detection |

---

## 🧱 Dataset Card

| Attribute          | Details                               |
| :----------------- | :------------------------------------ |
| **Name**           | Office-Goods Dataset                  |
| **Classes**        | 10 office items                       |
| **Sources**        | Roboflow + Synthetic Augmentation     |
| **Image Count**    | ≈ 21 000                              |
| **Pre-processing** | 224×224 resize, RGB JPEG, LANCZOS     |
| **Split**          | 70 % Train · 15 % Val · 15 % Test     |
| **Purpose**        | Classification for robotic perception |

---

## 💡 Expected Outputs

When running:

* `_04_evaluate_model.py` → prints accuracy, F1, per-class stats, and displays the confusion matrix
* `app.py` → web app predicts class and confidence from uploads / webcam

---

## 🛠️ Troubleshooting

| Issue              | Cause                     | Fix                                                       |
| :----------------- | :------------------------ | :-------------------------------------------------------- |
| `torch not found`  | Environment not activated | Run `source venv/bin/activate` or `venv\Scripts\activate` |
| Dataset not found  | Wrong extraction path     | Ensure `Classification/dataset/Final_Dataset/` exists     |
| Webcam not opening | Permission issue          | Allow camera access or run locally                        |
| Low accuracy       | Unbalanced dataset        | Increase epochs / augment data                            |

---

## 👩‍💻 Team Members

| Name                      | Student ID |
| :------------------------ | :--------- |
| **Yukta R. Emrith**       | M00977987  |
| **Rohaj Gokool Oopadhya** | M00955505  |
| **Kevan Chinapul**        | M00963905  |

