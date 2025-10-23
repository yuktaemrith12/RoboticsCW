
# 🧠 AI in Robotics (PDE3802) — Office Item Classification

## 📘 Overview
This repository contains the **Classification Module** for the *AI in Robotics (PDE3802)* coursework.  
It forms the **Perception** component of the desk-organising robotic arm — enabling the robot to **identify and label common office items** from images or a live webcam feed.  

---

## 🧩 Project Structure
```

ROBOTICSCW/
│
├── Classification/
│   ├── dataset/
│   │   ├── Main_Dataset/         # Raw dataset – 10 office item classes
│   │   ├── Processed_Dataset/    # Normalised 224×224 RGB images
│   │   └── Final_Dataset/        # Train / Validation / Test split
│   │
│   └── scripts/
│       ├── _01_normalize.py      # Image resizing and cleaning
│       ├── _02_split_script.py   # Train/Val/Test split automation
│       ├── _03_evaluate_model.py # Accuracy, F1, confusion matrix
│       └── Training_Classification_Model.ipynb
│
├── app.py                        # Flask app for upload + webcam classification
├── office_item_classifier.pth    # Trained ResNet-50 weights
│
├── static/
│   ├── style.css
│   └── logo.png
│
├── templates/
│   └── index.html                # Web interface (two-tab UI)
│
└── README.md

````

---

## 🧠 Recognised Classes
| Class | Example |
|:------|:---------|
| Chair | Office chair |
| Desk Lamp | Table/LED lamp |
| Headphones | Wired/wireless |
| Keyboard | Mechanical/membrane |
| Monitor | LCD/LED monitor |
| Mouse | Wired/wireless |
| Mug | Ceramic/cup |
| Notepad | Notebook |
| Pen | Writing pen |
| Table | Office table |

---

## ⚙️ Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/<your-repo-name>.git
cd ROBOTICSCW
````

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

Because the dataset is too large for GitHub, download it manually:
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

1. Ensure the virtual environment is active.
2. Launch the app:

   ```bash
   python app.py
   ```
3. Open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

You can:

* Upload an image to classify.
* Start your webcam and capture a frame for instant recognition.

**Expected Output:**

* The top predicted class (e.g., “Mug – 97 %”) displayed on screen.
* Confidence score + top-3 alternative predictions shown below.

---


## 📊 Model Details

| Feature                 | Description                              |
| :---------------------- | :--------------------------------------- |
| **Architecture**        | ResNet-50 (Transfer Learning)            |
| **Framework**           | PyTorch                                  |
| **Input Size**          | 224 × 224 RGB                            |
| **Optimizer**           | Adam                                     |
| **Loss Function**       | Cross-Entropy                            |
| **Augmentation**        | Random flip / rotation / brightness      |
| **Split**               | 70 % Train / 15 % Validation / 15 % Test |
| **Validation Accuracy** | ≈ 95 %                                   |
| **Macro F1-Score**      | ≈ 0.91                                   |
| **Saved Weights**       | `office_item_classifier.pth`             |


---

## 🎯 Object Detection (YOLOv8)

In addition to classification, a **YOLOv8-based object detection model** was developed to locate and label multiple desk items in real time.  
This detection system complements the classifier by providing **spatial awareness** — helping the robotic arm determine *where* each object is on the desk.

- **Framework:** Ultralytics YOLOv8  
- **Goal:** Real-time detection and bounding box localization of office items  
- **Dataset Source:** Roboflow (pre-annotated YOLO format)  
- **Training:** Fine-tuned pre-trained YOLOv8n model on the same office-item dataset using Google Colab GPU  
- **Augmentation:** Synthetic background variation and rotation for better generalisation  
- **Output:** Custom-trained weights `best.pt` integrated into the Flask web app  
- **Live Detection:** The app displays bounding boxes, labels, and confidence scores from the webcam feed using OpenCV  

This integration enables the robot to perform both **object recognition (ResNet-50)** and **object localisation (YOLOv8)** for a complete desk-organising perception system.


---

## 🧱 Dataset Card

* **Name:** Office-Goods Dataset
* **Classes:** 10 office items (see table above)
* **Sources:**  Roboflow 
* **Image Count:** ≈ 21 000 images
* **Pre-processing:** 224×224 px resize, RGB JPEG conversion, LANCZOS filter
* **Split:** Train 70 %  ·  Validation 15 %  ·  Test 15 %
* **Purpose:** Classification for robotic perception

---

## 💡 Expected Outputs

After running `app.py`, the Flask interface displays:

* **Prediction label** (e.g., “Pen”)
* **Confidence percentage**
* **Optional alternate predictions** if confidence < 95 %
* A preview of the processed image (upload or webcam capture).

When running `_03_evaluate_model.py`:

* Console prints overall accuracy, macro F1, and per-class accuracy.
* A **Seaborn confusion matrix** pops up for visual validation.


---

## 👩‍💻 Team Members

| Name                      | Student ID |
| :------------------------ | :--------- |
| **Yukta R. Emrith**       | M00977987  |
| **Rohaj Gokool Oopadhya** | M00955505  |
| **Kevan Chinapul**        | M00963905  |

---


