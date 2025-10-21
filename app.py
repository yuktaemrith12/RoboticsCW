# app.py
# Unified Flask app: Image Classification (upload/webcam snapshot) + YOLO Object Detection (live stream)

import os
import io
import base64
import threading
import time
from collections import deque

# --- Flask / HTTP ---
from flask import Flask, request, render_template, jsonify, Response

# --- Classification stack (PyTorch / torchvision) ---
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Detection stack (OpenCV / Ultralytics YOLO) ---
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# =========================================================
# =============== 1) CLASSIFICATION SECTION ===============
# =========================================================

CLASSES = [
    'chair', 'desk lamp', 'headphones', 'keyboard', 'monitor',
    'mouse', 'mug', 'notepad', 'pen', 'table'
]

def load_classifier():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    # adjust path if needed
    state = torch.load('office_item_classifier.pth', map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model

clf_model = load_classifier()

def preprocess_image(img_bytes: bytes) -> torch.Tensor:
    """Same pipeline for BOTH upload and webcam base64 frames."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)

def run_cls_inference(img_tensor: torch.Tensor):
    with torch.no_grad():
        probs = torch.softmax(clf_model(img_tensor), dim=1)
        top3_probs, top3_idx = torch.topk(probs, 3)

    main_idx = top3_idx[0][0].item()
    main_class = CLASSES[main_idx]
    main_conf = float(top3_probs[0][0].item() * 100)

    other_preds = None
    if main_conf < 95:
        other_preds = [
            {'class': CLASSES[top3_idx[0][i].item()],
             'confidence': f"{top3_probs[0][i].item() * 100:.2f}"}
            for i in range(1, 3)
        ]
    return main_class, main_conf, other_preds

# ---- Classification routes ----

@app.route('/', methods=['GET'])
def home():
    """Landing page for classification UI (keeps your existing index.html)."""
    return render_template('index.html', prediction=None, confidence=None)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Case 1: multipart file upload -> HTML render or JSON (AJAX).
    Case 2: webcam/dataURL base64 -> JSON.
    """
    # ---- Case 1: file upload ----
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        raw_bytes = file.read()

        img_tensor = preprocess_image(raw_bytes)
        pred, conf, others = run_cls_inference(img_tensor)

        # Optional preview (base64 PNG) for UI
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception:
            data_url = None

        # AJAX -> JSON (no refresh)
        if request.form.get('ajax') == '1' or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'prediction': pred,
                'confidence': f"{conf:.2f}",
                'other_predictions': others or [],
                'uploaded_preview': data_url
            })

        # Fallback -> server-rendered page
        return render_template(
            'index.html',
            prediction=pred,
            confidence=f"{conf:.2f}",
            other_predictions=others,
            uploaded_preview=data_url
        )

    # ---- Case 2: base64 from webcam/snapshot ----
    b64_str = request.json.get('image_data') if request.is_json else request.form.get('image_data')
    if not b64_str:
        return jsonify({'error': 'No image provided'}), 400

    if ',' in b64_str:  # strip data URL header if present
        b64_str = b64_str.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(b64_str)
        img_tensor = preprocess_image(img_bytes)
        pred, conf, others = run_cls_inference(img_tensor)

        return jsonify({
            'prediction': pred,
            'confidence': f"{conf:.2f}",
            'other_predictions': others or []
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {e}'}), 400



# ======================================================
# ================== 3) APP START/STOP =================
# ======================================================

def start_detection_threads(camera_index=0):
    global camera, capture_thread, processing_thread
    if capture_thread and capture_thread.is_alive():
        return  # already running

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise IOError("Cannot open webcam.")

    capture_thread = threading.Thread(target=capture_frames, args=(camera,), daemon=True)
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    capture_thread.start()
    processing_thread.start()
    print(f"[YOLO] Camera opened on index {camera_index}. Threads started.")

def stop_detection_threads():
    global camera, capture_thread, processing_thread
    stop_event.set()
    if capture_thread:
        capture_thread.join(timeout=1)
    if processing_thread:
        processing_thread.join(timeout=1)
    if camera:
        camera.release()
        print("[YOLO] Camera released.")



def boot_detection():
    """Initialize YOLO detection threads."""
    try:
        start_detection_threads(camera_index=0)
        print("[YOLO] Detection threads started successfully.")
    except Exception as e:
        print(f"[YOLO] Startup error: {e}")


@app.route('/healthz')
def health():
    """Optional: quick health endpoint."""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    try:
        boot_detection()  # start YOLO threads once before running Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        print("Shutdown signal received.")
    finally:
        print("Stopping threads and releasing resources...")
        stop_detection_threads()
        print("Shutdown complete.")
