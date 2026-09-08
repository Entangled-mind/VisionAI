# VisionAI — Intelligent Face & Object Recognition System

A production-grade, educational Computer Vision and Machine Learning application that performs real-time face detection, 128-dimensional facial embedding recognition, YOLOv8 object detection, event logging into SQLite, statistical analytics with Pandas, and interactive visualization via Streamlit.

---

## 🏗️ Project Architecture

```text
VisionAI/
│
├── .gitignore                   # Ignores .venv, cache, temporary files, models
├── requirements.txt             # Pinned production dependencies
├── config.py                    # Centralized path, model, and threshold configurations
├── README.md                    # Project documentation & master learning guide
│
├── data/
│   ├── faces/                   # Registered face identities (e.g. Lena/, Priyanka/)
│   ├── test_images/             # Static evaluation images (sample.jpg, face_sample.jpg)
│   ├── processed/               # Grayscale, resized, and annotated output images
│   ├── analytics/               # Generated Matplotlib charts and graphs
│   └── events.db                # SQLite database storing real-time detection events
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx   # OpenCV YuNet deep face detector
│   ├── face_recognition_sface_2021dec.onnx # OpenCV SFace 128-D embedding extractor
│   ├── yolov8n.pt                          # Ultralytics YOLOv8 nano weights
│   └── registered_faces.pkl                # Serialized identity embeddings database
│
├── src/
│   ├── __init__.py              # Core package initializer
│   ├── image_basics.py          # Milestone 1: Image loading, numerical inspection, resizing
│   ├── webcam_test.py           # Milestone 1: Camera capture, real-time FPS calculation
│   ├── face_detection.py        # Milestone 2: YuNet deep face & landmark detector
│   ├── embeddings.py            # Milestone 3: SFace 128-D embedding extractor & L2 normalization
│   ├── face_recognition.py      # Milestone 3: Identity registry & Cosine Similarity matching
│   ├── object_detection.py      # Milestone 4: Pretrained YOLOv8 object detector
│   ├── vision_engine.py         # Milestone 5: Combined Vision Engine (Spatial Fusion)
│   ├── database.py              # Milestone 6: SQLite Event Logger & Cooldown Manager
│   ├── analytics.py             # Milestone 7: Pandas Metrics & Matplotlib Visualizations
│   ├── evaluation.py            # Milestone 9: FAR, FRR, IoU, and Precision/Recall benchmarks
│   └── utils.py                 # Geometric helpers, styled boxes, and model downloaders
│
├── app/
│   └── app.py                   # Milestone 8: Multi-page Streamlit Web Application
│
├── tests/
│   ├── __init__.py
│   ├── test_milestone1.py       # Computer vision primitives unit tests
│   └── test_vision_pipeline.py  # End-to-end multi-model pipeline unit tests
│
└── generate_masterclass_pdf.py  # Compiles VisionAI_Masterclass_Guide.pdf
```

---

## ⚡ Quickstart Guide

### 1. Environment Activation
```powershell
# Open terminal inside the project directory:
cd C:\Users\priy2\.gemini\antigravity\scratch\VisionAI

# Activate virtual environment:
.\.venv\Scripts\Activate.ps1
```

### 2. Launch the Streamlit Web Application (Milestone 8)
```powershell
streamlit run app/app.py
```
This opens the multi-page web dashboard in your browser (`http://localhost:8501`) with:
* 📷 **Live Vision Feed**: Real-time webcam with combined Face Recognition + YOLO + Live SQLite Logging
* 👤 **Face Registration**: Register new identities via upload or webcam
* 🔍 **Static Image Inspector**: Test any photo with dynamic threshold tuning sliders
* 🗄️ **Event Database**: Filter, search, and export detection history to CSV
* 📊 **Visual Analytics**: Interactive KPI metrics, bar charts, and timeline graphs

---

## 💻 Running Individual Modules (CLI)

### Milestone 1 — Image Processing Fundamentals
```powershell
python src/image_basics.py --image data/test_images/sample.jpg
python src/webcam_test.py
```

### Milestone 2 — Face Detection (YuNet ONNX)
```powershell
python src/face_detection.py --image data/test_images/face_sample.jpg
# Or live camera:
python src/face_detection.py
```

### Milestone 3 — Face Recognition & Registration
```powershell
# Build/update embeddings database from data/faces/ folder:
python src/face_recognition.py --register-all

# Test recognition on a static image:
python src/face_recognition.py --image data/test_images/face_sample.jpg
```

### Milestone 4 — YOLOv8 Object Detection
```powershell
python src/object_detection.py --image data/test_images/face_sample.jpg
```

### Milestone 5 — Combined Vision Engine (Spatial Fusion)
```powershell
python src/vision_engine.py --image data/test_images/face_sample.jpg
```

### Milestone 6 — SQLite Event Database
```powershell
python src/database.py
```

### Milestone 7 — Analytics & Visualizations
```powershell
python src/analytics.py
```

### Milestone 9 — Quantitative Model Evaluation
```powershell
python src/evaluation.py
python tests/test_vision_pipeline.py
```

---

## 🧠 Key Mathematical & Theoretical Concepts

### 1. Face Recognition via Deep Metric Learning
Instead of classifying faces into fixed categories (which fails whenever a new person joins), VisionAI maps facial images into a continuous **128-dimensional embedding space** $\mathbb{R}^{128}$ using SFace.
Faces of the same person map to vectors pointing in almost the same direction, while faces of different people point in divergent directions.

### 2. Cosine Similarity vs. Euclidean Distance
All embeddings are normalized to unit length: $\|\vec{A}\|_2 = 1.0$.
$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\|_2 \|\vec{B}\|_2} = \sum_{i=1}^{128} A_i B_i$$
* **Similarity $\ge 0.50$**: Match recognized identity.
* **Similarity $< 0.50$**: Classified as `Unknown`.

### 3. Precision, Recall, and Accuracy
* **Similarity**: Geometric angle between two representation vectors.
* **Confidence**: Softmax probability output from an object detector indicating existence likelihood.
* **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$ (Total correct decisions over all trials).
* **False Acceptance Rate (FAR)**: Proportion of impostors erroneously accepted as known.
* **False Rejection Rate (FRR)**: Proportion of genuine identities erroneously rejected as unknown.

### 4. Object Detection (YOLOv8) & Non-Maximum Suppression (NMS)
* **Single-Shot Regression**: YOLO treats object detection as a single regression problem, predicting bounding box coordinates $(x, y, w, h)$ and class probabilities directly from full images in a single forward pass.
* **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
* **NMS**: Merges highly overlapping duplicate boxes (where $\text{IoU} > 0.45$) to keep only the highest-confidence bounding box.

---

## 🔒 Privacy and Ethics Statement

* **Local Inference**: All video frames, facial landmarks, and embedding vectors are processed and stored **exclusively on local hardware**.
* **Consent First**: No individual should have their face registered into the recognition database without prior informed consent.
* **Educational Purpose**: This project is built as an educational demonstration of computer vision algorithms. It is not designed or certified for surveillance, biometric authentication, or access-control decisions.
