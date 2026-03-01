# 🔍 DeepFake Images Detector

A binary CNN classifier that detects whether a face image is **Real** or **AI-generated (Fake)**, built with TensorFlow/Keras and an interactive Jupyter/Kaggle widget UI.

---

## 📁 Project Structure

```
deepfake-detector-system/
├── src/
│   └── model.py          # Model architecture, training & evaluation pipeline
├── app/
│   └── app.py            # Interactive ipywidgets GUI (Kaggle / Jupyter)
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

A 3-block CNN trained for binary classification (`Fake = 0`, `Real = 1`):

```
Conv(32) → BatchNorm → MaxPool → Dropout(0.25)
Conv(64) → BatchNorm → MaxPool → Dropout(0.25)
Conv(128)→ BatchNorm → MaxPool → Dropout(0.30)
Dense(128) → BatchNorm → Dropout(0.50)
Dense(1, sigmoid)
```

- **Input size:** 128 × 128 × 3  
- **Optimizer:** Adam (lr = 0.001)  
- **Loss:** Binary cross-entropy  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  

---

## 📊 Dataset

[deepfake-and-real-images](https://www.kaggle.com/datasets) — organised as:

```
Dataset/
├── Train/
│   ├── Fake/
│   └── Real/
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the model (run in Kaggle or locally with the dataset mounted)

```bash
python src/model.py
```

This saves `final_deepfake_model.h5` to `/kaggle/working/` (or the current directory).

### 3 — Launch the GUI

Open a Jupyter / Kaggle notebook and run:

```python
%run app/app.py
```

---

## 🖥️ App Features

| Tab | Description |
|-----|-------------|
| 🔍 Detect | Single-image or batch prediction with confidence scores |
| 📊 Dashboard | Live charts — real/fake distribution & confidence histogram |
| 📋 History | Scrollable log of all predictions; export to CSV |

**Detection modes:**
- **Basic** — label + confidence donut chart  
- **Diagnostic** — label + confidence + probability bar chart  

---

## 📈 Evaluation Metrics

The pipeline reports **accuracy, precision, recall**, a **confusion matrix**, and a full **classification report** on the held-out test set.

---

## 🛠️ Requirements

See `requirements.txt`. Core dependencies:

- Python ≥ 3.9  
- TensorFlow ≥ 2.10  
- OpenCV, Pillow, NumPy, scikit-learn  
- ipywidgets, matplotlib, pandas  

---

## 👤 Author

**huda-usman** — ML Semester Project
