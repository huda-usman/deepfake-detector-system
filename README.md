# 🔍 DeepFake Images Detector

A binary CNN classifier that detects whether a face image is **Real** or **AI-generated (Fake)**, with a full interactive GUI built using ipywidgets.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange) ![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Quickest Way to Run (Kaggle — No Setup Needed)

1. **Open Kaggle** → [kaggle.com](https://www.kaggle.com)
2. **Add the dataset** → Search for [deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) and add it to your notebook inputs
3. **Open the notebook** → `notebooks/deepfake_detector.ipynb`
4. Click **Run All** — training + evaluation + interactive app launches automatically

That's it. No installation, no configuration.

---

## 📁 Project Structure

```
deepfake-detector-system/
├── data/                        # Local dataset (not tracked by Git)
├── models/                      # Saved model files (not tracked by Git)
├── notebooks/
│   └── deepfake_detector.ipynb  # ⭐ Ready-to-run Kaggle notebook
├── outputs/                     # Evaluation outputs
├── src/
│   ├── data_loader.py           # Dataset paths + ImageDataGenerators
│   ├── model.py                 # CNN architecture
│   ├── train.py                 # Training loop + callbacks
│   ├── evaluate.py              # Test set evaluation + metrics
│   └── predict.py               # Single image + batch inference
├── app/
│   └── app.py                   # Interactive ipywidgets GUI
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

3-block CNN trained for binary classification:

```
Input (128×128×3)
    │
Conv(32) → BatchNorm → MaxPool → Dropout(0.25)
    │
Conv(64) → BatchNorm → MaxPool → Dropout(0.25)
    │
Conv(128)→ BatchNorm → MaxPool → Dropout(0.30)
    │
Dense(128) → BatchNorm → Dropout(0.50)
    │
Dense(1, sigmoid)  →  Fake = 0 | Real = 1
```

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Binary Cross-Entropy
- **Callbacks:** EarlyStopping · ReduceLROnPlateau · ModelCheckpoint

---

## 🖥️ App Features

| Tab | What it does |
|-----|-------------|
| 🔍 **Detect** | Upload a single image or batch — get Real/Fake prediction with confidence |
| 📊 **Dashboard** | Live charts showing Real vs Fake distribution + confidence histogram |
| 📋 **History** | Scrollable log of all predictions — export to CSV |

Two detection modes:
- **Basic** — Label + confidence donut chart
- **Diagnostic** — Label + confidence + full probability breakdown

---

## 💻 Run Locally (Alternative)

### 1. Clone the repo
```bash
git clone https://github.com/huda-usman/deepfake-detector-system
cd deepfake-detector-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) and place it at:
```
data/
└── Dataset/
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

### 4. Train the model
```bash
python src/train.py
```

### 5. Evaluate
```bash
python src/evaluate.py
```

### 6. Predict a single image
```bash
python src/predict.py --image path/to/face.jpg
```

### 7. Launch the app
```bash
jupyter notebook notebooks/deepfake_detector.ipynb
```

---

## 📦 Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| TensorFlow | ≥ 2.10 |
| OpenCV | ≥ 4.6 |
| scikit-learn | ≥ 1.1 |
| ipywidgets | ≥ 8.0 |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

**huda-usman** — ML Semester Project
