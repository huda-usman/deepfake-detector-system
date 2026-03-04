<div align="center">
<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a0a1a,100:7b2ff7&height=120&text=DeepFake%20Image%20Detector&fontSize=36&fontColor=ffffff&fontAlignY=55&desc=Can%20you%20tell%20what%27s%20real%3F%20Our%20CNN%20can.&descAlignY=78&descSize=14" width="100%"/>
# 🔍 DeepFake Image Detector

### *Can you tell what's real? Our CNN can.*

A deep learning system that detects AI-generated (fake) face images with high confidence — featuring a full interactive GUI, batch detection, and live analytics.

<br>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Ready-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br>

</div>

---

## ✨ What It Does

> Upload a face image → get an instant **REAL** or **FAKE** verdict with confidence score, probability breakdown, and visual analytics.

| | Feature |
|---|---|
| 🤖 | **CNN classifier** trained on thousands of real and AI-generated face images |
| 🖼️ | **Single & batch detection** — analyze one image or hundreds at once |
| 📊 | **Live analytics dashboard** — confidence charts, distribution graphs |
| 📋 | **Prediction history** — full log with CSV export |
| ⚡ | **Runs on Kaggle** — no installation or setup needed |

---

## 🚀 Quickstart — Run in 4 Steps on Kaggle

> No installation. No configuration. Just click and run.

```
1. Go to kaggle.com and create a free account
        ↓
2. Add the dataset → search "deepfake-and-real-images" in Kaggle datasets
        ↓
3. Open notebooks/deepfake_detector.ipynb in a new Kaggle notebook
        ↓
4. Click "Run All" — training + evaluation + interactive app launches automatically ✅
```

---

## 🧠 Model Architecture

```
Input Image (128 × 128 × 3)
         │
         ▼
┌─────────────────────────────┐
│  Conv2D(32)  + BatchNorm    │  ← Block 1
│  MaxPool    + Dropout(0.25) │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Conv2D(64)  + BatchNorm    │  ← Block 2
│  MaxPool    + Dropout(0.25) │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Conv2D(128) + BatchNorm    │  ← Block 3
│  MaxPool    + Dropout(0.30) │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Dense(128)  + BatchNorm    │  ← Classifier Head
│  Dropout(0.50)              │
│  Dense(1, sigmoid)          │
└─────────────────────────────┘
         │
         ▼
   🤖 FAKE (0)  or  👤 REAL (1)
```

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr = 0.001) |
| Loss | Binary Cross-Entropy |
| Input Size | 128 × 128 × 3 |
| Early Stopping | patience = 5 |
| LR Scheduler | ReduceLROnPlateau |

---

## 🖥️ Interactive App

The app runs directly inside the Kaggle notebook with 3 tabs:

### 🔍 Detect Tab
Upload a single image or a batch — choose between two modes:
- **Basic Mode** → Label + confidence donut chart
- **Diagnostic Mode** → Label + confidence + full probability bar chart

### 📊 Dashboard Tab
Live statistics that update after every detection:
- Real vs Fake distribution pie chart
- Confidence score histogram across all predictions

### 📋 History Tab
- Scrollable log of every prediction made
- Timestamp, filename, result, confidence, raw score
- One-click **CSV export**

---

## 📁 Project Structure

```
deepfake-detector-system/
│
├── 📓 notebooks/
│   └── deepfake_detector.ipynb   ← ⭐ Start here
│
├── 🧠 src/
│   ├── data_loader.py            ← Dataset paths + generators
│   ├── model.py                  ← CNN architecture
│   ├── train.py                  ← Training loop + callbacks
│   ├── evaluate.py               ← Metrics + confusion matrix
│   └── predict.py                ← Single & batch inference
│
├── 🖥️ app/
│   └── app.py                    ← Interactive GUI
│
├── 📦 data/                      ← Dataset (not tracked by Git)
├── 💾 models/                    ← Saved .h5 files (not tracked)
├── 📤 outputs/                   ← Evaluation outputs
│
├── requirements.txt
└── README.md
```

---

## 💻 Run Locally

<details>
<summary><b>Click to expand local setup instructions</b></summary>

### 1. Clone the repo
```bash
git clone https://github.com/huda-usman/deepfake-detector-system
cd deepfake-detector-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Get it from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) and place it as:
```
data/
└── Dataset/
    ├── Train/   → Fake/  Real/
    ├── Validation/ → Fake/  Real/
    └── Test/    → Fake/  Real/
```

### 4. Train
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

</details>

---

## 📦 Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| TensorFlow | ≥ 2.10 |
| OpenCV | ≥ 4.6 |
| scikit-learn | ≥ 1.1 |
| ipywidgets | ≥ 8.0 |
| pandas | ≥ 1.5 |
| matplotlib | ≥ 3.6 |

```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset

[deepfake-and-real-images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) on Kaggle

| Split | Fake | Real |
|-------|------|------|
| Train | ✅ | ✅ |
| Validation | ✅ | ✅ |
| Test | ✅ | ✅ |

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:7b2ff7,100:0a0a1a&height=80&section=footer" width="100%"/>

### 🙋‍♀️ Connect with Me

Developed by **Huda Usman**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Huda%20Usman-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hudausman010)

<br/>

⭐ **If you found this project interesting, please give it a star!** ⭐

</div>
