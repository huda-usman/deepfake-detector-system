"""
src/predict.py
==============
Single-image and batch inference helpers.

Usage (CLI)
-----------
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --image path/to/image.jpg --model path/to/model.h5
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from data_loader import IMG_SIZE

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DEFAULT_MODEL_PATHS = [
    "/kaggle/working/final_deepfake_model.h5",
    "/kaggle/working/best_model.h5",
    "final_deepfake_model.h5",
    "best_model.h5",
]


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────

def load_model(path: str | None = None) -> tf.keras.Model:
    """
    Load a saved Keras model.

    Parameters
    ----------
    path : explicit file path, or None to auto-search DEFAULT_MODEL_PATHS
    """
    search = [path] if path else DEFAULT_MODEL_PATHS
    for p in search:
        if p and os.path.exists(p):
            model = tf.keras.models.load_model(p)
            print(f"✅ Model loaded from: {p}")
            return model
    raise FileNotFoundError(
        "No model file found. Run src/train.py first, or pass --model <path>."
    )


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def preprocess(image: "str | np.ndarray") -> np.ndarray:
    """
    Load (if path) and preprocess an image for inference.

    Parameters
    ----------
    image : file path (str) or an RGB numpy array (H x W x 3)

    Returns
    -------
    np.ndarray of shape (1, H, W, 3), dtype float32, values in [0, 1]
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(image)

    # Drop alpha channel
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    img = cv2.resize(img, IMG_SIZE).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict_image(model: tf.keras.Model, image: "str | np.ndarray") -> dict:
    """
    Predict whether a single image is REAL or FAKE.

    Model convention
    ----------------
    raw score > 0.5  ->  REAL (label = 1)
    raw score <= 0.5 ->  FAKE (label = 0)

    Parameters
    ----------
    model : loaded Keras model
    image : file path or RGB numpy array

    Returns
    -------
    dict
        raw_prediction   : float in [0, 1]
        prediction       : "REAL" or "FAKE"
        real_probability : float (%)
        fake_probability : float (%)
        confidence       : float (%) — probability of the predicted class
    """
    x   = preprocess(image)
    raw = float(model.predict(x, verbose=0)[0][0])

    is_real = raw > 0.5
    return {
        "raw_prediction":   raw,
        "prediction":       "REAL" if is_real else "FAKE",
        "real_probability": raw * 100,
        "fake_probability": (1 - raw) * 100,
        "confidence":       max(raw, 1 - raw) * 100,
    }


def predict_batch(model: tf.keras.Model,
                  image_paths: list[str]) -> list[dict]:
    """
    Run predict_image() on a list of file paths.

    Returns
    -------
    List of result dicts (same structure as predict_image), each with
    an added "filename" key.
    """
    results = []
    for path in image_paths:
        try:
            res = predict_image(model, path)
            res["filename"] = os.path.basename(path)
        except Exception as exc:
            res = {"filename": os.path.basename(path), "error": str(exc)}
        results.append(res)
    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFake detector — single image prediction")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a saved .h5 model file")
    args = parser.parse_args()

    model  = load_model(args.model)
    result = predict_image(model, args.image)

    emoji = "👤" if result["prediction"] == "REAL" else "🤖"
    print(f"\n{emoji}  Prediction  : {result['prediction']}")
    print(f"   Confidence  : {result['confidence']:.1f}%")
    print(f"   Real prob   : {result['real_probability']:.1f}%")
    print(f"   Fake prob   : {result['fake_probability']:.1f}%")
    print(f"   Raw score   : {result['raw_prediction']:.6f}")
