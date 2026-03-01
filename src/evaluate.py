"""
src/evaluate.py
===============
Evaluate a saved model on the test set.

Usage
-----
    python src/evaluate.py
    python src/evaluate.py --model /path/to/model.h5
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import build_generators, BATCH_SIZE

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
# Helpers
# ──────────────────────────────────────────────

def load_model(path: str | None = None) -> tf.keras.Model:
    """
    Load a saved Keras model.

    Parameters
    ----------
    path : explicit file path, or None to auto-search DEFAULT_MODEL_PATHS

    Returns
    -------
    Loaded tf.keras.Model
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


def evaluate(model: tf.keras.Model, test_gen) -> dict:
    """
    Evaluate model on test_gen and print a full report.

    Returns
    -------
    dict with keys: loss, accuracy, precision, recall
    """
    steps = test_gen.samples // BATCH_SIZE
    loss, acc, prec, rec = model.evaluate(test_gen, steps=steps, verbose=1)

    print(f"\n📊 Test Results")
    print("─" * 40)
    print(f"  Loss      : {loss:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")

    # Confusion matrix & classification report
    test_gen.reset()
    y_true = test_gen.classes
    y_pred = (model.predict(test_gen, verbose=0).flatten() >= 0.5).astype(int)

    print("\n📌 Confusion Matrix  (Fake = 0 | Real = 1)")
    print(confusion_matrix(y_true, y_pred))

    print("\n📌 Classification Report")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

    return {"loss": loss, "accuracy": acc, "precision": prec, "recall": rec}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the DeepFake detector")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a saved .h5 model file")
    args = parser.parse_args()

    print("📊 DeepFake Detector — Evaluation")
    print("=" * 50)

    model = load_model(args.model)
    _, _, test_gen = build_generators()
    evaluate(model, test_gen)
