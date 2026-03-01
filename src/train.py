"""
src/train.py
============
Training entry point.

Usage
-----
    python src/train.py
"""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loader import build_generators, verify_dataset, BATCH_SIZE
from model import build_model

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

WORKING_PATH = "/kaggle/working"
EPOCHS       = 15
SEED         = 42


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def set_seeds(seed: int = SEED) -> None:
    """Fix all random seeds for fully reproducible runs."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"]       = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def train(model: keras.Model, train_gen, val_gen) -> keras.callbacks.History:
    """
    Fit the model with early stopping, LR scheduling, and checkpointing.

    Parameters
    ----------
    model     : compiled Keras model
    train_gen : training ImageDataGenerator
    val_gen   : validation ImageDataGenerator

    Returns
    -------
    keras History object
    """
    os.makedirs(WORKING_PATH, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(WORKING_PATH, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=min(200, train_gen.samples // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=min(50, val_gen.samples // BATCH_SIZE),
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(WORKING_PATH, "final_deepfake_model.h5")
    model.save(final_path)
    print(f"\n💾 Model saved → {final_path}")

    return history


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 DeepFake Detector — Training")
    print("=" * 50)
    print(f"TensorFlow : {tf.__version__}")
    print(f"GPU devices: {len(tf.config.list_physical_devices('GPU'))}")

    set_seeds(SEED)
    verify_dataset()

    train_gen, val_gen, _ = build_generators()

    model = build_model()
    model.summary()

    train(model, train_gen, val_gen)
    print("\n✅ Training complete.")
