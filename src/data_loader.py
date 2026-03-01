"""
src/data_loader.py
==================
Dataset verification and Keras ImageDataGenerator builders.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# Constants (imported by other modules)
# ──────────────────────────────────────────────

# Auto-detect environment (Kaggle vs Local)
if os.path.exists("/kaggle/input"):
    BASE_PATH = "/kaggle/input/deepfake-and-real-images/Dataset"
else:
    BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Dataset")

TRAIN_DIR  = os.path.join(BASE_PATH, "Train")
VAL_DIR    = os.path.join(BASE_PATH, "Validation")
TEST_DIR   = os.path.join(BASE_PATH, "Test")

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
SEED       = 42
IMG_EXTS   = (".jpg", ".jpeg", ".png")


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def verify_dataset() -> dict:
    """
    Print class counts for every split and return Keras class-index mapping.

    Returns
    -------
    dict  e.g. {"Fake": 0, "Real": 1}
    """
    print("\n📂 Dataset structure")
    print("─" * 50)

    for split in ("Train", "Validation", "Test"):
        split_path = os.path.join(BASE_PATH, split)
        for cls in ("Fake", "Real"):
            folder = os.path.join(split_path, cls)
            count = (
                len([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])
                if os.path.exists(folder) else 0
            )
            print(f"  {split:12s} | {cls}: {count}")

    # Tiny generator — used only to read Keras's automatic folder→label mapping
    temp_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        TRAIN_DIR,
        target_size=(32, 32),
        batch_size=10,
        class_mode="binary",
        shuffle=False,
    )
    mapping = temp_gen.class_indices
    print(f"\n✅ Keras class mapping: {mapping}")

    if mapping.get("Fake") != 0 or mapping.get("Real") != 1:
        print("⚠️  Unexpected mapping — update interpretation logic accordingly.")

    return mapping


def build_generators():
    """
    Create train / validation / test ImageDataGenerators.

    Returns
    -------
    (train_gen, val_gen, test_gen)
    """
    rescale = dict(rescale=1.0 / 255)

    train_gen = ImageDataGenerator(**rescale).flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    val_gen = ImageDataGenerator(**rescale).flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    test_gen = ImageDataGenerator(**rescale).flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen
