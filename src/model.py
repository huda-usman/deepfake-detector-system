"""
src/model.py
============
CNN model definition only.
Import build_model() wherever you need to create or reload the architecture.
"""

from tensorflow import keras
from tensorflow.keras import layers

from data_loader import IMG_SIZE


def build_model() -> keras.Model:
    """
    Build and compile the DeepFake detector CNN.

    Architecture
    ------------
    3 convolutional blocks (Conv -> BatchNorm -> MaxPool -> Dropout)
    followed by a Dense classifier head with sigmoid output.

    Label convention:  Fake = 0  |  Real = 1

    Returns
    -------
    Compiled keras.Sequential model
    """
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(32,  (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64,  (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(1, activation="sigmoid"),

    ], name="deepfake_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
