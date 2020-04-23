#!/usr/bin/env python3
"""
A classification model trainer with Tensorflow 2.1.

Check here for a list of prebuilt Keras models: 
https://keras.io/applications/.
"""

from typing import Tuple, List
import pathlib
from datetime import datetime

import tensorflow as tf
import numpy as np
from matplotlib import pyplot

with pathlib.Path("config.yaml").open("r") as f:
    import yaml

    config = yaml.safe_load(f)

IMG_SHAPE = [
    config["inputs"]["preclf"]["width"],
    config["inputs"]["preclf"]["height"],
    3,
]

NUM_CLASSES = len(config["classes"]["types"])
BATCH_SIZE = 64
DATA_DIR = pathlib.Path("scripts_generate/data")
IMG_EXT = config["generate"]["img_ext"]
MODEL_SAVE_DIR = pathlib.Path("~/runs").expanduser() / "MobileNetV2"


def create_datasets() -> tf.data.Dataset:
    """Create the train and eval datasets using keras API."""

    # Create the preprocessing functions
    def normalize_and_resize(img: tf.Tensor) -> tf.Tensor:
        # This function is specific to mobilnetV2
        return tf.keras.applications.mobilenet_v2.preprocess_input(img)

    preprocess_fn = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        dtype=tf.float32,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=normalize_and_resize,
    )

    train_ds = preprocess_fn.flow_from_directory(
        DATA_DIR / "clf_train",
        batch_size=BATCH_SIZE,
        target_size=tuple(IMG_SHAPE[:2]),
        shuffle=True,
        class_mode="binary",
        seed=42,
    )
    eval_ds = preprocess_fn.flow_from_directory(
        DATA_DIR / "clf_val",
        batch_size=BATCH_SIZE,
        target_size=tuple(IMG_SHAPE[:2]),
        shuffle=False,
        class_mode="binary",
        seed=42,
    )

    return train_ds, eval_ds


def create_model() -> tf.keras.models.Model:
    """ Create a model using the keras prebuilt backbones.
    Add a classification head after the convolutional layers."""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights=None, alpha=0.5
    )

    base_model.trainable = True  # We want to train the classifier's params.
    # Use the global pooling to take any [N, M] matrix to [1, 1]
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    # Take the output of the layer above and linear layer to output classes
    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")
    # Add these layers on top of the model now
    model = tf.keras.Sequential([base_model, global_pooling, prediction_layer,])

    return model


def train(run_dir: pathlib.Path) -> None:
    """Main training function."""

    run_dir.mkdir(parents=True)

    # Get the model
    model = create_model()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            lr=1e-1, decay=1e-4, momentum=0.9, nesterov=True
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Get the training and eval sets
    train_ds, eval_ds = create_datasets()
    # Create a Callback to monitor validation accuracy
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.001,
        verbose=1,
        restore_best_weights=True,
        patience=4,
    )
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(run_dir / "checkpoints" / "ckpts"),
        save_weights_only=True,
        verbose=1,
    )

    # TODO(alex) Use this history for plotting or TensorBoard logging
    history = model.fit(
        train_ds,
        epochs=300,
        validation_data=eval_ds,
        use_multiprocessing=True,
        shuffle=False,
        validation_steps=100,
        callbacks=[callback, cp_callback],
    )

    save_dir = run_dir / "saved_model"
    model.save(str(save_dir), save_format="tf")


if __name__ == "__main__":
    run_dir = MODEL_SAVE_DIR / datetime.now().isoformat(" ")
    train(run_dir)
