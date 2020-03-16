#!/usr/bin/env python3
"""
A classification model trainer with Tensorflow 2.x.

Check here for a list of prebuilt Keras models: 
https://keras.io/applications/.
"""

from typing import Tuple
import pathlib 
import os 

import tensorflow as tf 

with pathlib.Path("config.yaml").open("r") as f:
    import yaml
    config = yaml.safe_load(f)

CLASSES = {}
for idx, item in enumerate(config["classes"]["types"]):
    CLASSES[item] = idx

IMG_SHAPE = (
    config["inputs"]["preclf"]["width"], 
    config["inputs"]["preclf"]["height"], 
    3
)

NUM_CLASSES = len(config["classes"]["types"])
BATCH_SIZE = 32
TRAIN_DIR = pathlib.Path("scripts_generate/data/clf_train/images")
EVAL_DIR = pathlib.Path("scripts_generate/data/clf_val/images")
IMG_EXT = config["generate"]["img_ext"]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def process_file(file_path) -> Tuple[tf.Tensor, int]:
    """Function that takes path to image and preprocesses it."""

    img = tf.io.read_file(file_path)
    # Expand animations because png has alpha channel
    img = tf.image.decode_image(img, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = (img / 127.5) - 1
    img = tf.image.resize(
            img, 
            [
                config["inputs"]["preclf"]["width"], 
                config["inputs"]["preclf"]["height"]
            ]
        )

    return img, file_path


def create_datasets() -> tf.data.Dataset:
    """Create the train and eval datasets."""
    train_ds = tf.data.Dataset.list_files(str(TRAIN_DIR / f"*.{IMG_EXT}"))
    train_ds = train_ds.map(process_file, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.list_files(str(EVAL_DIR / f"*.{IMG_EXT}"))
    val_ds = val_ds.map(process_file, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    
    return train_ds, val_ds


def create_model() -> tf.keras.models.Model:
    """
    Create a model using the keras prebuilt backbones.
    Add a classification head after the convolutional layers.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    base_model.trainable = True  # We want to train the classifier's params.
    # Use the global pooling to take any [N, M] matrix to [1, 1]
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    # Take the output of the layer above and linear layer to output classes
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES)

    model = tf.keras.Sequential([
        base_model,
        global_pooling,
        prediction_layer,
        tf.keras.layers.Softmax()
    ])    

    # Compile the model. Use non-logits because these are softmax outputs
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=1e-3,
            momentum=0.9,
            nesterov=True,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
    )

    return model


def path_to_classes(y: tf.Tensor) -> tf.Tensor:
    """Converts tensor of paths to tensor of class ids."""
    y = [pathlib.Path(path.decode()) for path in y.numpy()]
    y = [path.name.split('_')[0] for path in y]
    y = tf.convert_to_tensor(
        [[0, 1] if int(CLASSES[name]) == 0 else [1, 0] for name in y]
        )
    return y

def loss(model, x, y, loss_fn, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    y = path_to_classes(y)
    return loss_fn(y_true=y, y_pred=y_), y


def grad(model, inputs, targets, loss_fn):
    with tf.GradientTape() as tape:
        loss_value, y = loss(model, inputs, targets, loss_fn, training=True)
    return loss_value, y, tape.gradient(loss_value, model.trainable_variables)


def train() -> None:
    """Main training loop."""
    # Get the model
    model = create_model()
    # Get the training and eval sets
    train_ds, eval_ds = create_datasets()

    # Loss function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # Optimizer, choosing stochastic grad descent with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    
    # Training loop
    for epoch in range(200):
        epoch_loss_avg = tf.keras.metrics.Mean()      
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
        
        # Train
        for idx, (x, y) in enumerate(train_ds):
            loss_value, y, grads = grad(model, x, y, loss_fn)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg(loss_value)  # Add current batch loss
            if idx % 100 == 0:
                print(f"Epoch {epoch}: Loss: {epoch_loss_avg.result()}.")

        # Evaluate
        for x, y, in eval_ds:
            epoch_accuracy(path_to_classes(y), model(x, training=False))

        print(f"Eval Accuracy: {epoch_accuracy.result()}.")
        
    return None


if __name__ == '__main__':
    train()

