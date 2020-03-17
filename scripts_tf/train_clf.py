#!/usr/bin/env python3
"""
A classification model trainer with Tensorflow 2.x.

Check here for a list of prebuilt Keras models: 
https://keras.io/applications/.
"""

from typing import Tuple, List
import pathlib 
import os 

import tensorflow as tf 
import numpy as np

with pathlib.Path("config.yaml").open("r") as f:
    import yaml
    config = yaml.safe_load(f)

CLASSES = {}
for idx, item in enumerate(config["classes"]["types"]):
    CLASSES[item] = idx

IMG_DIM = 160
IMG_SHAPE = (
    IMG_DIM, 
    IMG_DIM, 
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
    img = tf.image.resize(img, [IMG_DIM, IMG_DIM])

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
        input_shape=IMG_SHAPE, include_top=False)

    base_model.trainable = True  # We want to train the classifier's params.
    # Use the global pooling to take any [N, M] matrix to [1, 1]
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    # Take the output of the layer above and linear layer to output classes
    prediction_layer = tf.keras.layers.Dense(2)

    model = tf.keras.Sequential([
        base_model,
        global_pooling,
        prediction_layer,
    ])    
    
    return model


def path_to_classes(y: tf.Tensor) -> tf.Tensor:
    """Converts tensor of paths to tensor of class ids."""
    y = [pathlib.Path(path.decode()) for path in y.numpy()]
    y = [path.name.split('_')[0] for path in y]
    
    #y = tf.convert_to_tensor(
    #        [[0, 1] if int(CLASSES[name]) == 0 else [1, 0] for name in y]
    #    )
    
    y = tf.convert_to_tensor([int(CLASSES[name]) for name in y])

    return y


def loss(model, x: tf.Tensor, y: tf.Tensor, training: bool):
    # https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    y_ = model(x, training=training)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)


def grad(model, inputs: tf.Tensor, targets: tf.Tensor):
    """Use the GradientTape method to calculate gradients of model."""
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train() -> None:
    """Main training loop."""
    # Get the model
    model = create_model()
    # Get the training and eval sets
    train_ds, eval_ds = create_datasets()

    # Optimizer, choosing stochastic grad descent with momentum
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)
    
    # Training loop
    for epoch in range(200):
        # Create the metrics
        epoch_loss_avg = tf.keras.metrics.Mean()      
        test_accuracy = tf.keras.metrics.Accuracy()
        
        # Train
        for idx, (x, y) in enumerate(train_ds):
            y = path_to_classes(y)
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg(loss_value)  # Add current batch loss
            
            if idx % 100 == 0:
                # Evaluate
                for x, y, in eval_ds:
                    logits = model(x, training=False)
                    print(tf.nn.softmax(logits))
                    predictions = tf.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int32)
                    test_accuracy(predictions, tf.argmax(path_to_classes(y), axis=1))

                print(f"Epoch {epoch}: Loss: {epoch_loss_avg.result()}: Eval Accuracy: {test_accuracy.result()}.")
        
    return None


if __name__ == '__main__':
    train()

