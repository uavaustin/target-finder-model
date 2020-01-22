# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim as contrib_slim

# models/research/slim
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

with open(
    os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"), "r"
) as stream:
    import yaml

    config = yaml.safe_load(stream)


slim = contrib_slim

CLASSES = config["classes"]["types"]
# num * 2 to account for targets and backgrounds
NUM_IMGS = config["generate"]["eval_batch"]["images"] * 2

tf.app.flags.DEFINE_integer("batch_size", 10, "The number of samples in each batch.")

tf.app.flags.DEFINE_integer(
    "max_num_batches", None, "Max number of batches to evaluate by default use all."
)

tf.app.flags.DEFINE_string("master", "", "The address of the TensorFlow master to use.")

tf.app.flags.DEFINE_string(
    "checkpoint_path",
    "",
    "The directory where the model was written to or an absolute path to a "
    "checkpoint file.",
)

tf.app.flags.DEFINE_string("eval_dir", "", "Directory where the results are saved to.")

tf.app.flags.DEFINE_integer(
    "num_preprocessing_threads", 4, "The number of threads used to create the batches."
)

tf.app.flags.DEFINE_string(
    "dataset_split_name", "val", "The name of the train/test split."
)

tf.app.flags.DEFINE_string(
    "dataset_dir",
    "model_data/records",
    "The directory where the dataset files are stored.",
)

tf.app.flags.DEFINE_string(
    "model_name", "", "The name of the architecture to evaluate."
)

tf.app.flags.DEFINE_string(
    "preprocessing_name",
    None,
    "The name of the preprocessing to use. If left "
    "as `None`, then the model_name flag is used.",
)

tf.app.flags.DEFINE_float(
    "moving_average_decay",
    None,
    "The decay to use for the moving average."
    "If left as None, then moving averages are not used.",
)

tf.app.flags.DEFINE_integer("eval_image_size", None, "Eval image size")

tf.app.flags.DEFINE_bool("quantize", False, "whether to use quantized graph or not.")

tf.app.flags.DEFINE_bool(
    "use_grayscale", False, "Whether to convert input images to grayscale."
)

FLAGS = tf.app.flags.FLAGS


def main(_):

    if not FLAGS.model_name:
        raise ValueError("Please specify a --model_name")
    elif not FLAGS.checkpoint_path:
        raise ValueError("Please specify a --checkpoint_path")
    elif not FLAGS.eval_dir:
        raise ValueError("Please specify an --eval_dir")
    else:
        pass

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        # Specify which TFRecord components to parse
        keys_to_features = {
            "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "image/format": tf.FixedLenFeature((), tf.string, default_value="png"),
            "image/class/label": tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)
            ),
        }

        items_to_handlers = {
            "image": slim.tfexample_decoder.Image(),
            "label": slim.tfexample_decoder.Tensor("image/class/label"),
        }

        items_to_descs = {
            "image": "Color image",
            "label": "Class idx",
        }

        label_idx_to_name = {}
        for i, label in enumerate(CLASSES):
            label_idx_to_name[i] = label

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers
        )

        file_pattern = "tfm_clf_%s.*"
        file_pattern = os.path.join(
            FLAGS.dataset_dir, file_pattern % FLAGS.dataset_split_name
        )

        dataset = slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=NUM_IMGS,
            items_to_descriptions=items_to_descs,
            num_classes=len(CLASSES),
            labels_to_names=label_idx_to_name,
        )

        #  Create model
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name, num_classes=dataset.num_classes, is_training=False
        )

        # Create a dataset provider to load data from the dataset
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size,
        )
        [image, label] = provider.get(["image", "label"])
        label -= FLAGS.labels_offset

        # Select the preprocessing function
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False, use_grayscale=FLAGS.use_grayscale
        )

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size,
        )

        # Define the model
        logits, _ = network_fn(images)

        if FLAGS.quantize:
            contrib_quantize.create_eval_graph()

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step
            )
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables()
            )
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
            {
                "Accuracy": slim.metrics.streaming_accuracy(predictions, labels),
                "Recall_5": slim.metrics.streaming_recall_at_k(logits, labels, 5),
            }
        )

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = "eval/%s" % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info("Evaluating %s" % checkpoint_path)

        slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore,
        )


if __name__ == "__main__":
    tf.app.run()
