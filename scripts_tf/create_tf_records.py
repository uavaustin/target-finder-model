# Adapted from:
# https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
AFTER running create_full_images.py and create_detection_data.py:

$ python scripts_tf/create_tf_records.py --image_dir ./scripts_generate/data
"""
import hashlib
import random
import io
import json
import os
import contextlib2
import numpy as np
from PIL import Image
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util

with open(
    os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"), "r"
) as stream:
    import yaml

    config = yaml.safe_load(stream)

CLASSES = config["classes"]["shapes"] + config["classes"]["alphas"]

CLF_CLASSES = config["classes"]["types"]
FORMAT = config["generate"]["img_ext"]

flags = tf.app.flags
tf.flags.DEFINE_string(
    "image_dir", "scripts_generate/data", "Training image directory."
)
tf.flags.DEFINE_string("output_dir", "model_data/records", "Output data directory.")
tf.flags.DEFINE_bool("clf", "False", "Create classification records")
tf.flags.DEFINE_bool("det", "False", "Create detection records")

FLAGS = flags.FLAGS

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def parse_annotation_data(data):
    labels = []
    data = data.replace("\r", "").strip()
    for line in data.split("\n"):
        labels.append([float(val) for val in line.split(" ")])
    return labels


def create_tf_example(image_path_prefix, image_dir):
    """Converts image and txt annotations to a tf.Example proto.
    """
    image_id = os.path.basename(image_path_prefix)
    filename = image_path_prefix + "." + FORMAT

    with tf.io.gfile.GFile(filename, "rb") as fid:
        encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)

    image = Image.open(encoded_img_io)
    image_width, image_height = image.size
    
    annotations = []
    has_target = 1

    if os.path.exists(image_path_prefix + ".txt"):
        # For object detection
        with open(image_path_prefix + ".txt", "r") as annotations_fp:
            annotations = parse_annotation_data(annotations_fp.read())
    else:
        # For clf
        has_target = 0 if "target" not in image_path_prefix else 1

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    category_names = []
    category_ids = []

    for idx, (obj_id, x_n, y_n, w_n, h_n) in enumerate(annotations):

        xmin.append(x_n)
        xmax.append(x_n + w_n)
        ymin.append(y_n)
        ymax.append(y_n + h_n)

        category_id = int(obj_id)
        category_ids.append(category_id + 1)
        category_names.append(CLASSES[category_id].encode("utf8"))

    feature_dict = {
        "image/height": dataset_util.int64_feature(image_height),
        "image/width": dataset_util.int64_feature(image_width),
        "image/filename": dataset_util.bytes_feature(filename.encode("utf8")),
        "image/source_id": dataset_util.bytes_feature(image_id.encode("utf8")),
        "image/encoded": dataset_util.bytes_feature(encoded_img),
        "image/colorspace": dataset_util.bytes_feature("RGB".encode("utf8")),
        "image/format": dataset_util.bytes_feature(FORMAT.encode("utf8")),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmin),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmax),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymin),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymax),
        "image/object/class/text": dataset_util.bytes_list_feature(category_names),
        "image/object/class/label": dataset_util.int64_list_feature(category_ids),
        "image/class/label": dataset_util.int64_feature(has_target),
        "image/class/text": dataset_util.bytes_feature(
            CLF_CLASSES[has_target].encode("utf8")
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    return example


def _create_tf_record_from_images(data_dir, output_path):
    """Loads images generated by generate/*.py scripts and converts
    them into tf records.
    """
    # Determine number of shards. Recommended ~2000 images per shard
    image_fns = glob.glob(os.path.join(data_dir, "*." + FORMAT))
    random.shuffle(image_fns)

    num_shards = len(image_fns) // 2000 + 1

    with contextlib2.ExitStack() as tf_record_close_stack:

        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards
        )

        for idx, image_fn in enumerate(image_fns):
            if idx % 100 == 0:
                tf.compat.v1.logging.info("On image %d of %d", idx, len(image_fns))

            image_path_prefix = image_fn.replace("." + FORMAT, "")
            tf_example = create_tf_example(image_path_prefix, data_dir)
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())


def main(_):

    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # If neither input supplied, do both det and clf
    if not FLAGS.clf and not FLAGS.det:
        FLAGS.clf = True
        FLAGS.det = True

    if FLAGS.det:
        
        _create_tf_record_from_images(
            os.path.join(FLAGS.image_dir, "detector_train", "images"),
            os.path.join(FLAGS.output_dir, "tfm_train.record"),
        )

        _create_tf_record_from_images(
            os.path.join(FLAGS.image_dir, "detector_val", "images"),
            os.path.join(FLAGS.output_dir, "tfm_val.record"),
        )

    if FLAGS.clf:
        _create_tf_record_from_images(
            os.path.join(FLAGS.image_dir, "clf_train", "images"),
            os.path.join(FLAGS.output_dir, "tfm_clf_train.record"),
        )

        _create_tf_record_from_images(
            os.path.join(FLAGS.image_dir, "clf_val", "images"),
            os.path.join(FLAGS.output_dir, "tfm_clf_val.record"),
        )


if __name__ == "__main__":
    tf.compat.v1.app.run()
