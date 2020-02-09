# ADAPTED FROM:
# https://github.com/tensorflow/tensorrt/blob/master/tftrt/examples/object_detection/object_detection.py
# =============================================================================
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import argparse
import os
import sys
import glob
import logging
import time
from functools import partial
import json
import numpy as np
import subprocess

# Limit TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with open(
    os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"), "r"
) as stream:
    import yaml

    config = yaml.safe_load(stream)

CLASSES = config["classes"]["shapes"] + config["classes"]["alphas"]
NUM_VAL_IMAGES = config["generate"]["eval_batch"]["images"]

image_width = config["inputs"]["detector"]["width"]
image_height = config["inputs"]["detector"]["height"]


def get_annotations(records_dir):
    """
    Gets COCO format annotations from records.
    """
    full_annotation = {"type": "instances"}
    # Get categories
    categories = []
    for idx, class_type in enumerate(CLASSES):
        temp_category = {"supercategory": "none", "name": class_type, "id": str(idx)}
        categories.append(temp_category)

    full_annotation["categories"] = categories

    coco_labels = []
    images = []

    filenames = glob.glob(os.path.join(records_dir, "tfm_val*"))
    raw_dataset = tf.data.TFRecordDataset(filenames)

    image_feature_description = {
        "image/source_id": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/object/bbox/xmin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(dtype=tf.int64),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.

        record = tf.io.parse_single_example(example_proto, image_feature_description)
        record["image/encoded"] = tf.image.decode_image(
            record["image/encoded"], channels=3, dtype=tf.dtypes.uint8
        )
        return record

    dataset = raw_dataset.map(_parse_image_function, num_parallel_calls=8)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.enumerate(0)
    dataset = dataset.shuffle(NUM_VAL_IMAGES)

    for i, data in dataset:
        img = {
            "file_name": str(data["image/source_id"].numpy()),
            "height": int(data["image/height"].numpy()),
            "width": int(data["image/width"].numpy()),
            "id": int(i.numpy()),
            "coco_url": "n/a",
            "date_captured": "2000-01-01 00:00:00",
        }
        images.append(img)

        for j in range(0, data["image/object/class/label"].shape[0]):
            x1 = data["image/object/bbox/xmin"].values[j].numpy()
            y1 = data["image/object/bbox/ymin"].values[j].numpy()
            x2 = data["image/object/bbox/xmax"].values[j].numpy()
            y2 = data["image/object/bbox/ymax"].values[j].numpy()

            bbox_coco_fmt = [
                x1 * data["image/width"].numpy(),  # x0
                y1 * data["image/height"].numpy(),  # x1
                (x2 - x1) * data["image/width"].numpy(),  # width
                (y2 - y1) * data["image/height"].numpy(),  # height
            ]
            coco_label = {
                "id": int(j),
                "image_id": int(i.numpy()),
                "category_id": int(data["image/object/class/label"].values[j].numpy()),
                "bbox": [int(coord) for coord in bbox_coco_fmt],
                "iscrowd": 0,
                "area": float(
                    (x2 - x1) * data["image/width"].numpy() *  
                    (y2 - y1) * data["image/height"].numpy()),
            }
            coco_labels.append(coco_label)

    full_annotation["images"] = images
    full_annotation["annotations"] = coco_labels

    tmp_dir = "tmp_annotations"
    subprocess.call(["mkdir", "-p", tmp_dir])
    coco_annotations_path = os.path.join(tmp_dir, "coco_labels.json")
    with open(coco_annotations_path, "w") as f:
        json.dump(full_annotation, f)

    coco = COCO(annotation_file=coco_annotations_path)
    subprocess.call(["rm", "-r", tmp_dir])

    return coco


def get_dataset(records_dir, batch_size, input_size):
    """
    Produces a dataset from TFRecords.
    """
    filenames = glob.glob(os.path.join(records_dir, "tfm_val*"))
    raw_dataset = tf.data.TFRecordDataset(filenames)

    image_feature_description = {
        "image/source_id": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/object/bbox/xmin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(dtype=tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(dtype=tf.int64),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.

        record = tf.io.parse_single_example(example_proto, image_feature_description)
        record["image/encoded"] = tf.image.decode_image(
            record["image/encoded"], channels=3, dtype=tf.dtypes.uint8
        )
        return record

    dataset = raw_dataset.map(_parse_image_function, num_parallel_calls=8)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(1)
    dataset = dataset.enumerate(0)
    coco = get_annotations(records_dir)

    return dataset, coco


def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING]
    )
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    return graph_func


def get_graph_func(
    input_saved_model_dir,
    data_dir,
    calib_data_dir,
    input_size,
    output_saved_model_dir=None,
    conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
    use_trt=False,
    num_calib_inputs=None,
    batch_size=None,
    optimize_offline=False,
):
    """Retreives a frozen SavedModel and applies TF-TRT
    use_trt: bool, if true use TensorRT
    precision: str, floating point precision (FP32, FP16, or INT8)
    batch_size: int, batch size for TensorRT optimizations
    returns: TF function that is ready to run for inference
    """
    start_time = time.time()
    graph_func = get_func_from_saved_model(input_saved_model_dir)
    if use_trt:
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=input_saved_model_dir,
            conversion_params=conversion_params,
        )

        def input_fn(input_data_dir, num_iterations):
            dataset, _ = get_dataset(
                records_dir=input_data_dir, batch_size=batch_size, input_size=input_size
            )

            for i, batch_images in enumerate(dataset):
                if i >= num_iterations:
                    break
                yield (batch_images,)
                print("  step %d/%d" % (i + 1, num_iterations))
                i += 1

        if conversion_params.precision_mode != "INT8":
            print("Graph conversion...")
            converter.convert()
            if optimize_offline:
                print("Building TensorRT engines...")
                converter.build(input_fn=partial(input_fn, data_dir, 1))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
        else:
            print("Graph conversion and INT8 calibration...")
            converter.convert(
                calibration_input_fn=partial(
                    input_fn, calib_data_dir, num_calib_inputs // batch_size
                )
            )
            if optimize_offline:
                print("Building TensorRT engines...")
                converter.build(input_fn=partial(input_fn, data_dir, 1))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
    return graph_func, {"conversion": time.time() - start_time}


def run_inference(
    graph_func,
    data_dir,
    batch_size,
    input_size,
    num_iterations,
    num_warmup_iterations,
    display_every=100,
    mode="validation",
    target_duration=None,
):
    """Run the given graph_func on the data files provided. In validation mode,
    it consumes TFRecords with labels and reports accuracy. In benchmark mode,
    it times inference on real data (.jpgs).
    """
    # Delete the ouputs not needed
    nodes = [
        "detection_boxes",
        "detection_classes",
        "detection_scores",
        "num_detections",
    ]

    results = {}
    predictions = {}
    iter_times = []
    initial_time = time.time()

    dataset_full, coco = get_dataset(
        records_dir=data_dir, batch_size=batch_size, input_size=input_size
    )
    if mode == "validation":
        # Process the dataset batch and calc time
        for i, dataset_batch in dataset_full:
            start_time = time.time()
            batch_preds = graph_func(dataset_batch["image/encoded"])
            end_time = time.time()
            iter_times.append(end_time - start_time)

            # Add the outputs to the prediction map
            for key in list(batch_preds.keys()):
                if key not in nodes:
                    del batch_preds[key]
                    continue
                if key not in predictions:
                    predictions[key] = [batch_preds[key].numpy()]
                else:
                    predictions[key].append(batch_preds[key].numpy())

            if i.numpy() % display_every == 0:
                print(
                    "step {}/{}, iter_time(ms)={}".format(
                        i.numpy() + 1,
                        NUM_VAL_IMAGES // batch_size,
                        iter_times[-1] * 1000,
                    )
                )

            if (
                i.numpy() > 1
                and target_duration is not None
                and time.time() - initial_time > target_duration
            ):
                break

    elif mode == "benchmark":
        for i, dataset_batch in dataset_full:
            if i.numpy() >= num_warmup_iterations:
                start_time = time.time()
                batch_preds = list(graph_func(dataset_batch["image/encoded"]).values())[
                    0
                ].numpy()
                iter_times.append(time.time() - start_time)
                if i.numpy() % display_every == 0:
                    print(
                        "step {}/{}, iter_time(ms)={}".format(
                            i.numpy() + 1, num_iterations, iter_times[-1] * 1000
                        )
                    )
            else:
                batch_preds = list(graph_func(dataset_batch["image/encoded"]).values())[
                    0
                ].numpy()
            if (
                i > 0
                and target_duration is not None
                and time.time() - initial_time > target_duration
            ):
                break
            if num_iterations is not None and i >= num_iterations:
                break
            i += 1

    if not iter_times:
        return results
    iter_times = np.array(iter_times)
    iter_times = iter_times[num_warmup_iterations:]
    results["total_time"] = np.sum(iter_times)
    results["images_per_sec"] = np.mean(batch_size / iter_times)
    results["99th_percentile"] = (
        np.percentile(iter_times, q=99, interpolation="lower") * 1000
    )
    results["latency_mean"] = np.mean(iter_times) * 1000
    results["latency_median"] = np.median(iter_times) * 1000
    results["latency_min"] = np.min(iter_times) * 1000
    return results, predictions, coco


def eval_model(predictions, coco, batch_size):

    coco_detections = []
    num_batches = len(predictions["detection_classes"])
    # Loop over all the image batches processed
    img = 0
    for i in range(num_batches):
        for j in range(int(batch_size)):
            for k in range(int(predictions["num_detections"][i][j])):
                bbox = predictions["detection_boxes"][i][j][k]
                y1, x1, y2, x2 = list(bbox)
                bbox_coco_fmt = [
                    x1 * image_width,  # x0
                    y1 * image_height,  # x1
                    (x2 - x1) * image_width,  # width
                    (y2 - y1) * image_height,  # height
                ]
                coco_detection = {
                    "image_id": int(img),
                    "category_id": int(predictions["detection_classes"][i][j][k]),
                    "bbox": [int(coord) for coord in bbox_coco_fmt],
                    "score": float(predictions["detection_scores"][i][j][k]),
                }
                coco_detections.append(coco_detection)
            img += 1

    # write coco detections to file
    tmp_dir = "tmp_detection_results"
    subprocess.call(["mkdir", "-p", tmp_dir])
    coco_detections_path = os.path.join(tmp_dir, "coco_detections.json")
    with open(coco_detections_path, "w") as f:
        json.dump(coco_detections, f)

    cocoDt = coco.loadRes(coco_detections_path)
    subprocess.call(["rm", "-r", tmp_dir])

    # compute coco metrics
    eval = COCOeval(coco, cocoDt, "bbox")
    eval.params.imgIds = list(range(0, NUM_VAL_IMAGES))

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return eval.stats[0]


def config_gpu_memory(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        return
    print("Found the following GPUs:")
    for gpu in gpus:
        print("  ", gpu)
    for gpu in gpus:
        try:
            if not gpu_mem_cap:
                tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_mem_cap
                        )
                    ],
                )
        except RuntimeError as e:
            print("Can not set GPU memory config", e)


def get_trt_conversion_params(
    max_workspace_size_bytes, precision_mode, minimum_segment_size, max_batch_size
):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=max_workspace_size_bytes
    )
    conversion_params = conversion_params._replace(precision_mode=precision_mode)
    conversion_params = conversion_params._replace(
        minimum_segment_size=minimum_segment_size
    )
    conversion_params = conversion_params._replace(
        use_calibration=precision_mode == "INT8"
    )
    conversion_params = conversion_params._replace(max_batch_size=max_batch_size)
    return conversion_params


if __name__ == "__main__":
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    assert (
        config["inputs"]["detector"]["width"] == config["inputs"]["detector"]["height"]
    )

    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--input_saved_model_dir",
        type=str,
        default=None,
        help="Directory containing the input saved model.",
    )
    parser.add_argument(
        "--output_saved_model_dir",
        type=str,
        default=None,
        help="Directory in which the converted model is saved",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=config["inputs"]["detector"]["width"],
        help="Size of input images expected by the model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing validation set" "TFRecord files.",
    )
    parser.add_argument(
        "--calib_data_dir",
        type=str,
        help="Directory containing TFRecord files for" "calibrating INT8.",
    )
    parser.add_argument(
        "--use_trt",
        action="store_true",
        help="If set, the graph will be converted to a" "TensorRT graph.",
    )
    parser.add_argument(
        "--optimize_offline",
        action="store_true",
        help="If set, TensorRT engines are built" "before runtime.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["FP32", "FP16", "INT8"],
        default="FP32",
        help="Precision mode to use. FP16 and INT8 only"
        "work in conjunction with --use_trt",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of images per batch."
    )
    parser.add_argument(
        "--minimum_segment_size",
        type=int,
        default=2,
        help="Minimum number of TF ops in a TRT engine.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=500,
        help="How many iterations(batches) to evaluate."
        "If not supplied, the whole set will be evaluated.",
    )
    parser.add_argument(
        "--display_every",
        type=int,
        default=100,
        help="Number of iterations executed between"
        "two consecutive display of metrics",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=1,
        help="Number of initial iterations skipped from timing",
    )
    parser.add_argument(
        "--num_calib_inputs",
        type=int,
        default=500,
        help="Number of inputs (e.g. images) used for"
        "calibration (last batch is skipped in case"
        "it is not full)",
    )
    parser.add_argument(
        "--gpu_mem_cap",
        type=int,
        default=0,
        help="Upper bound for GPU memory in MB."
        "Default is 0 which means allow_growth will be used",
    )
    parser.add_argument(
        "--max_workspace_size",
        type=int,
        default=(1 << 30),
        help="workspace size in bytes",
    )
    parser.add_argument(
        "--mode",
        choices=["validation", "benchmark"],
        default="validation",
        help="Which mode to use (validation or benchmark)",
    )
    parser.add_argument(
        "--target_duration",
        type=int,
        default=None,
        help="If set, script will run for specified" "number of seconds.",
    )
    args = parser.parse_args()

    if args.precision != "FP32" and not args.use_trt:
        raise ValueError(
            "TensorRT must be enabled for FP16" "or INT8 modes (--use_trt)."
        )
    if args.precision == "INT8" and not args.calib_data_dir:
        raise ValueError("--calib_data_dir is required for INT8 mode")
    if (
        args.num_iterations is not None
        and args.num_iterations <= args.num_warmup_iterations
    ):
        raise ValueError(
            "--num_iterations must be larger than --num_warmup_iterations "
            "({} <= {})".format(args.num_iterations, args.num_warmup_iterations)
        )
    if args.num_calib_inputs < args.batch_size:
        raise ValueError(
            "--num_calib_inputs must not be smaller than --batch_size"
            "({} <= {})".format(args.num_calib_inputs, args.batch_size)
        )
    if args.use_trt and not args.output_saved_model_dir:
        raise ValueError("--output_saved_model_dir must be set if use_trt=True")

    config_gpu_memory(args.gpu_mem_cap)

    get_annotations(args.data_dir)

    params = get_trt_conversion_params(
        args.max_workspace_size,
        args.precision,
        args.minimum_segment_size,
        args.batch_size,
    )
    graph_func, times = get_graph_func(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        data_dir=args.data_dir,
        calib_data_dir=args.calib_data_dir,
        input_size=args.input_size,
        conversion_params=params,
        use_trt=args.use_trt,
        batch_size=args.batch_size,
        num_calib_inputs=args.num_calib_inputs,
        optimize_offline=args.optimize_offline,
    )

    def print_dict(input_dict, prefix="  ", postfix=""):
        for k, v in sorted(input_dict.items()):
            print(
                "{}{}: {}{}".format(
                    prefix, k, "%.1f" % v if isinstance(v, float) else v, postfix
                )
            )

    print("Benchmark arguments:")
    print_dict(vars(args))
    print("TensorRT Conversion Params:")
    print_dict(dict(params._asdict()))
    print("Conversion times:")
    print_dict(times, postfix="s")

    results, predictions, coco = run_inference(
        graph_func,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        input_size=args.input_size,
        display_every=args.display_every,
        mode=args.mode,
        target_duration=args.target_duration,
    )

    print("Results:")
    if args.mode == "validation":
        mAP = eval_model(predictions, coco, args.batch_size)
        print("  mAP: %f" % mAP)
    print("  images/sec: %d" % results["images_per_sec"])
    print("  99th_percentile(ms): %.2f" % results["99th_percentile"])
    print("  total_time(s): %.1f" % results["total_time"])
    print("  latency_mean(ms): %.2f" % results["latency_mean"])
    print("  latency_median(ms): %.2f" % results["latency_median"])
    print("  latency_min(ms): %.2f" % results["latency_min"])
