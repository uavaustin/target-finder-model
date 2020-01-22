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

import argparse
import os
import glob
import logging
import time
import pprint
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

with open(
    os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml"), "r"
) as stream:
    import yaml

    config = yaml.safe_load(stream)


CLF_SIZE = config["inputs"]["preclf"]["width"]
CLF_CLASSES = config["classes"]["types"]
NUM_IMGS = config["generate"]["eval_batch"]["images"]


def get_dataset(data_files, batch_size, input_size, mode="validation"):

    image_feature_description = {
        "image/source_id": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.

        record = tf.io.parse_single_example(example_proto, image_feature_description)
        record["image/encoded"] = tf.image.decode_image(
            record["image/encoded"], channels=3, dtype=tf.dtypes.int8
        )
        # Inception requires image data / 255
        record["image/encoded"] = tf.cast(
            record["image/encoded"], dtype=tf.dtypes.float32
        )
        record["image/encoded"] = tf.divide(record["image/encoded"], 255.0)
        return record

    if mode == "validation":
        dataset = tf.data.TFRecordDataset(data_files)
        dataset = dataset.map(map_func=_parse_image_function, num_parallel_calls=8)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(count=1)
        dataset = dataset.enumerate(0)
    elif mode == "benchmark":
        dataset = tf.data.TFRecordDataset(data_files)
        dataset = dataset.map(map_func=_parse_image_function, num_parallel_calls=8)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(count=1)
        dataset = dataset.enumerate(0)
    else:
        raise ValueError("Mode must be either 'validation' or 'benchmark'")
    return dataset


def get_func_from_saved_model(saved_model_dir):
    saved_model_loaded = tf.saved_model.load(
        saved_model_dir, tags=[tag_constants.SERVING]
    )
    graph_func = saved_model_loaded.signatures[signature_constants.PREDICT_OUTPUTS]
    graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
    return graph_func


def get_graph_func(
    input_saved_model_dir,
    input_size,
    output_saved_model_dir=None,
    conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
    use_trt=False,
    calib_files=None,
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

        def input_fn(input_files, num_iterations):
            dataset = get_dataset(
                data_files=input_files,
                batch_size=batch_size,
                input_size=input_size,
                mode="validation",
            )
            for i, batch_images in dataset:
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
                converter.build(input_fn=partial(input_fn, data_files, 1))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
        else:
            print("Graph conversion and INT8 calibration...")
            converter.convert(
                calibration_input_fn=partial(
                    input_fn, calib_files, num_calib_inputs // batch_size
                )
            )
            if optimize_offline:
                print("Building TensorRT engines...")
                converter.build(input_fn=partial(input_fn, data_files, 1))
            converter.save(output_saved_model_dir=output_saved_model_dir)
            graph_func = get_func_from_saved_model(output_saved_model_dir)
    return graph_func, {"conversion": time.time() - start_time}


def eval_fn(preds, labels):
    """Measures number of correct predicted labels in a batch.
       Assumes preds and labels are numpy arrays.
    """
    return np.sum((labels == preds).astype(np.float32))


def run_inference(
    graph_func,
    data_files,
    batch_size,
    input_size,
    num_classes,
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
    results = {}
    corrects = 0
    iter_times = []
    initial_time = time.time()
    dataset = get_dataset(
        data_files=data_files, batch_size=batch_size, input_size=input_size, mode=mode
    )
    # TODO The print of num_iterations and time statement seems to be off
    if mode == "validation":
        for i, batch_images in dataset:
            start_time = time.time()
            batch_preds = graph_func(batch_images["image/encoded"])[0].numpy()
            end_time = time.time()
            iter_times.append(end_time - start_time)
            if i % display_every == 0:
                print(
                    "  step %d/%d, iter_time(ms)=%.0f"
                    % (i + 1, NUM_IMGS // batch_size, iter_times[-1] * 1000)
                )
            corrects += eval_fn(batch_preds, batch_images["image/class/label"].numpy())
            if (
                i > 1
                and target_duration is not None
                and time.time() - initial_time > target_duration
            ):
                break
            accuracy = corrects / (batch_size * i)
            results["accuracy"] = accuracy

    elif mode == "benchmark":
        for i, batch_images in dataset:
            if i >= num_warmup_iterations:
                start_time = time.time()
                batch_preds = graph_func(batch_images["image/encoded"])[0].numpy()
                iter_times.append(time.time() - start_time)
                if i % display_every == 0:
                    print(
                        "  step %d/%d, iter_time(ms)=%.0f"
                        % (i + 1, num_iterations, iter_times[-1] * 1000)
                    )
            else:
                batch_preds = graph_func(batch_images["image/encoded"])[0].numpy()
            if (
                i > 0
                and target_duration is not None
                and time.time() - initial_time > target_duration
            ):
                break
            if num_iterations is not None and i >= num_iterations:
                break

    if not iter_times:
        return results

    results["total_time"] = sum(iter_times)
    avg_img_sec = [batch_size / t for t in iter_times]
    results["images_per_sec"] = sum(avg_img_sec) / len(avg_img_sec)
    return results


def config_gpu_memory(gpu_mem_cap):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        return
    print("Found the following GPUs:")
    for gpu in gpus:
        print(" ", gpu)
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
        default=CLF_SIZE,
        help="Size of input images expected by the model",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=len(CLF_CLASSES),
        help="Number of classes used when training the model",
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
        default=None,
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
        default=100,
        help="Minimum number of TF ops in a TRT engine.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=2048,
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
        default=50,
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
        "Default is 0 which means allow_growth will be used.",
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

    params = get_trt_conversion_params(
        args.max_workspace_size,
        args.precision,
        args.minimum_segment_size,
        args.batch_size,
    )
    graph_func, times = get_graph_func(
        input_saved_model_dir=args.input_saved_model_dir,
        output_saved_model_dir=args.output_saved_model_dir,
        input_size=args.input_size,
        conversion_params=params,
        use_trt=args.use_trt,
        calib_files=args.calib_data_dir,
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

    # Get the data and calibration files
    data_files = glob.glob(os.path.join(args.data_dir, "tfm_clf_val*"))
    calib_files = []
    if args.calib_data_dir:
        calib_files = glob.glob(os.path.join(args.data_dir, "tfm_clf_val*"))

    results = run_inference(
        graph_func,
        data_files=data_files,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        input_size=args.input_size,
        num_classes=args.num_classes,
        display_every=args.display_every,
        mode=args.mode,
        target_duration=args.target_duration,
    )
    if args.mode == "validation":
        print("  accuracy:", results["accuracy"].numpy() * 100)
        print("  images/sec:", results["images_per_sec"])
        print("  total_time(s):", results["total_time"])
