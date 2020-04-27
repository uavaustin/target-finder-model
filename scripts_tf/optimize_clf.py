#!/usr/bin/env python3

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
from typing import List, Dict
import argparse
import pathlib
import logging
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python import saved_model
from tensorflow.python.framework import convert_to_constants

with pathlib.Path("config.yaml").open("r") as stream:
    import yaml

    config = yaml.safe_load(stream)

CLF_SIZE = config["inputs"]["preclf"]["width"]
CLF_CLASSES = config["classes"]["types"]
NUM_IMGS = config["generate"]["eval_batch"]["images"]


def parse_image_function(example_proto: tf.Tensor) -> dict:
    """ Parse a tf proto. """
    # A dictionary which details the information wanted from the record.
    image_feature_description = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
    }

    record = tf.io.parse_single_example(example_proto, image_feature_description)
    record["image/encoded"] = tf.image.decode_image(
        record["image/encoded"], channels=3, dtype=tf.dtypes.float32
    )
    record["image/encoded"] = tf.keras.applications.mobilenet_v2.preprocess_input(
        record["image/encoded"]
    )
    return record


def get_dataset(
    data_files: List[pathlib.Path], batch_size: int
) -> tf.data.TFRecordDataset:
    """ Function to create a TF Record dataset. 
    Args:
        data_files: List of the paths to record files.
        batch_size: batch size of loader data.
    
    Returns:
        A TF dataset. 
    """

    dataset = tf.data.TFRecordDataset([str(path) for path in data_files])
    dataset = dataset.map(map_func=parse_image_function, num_parallel_calls=5)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(-1)
    dataset = dataset.enumerate(0)

    return dataset


def get_func_from_saved_model(
    input_model_dir: pathlib.Path,
) -> "tf.python.eager.wrap_function.WrappedFunction":
    """ Load the SavedModel proto.
    
    Args:
        input_model_dir: The directory containing the SavedModel to convert.

    Returns:
        A frozen model for conversion. 
    """

    saved_model_loaded = tf.saved_model.load(
        input_model_dir, tags=[saved_model.tag_constants.SERVING]
    )
    graph_func = saved_model_loaded.signatures[
        saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)

    return graph_func


def get_graph_func(
    input_model_dir: pathlib.Path,
    output_model_dir: pathlib.Path,
    data_files: List[pathlib.Path],
    conversion_params=trt.DEFAULT_TRT_CONVERSION_PARAMS,
    batch_size=None,
) -> Dict["tf.python.eager.wrap_function.WrappedFunction", float]:
    """ Retreives a SavedModel and applies TF-TRT conversions.
    
    Args:
        input_model_dir: Path to SavedModel to load.
        output_model_dir: Where to save the converted model.
        data_files: List of the TF-Record data files.

    Returns:
        The optimized graph and the time taken for conversion.
    """
    start_time = time.perf_counter()
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=str(input_model_dir), conversion_params=conversion_params,
    )

    print("Graph conversion...")
    converter.convert()
    converter.save(output_saved_model_dir=str(output_model_dir))
    graph_func = get_func_from_saved_model(str(output_model_dir))

    return graph_func, {"conversion": time.perf_counter() - start_time}


def run_inference(
    graph_func: "tf.python.eager.wrap_function.WrappedFunction",
    data_files: List[pathlib.Path],
    batch_size: int,
    num_iterations: int,
    display_every: int = 100,
    classifcation_score: float = 0.1,
) -> dict:
    """ Run the given graph_func on the data files provided. Do some warm up loops 
    so prime the gpu. 
    
    Args:
        graph_func: The optimized grapg:
        data_files: List of the TF-Record files.
        batch_size: The batch size of the dataset.
        num_iterations: The number of batches to process before stopping inference.
        display_every: Frequency of logging.
        classifcation_score: The score threshold when judging classifications.

    Returns:
        A dictionary with information from inference.
    """

    iter_times = []
    num_correct = 0
    start_time_total = time.perf_counter()

    # Get the dataset loader
    dataset = get_dataset(data_files=data_files, batch_size=batch_size)

    for idx, batch_images in dataset:

        if idx > num_iterations:
            break

        start_time = time.perf_counter()
        batch_preds = graph_func(batch_images["image/encoded"])[0].numpy()
        end_time = time.perf_counter()
        iter_times.append(end_time - start_time)

        if idx % display_every == 0:
            print(f"  step {idx}/{NUM_IMGS}, iter_time(ms)={iter_times[-1] * 1000:.3}")

        preds = np.where(batch_preds > classifcation_score, 1, 0)
        num_correct += np.sum(
            preds.squeeze(1) == batch_images["image/class/label"].numpy()
        )

    results = {
        "accuracy": num_correct / (batch_size * idx),
        "total_time": start_time_total,
        "images_per_sec": np.average([batch_size / t for t in iter_times]),
    }

    return results


def get_trt_conversion_params(
    max_workspace_size_bytes: int,
    precision_mode: str,
    minimum_segment_size: int,
    max_batch_size: int,
) -> trt.TrtConversionParams:
    """ Collate the conversion parameters. """

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(
        max_workspace_size_bytes=max_workspace_size_bytes
    )
    conversion_params = conversion_params._replace(precision_mode=precision_mode)
    conversion_params = conversion_params._replace(
        minimum_segment_size=minimum_segment_size
    )
    # No INT8 calibration for the time being.
    conversion_params = conversion_params._replace(
        use_calibration=precision_mode == "INT8"
    )
    conversion_params = conversion_params._replace(max_batch_size=max_batch_size)

    return conversion_params


if __name__ == "__main__":
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Convert model to fp16. "
        "Note, FP16 converted models only will see benefits "
        "on Volta and Turning gpus (Jetson Xavier). "
    )
    parser.add_argument(
        "--input_model_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing the input saved model.",
        required=True,
    )
    parser.add_argument(
        "--output_model_dir",
        type=pathlib.Path,
        help="Directory in which the converted model is saved",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        default="model_data/records",
        help="Directory containing validation set" "TFRecord files.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Number of images per batch."
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
        "--max_workspace_size",
        type=int,
        default=(1 << 30),
        help="workspace size in bytes",
    )
    parser.add_argument(
        "--display_every",
        type=int,
        default=100,
        help="Fequency of logging while inferencing.",
    )
    parser.add_argument(
        "--classifcation_score",
        type=float,
        default=0.1,
        help="Confidence threshold during the benchmark.",
    )
    args = parser.parse_args()

    # Allow gpu-mem growth
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Get the data and calibration files
    data_files = list(args.data_dir.glob("tfm_clf_val*"))

    params = get_trt_conversion_params(
        args.max_workspace_size, "FP16", args.minimum_segment_size, args.batch_size,
    )

    input_dir = args.input_model_dir.expanduser()
    assert input_dir.is_dir(), f"Could not find {input_dir}."

    output_dir = args.output_model_dir.expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)

    graph_func, conversion_time = get_graph_func(
        input_model_dir=input_dir,
        output_model_dir=output_dir,
        data_files=data_files,
        conversion_params=params,
        batch_size=args.batch_size,
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
    print_dict(conversion_time, postfix="s")

    results = run_inference(
        graph_func=graph_func,
        data_files=data_files,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        display_every=args.display_every,
        classifcation_score=args.classifcation_score,
    )

    print(f"  accuracy: {results['accuracy'].numpy() * 100:.3}")
    print(f"  images/sec: {results['images_per_sec']:.3}")
    print(f"  total_time(s): {results['total_time']:.3}")
