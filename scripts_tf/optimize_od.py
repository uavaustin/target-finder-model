"""
FIRST to freeze the graph from training checkpoints.

python scripts_tf/optimize.py --frozen_model model/frozen_graph.pb
                              --output_dir ./models/frcnn.pb

"""

import os
import argparse
import time
import tensorflow as tf
from tftrt.examples.object_detection import optimize_model

INPUT_NAME = 'image_tensor'
BOXES_NAME = 'detection_boxes'
CLASSES_NAME = 'detection_classes'
SCORES_NAME = 'detection_scores'
NUM_DETECTIONS_NAME = 'num_detections'


def load_graph(args):
    # Read in frozen graph
    frozen_graph = tf.GraphDef()
    with open(args.frozen_model, 'rb') as f:
        frozen_graph.ParseFromString(f.read())

    optimize(frozen_graph, args.output_path)


def optimize(frozen_graph, output_path=None):

    optimized_frozen_graph = optimize_model(
        frozen_graph,
        use_trt=True,
        precision_mode="FP32",
        calib_images_dir="/data/coco-2017/train2017",
        num_calib_images=8,
        calib_batch_size=8,
        calib_image_shape=[640, 640],
        max_workspace_size_bytes=17179869184,
        output_path=output_path
    )

    return optimized_frozen_graph


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model",
                        type=str,
                        default="",
                        help="Model folder to export")
    parser.add_argument("--output_path",
                        type=str,
                        default="model_data/frozen.pb",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    load_graph(args)
