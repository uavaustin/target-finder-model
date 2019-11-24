"""
python scripts_tf/optimize.py --frozen_model ./models/faster_rcnn_resnet50_coco_2018_01_28/output/frozen_inference_graph.pb --output_dir ./models/frcnn.pb

FIRST to freeze the graph from training checkpoint, run:

python /sources/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=/host/mounted/models/faster_rcnn_resnet50_coco_2018_01_28/pipeline.config \
    --trained_checkpoint_prefix=/host/mounted/models/faster_rcnn_resnet50_coco_2018_01_28/checkpoints/model.ckpt-11300 \
    --output_directory=/host/mounted/models/faster_rcnn_resnet50_coco_2018_01_28/frozen


"""
import os, argparse, time

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tftrt.examples.object_detection.graph_utils import force_nms_cpu as f_force_nms_cpu
from tftrt.examples.object_detection.graph_utils import replace_relu6 as f_replace_relu6
from tftrt.examples.object_detection.graph_utils import remove_assert as f_remove_assert

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
    """
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    graph_size = len(frozen_graph.SerializeToString())
    num_nodes = len(frozen_graph.node)
    start_time = time.time()

    frozen_graph = f_force_nms_cpu(frozen_graph)
    frozen_graph = f_replace_relu6(frozen_graph)
    frozen_graph = f_remove_assert(frozen_graph)

    converter = trt.TrtGraphConverter(
        input_graph_def=frozen_graph,
        nodes_blacklist=output_names,
        max_workspace_size_bytes=1 << 32,
        precision_mode='FP32',
        minimum_segment_size=2,
        is_dynamic_op=False,
        maximum_cached_engines=100)
    frozen_graph_out = converter.convert()

    end_time = time.time()
    print("graph_size(MB)(native_tf): %.1f" % (float(graph_size)/(1<<20)))
    print("graph_size(MB)(trt): %.1f" %
        (float(len(frozen_graph_out.SerializeToString()))/(1<<20)))
    print("num_nodes(native_tf): %d" % num_nodes)
    print("num_nodes(tftrt_total): %d" % len(frozen_graph.node))
    print("num_nodes(trt_only): %d" % len([1 for n in frozen_graph_out.node if str(n.op)=='TRTEngineOp']))
    print("time(s) (trt_conversion): %.4f" % (end_time - start_time))

    # write optimized model to disk
    if output_path is not None:
        with open(output_path, 'wb') as f:
            f.write(frozen_graph_out.SerializeToString())
    """
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
    parser.add_argument("--frozen_model", type=str, default="", help="Model folder to export")
    parser.add_argument("--output_path", type=str, default="model_data/frozen.pb", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    load_graph(args)
