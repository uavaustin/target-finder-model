"""
Object detection inference API
"""
import os
import time
# Limit TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from pkg_resources import resource_filename
from dataclasses import dataclass
import nets.nets_factory
# see https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information


from PIL import Image
import numpy as np

from . import CLASSES, OD_MODEL_PATH, CLF_MODEL_PATH
from . import optimize_od

from .optimize_od import optimize as optimize_od

class DetectionModel:
    
    def __init__(self, model_path=None):
        # Choose between default and custom model
        if model_path is None:
            self.model_path = OD_MODEL_PATH
        else:
            self.model_path = model_path

    def load(self, use_trt=False):

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        frozen_graph = tf.compat.v1.GraphDef()
        with open(self.model_path, 'rb') as f:
            frozen_graph.ParseFromString(f.read())

        if use_trt:
            optimized_frozen_graph = optimize_od(frozen_graph)
            tf.import_graph_def(optimized_frozen_graph, name='')
        else:
            tf.import_graph_def(frozen_graph, name='')

        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(config=tf_config)

    def predict(self, input_data, batch_size=4):

        if isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array([np.asarray(fn) for fn in input_data])

        #assert len(input_data.shape) == 4  # (num_imgs, height, width, channel)
        num_imgs, im_width, im_height, _ = input_data.shape

        output_tensors = [
            self.graph.get_tensor_by_name('num_detections:0'),
            self.graph.get_tensor_by_name('detection_classes:0'),
            self.graph.get_tensor_by_name('detection_boxes:0'),
            self.graph.get_tensor_by_name('detection_scores:0')
        ]

        results = []
        for idx in range(batch_size, num_imgs, batch_size):

            [nums, obj_types, boxes, scores] = self.sess.run(output_tensors, feed_dict={
                'image_tensor:0': input_data[(idx - batch_size):idx]
            })

            for i in range(batch_size):
                image_detects = []
                for k in range(int(nums[i])):
                    obj = DetectedObject()
                    obj.class_idx = int(obj_types[i][k])
                    obj.class_name = CLASSES[obj.class_idx - 1]
                    obj.confidence = scores[i][k]
                    bbox = boxes[i][k]
                    obj.x = int(bbox[0] * im_width)
                    obj.y = int(bbox[1] * im_height)
                    obj.width = int((bbox[3] - bbox[1]) * im_width)
                    obj.height = int((bbox[2] - bbox[0]) * im_height)
                    image_detects.append(obj)
                results.append(image_detects)

        return results


class ClfModel:
    
    def __init__(self, model_path=None):
        if model_path is None:
            self.model_path = CLF_MODEL_PATH
        else:
            self.model_path = model_path

    def load(self, use_trt=False):

        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(graph = graph, config=tf_config)
        
        self.tf_output = 'prefix/classes:0'

    def predict(self, input_data, batch_size=40):

        if isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array([np.asarray(fn) for fn in input_data])

        assert len(input_data.shape) == 4  # (batch_size, height, width, channel)
        num_imgs, im_width, im_height, _ = input_data.shape

        for idx in range(batch_size, num_imgs, batch_size):

            [preds] = self.sess.run([self.tf_output], feed_dict={
                'prefix/input:0': input_data[(idx - batch_size):idx]
            })
            
            results = []
            for i in range(batch_size):
                obj = DetectedObject()
                obj.class_idx = preds[i]
                results.append(obj)

        return results


@dataclass
class DetectedObject:
    class_idx: int = 0
    class_name: str = 'unk'
    confidence: float = 0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
