"""
Object detection inference API
"""
from pkg_resources import resource_filename
from dataclasses import dataclass
import nets.nets_factory
import tensorflow as tf
from PIL import Image
import numpy as np
import os

from . import CLASSES, MODEL_PATH, CLF_MODEL_PATH


class DetectionModel:
    
    def __init__(self, model_path=None):
        # Choose between default and custom model
        if model_path is None:
            self.model_path = MODEL_PATH
        else:
            self.model_path = model_path

    def load(self):

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        frozen_graph = tf.compat.v1.GraphDef()
        with open(self.model_path, 'rb') as f:
            frozen_graph.ParseFromString(f.read())

        tf.import_graph_def(frozen_graph, name='')

        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(config=tf_config)

    def predict(self, input_data):
        if isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array([np.asarray(Image.open(fn)) for fn in input_data])
        assert len(input_data.shape) == 4  # (batch_size, height, width, channel)
        batch_size, im_width, im_height, _ = input_data.shape

        output_tensors = [
            self.graph.get_tensor_by_name('num_detections:0'),
            self.graph.get_tensor_by_name('detection_classes:0'),
            self.graph.get_tensor_by_name('detection_boxes:0'),
            self.graph.get_tensor_by_name('detection_scores:0')
        ]

        [nums, obj_types, boxes, scores] = self.sess.run(output_tensors, feed_dict={
            'image_tensor:0': input_data
        })

        results = []
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

    def load(self):

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(config=tf_config)

        # TODO unhardcode these constants
        tf_input = tf.placeholder(tf.float32, [None, 299, 299, 3], name='image_tensor')
        network_fn = nets.nets_factory.get_network_fn('inception_v3', 1001, is_training=False)
        tf_net, tf_end_points = network_fn(tf_input)
        self.tf_output = tf_end_points

        tf_saver = tf.train.Saver()
        tf_saver.restore(save_path=self.model_path, sess=self.sess)

    def predict(self, input_data):
        if isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array([np.asarray(Image.open(fn).resize((299, 299))) for fn in input_data])
        assert len(input_data.shape) == 4  # (batch_size, height, width, channel)
        batch_size = input_data.shape[0]

        [net_out] = self.sess.run([self.tf_output], feed_dict={
            'image_tensor:0': input_data
        })

        pred = net_out['Predictions']

        results = []
        for i in range(batch_size):
            obj = DetectedObject()
            obj.class_idx = np.argmax(pred[i])
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
