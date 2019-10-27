"""
Object detection inference API
"""
from pkg_resources import resource_filename
from dataclasses import dataclass
import tensorflow as tf
from PIL import Image
import numpy as np
import os

from . import CLASSES, MODEL_PATH


class DetectionModel:
    
    def __init__(self, model_path=None):
        # Choose between default and custom model
        if model_path is None:
            self.model_path = MODEL_PATH
        else:
            self.model_path = model_path

    def load(self):
        self.sess = tf.compat.v1.Session()
        tf.compat.v1.saved_model.load(self.sess, ['serve'], self.model_path)
        self.graph = tf.compat.v1.get_default_graph()

    def predict(self, input_data):
        if isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array([np.asarray(Image.open(fn)) for fn in input_data])
        assert len(input_data.shape) == 4  # (batch_size, height, width, channel)
        batch_size, im_width, im_height, _ = input_data.shape

        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
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
                obj.confidence = scores[i][k]
                bbox = boxes[i][k]
                obj.x = bbox[0] * im_width
                obj.y = bbox[1] * im_height
                obj.width = (bbox[3] - bbox[1]) * im_width
                obj.height = (bbox[2] - bbox[0]) * im_height
                image_detects.append(obj)
            results.append(image_detects)

        return results


@dataclass
class DetectedObject:
    class_idx: int = 0
    class_name: str = 'unk'
    confidence: float = 0
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0