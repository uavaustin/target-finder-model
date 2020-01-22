import os
import time
# Limit TF logs:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.python.compiler.tensorrt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from pkg_resources import resource_filename
from dataclasses import dataclass
from PIL import Image
import numpy as np

from . import OD_CLASSES, OD_MODEL_PATH, CLF_MODEL_PATH, DET_SIZE


class DetectionModel:

    def __init__(self, model_path=None):
        # Choose between default and custom model
        if model_path is None:
            self.model_path = OD_MODEL_PATH
        else:
            self.model_path = model_path

    def load(self):

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=tf_config)

        try:
            saved_model_loaded = tf.saved_model.load(
                self.model_path, tags=[tag_constants.SERVING])
            self.graph = saved_model_loaded.signatures[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        except:
            frozen_graph = tf.compat.v1.GraphDef()
            with open(self.model_path, 'rb') as f:
                frozen_graph.ParseFromString(f.read())
            tf.import_graph_def(frozen_graph, name='')

    def predict(self, input_data, batch_size=4):

        if len(input_data) == 0:
            return []
        elif isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array(
                [np.asarray(fn) for fn in input_data], dtype=np.uint8)
        else:
            pass

        num_imgs, im_width, im_height, _ = input_data.shape

        results = []
        if num_imgs < batch_size:
            batch_size = num_imgs

        for idx in range(batch_size, num_imgs + batch_size, batch_size):
            input_data = tf.convert_to_tensor(
                input_data[(idx - batch_size):idx])
            preds = self.graph(input_data)

            for i in range(len(preds['detection_classes'].numpy())):
                image_detects = []
                for j in range(int(preds['num_detections'][i].numpy())):
                    obj = DetectedObject()
                    obj.class_idx =  \
                        int(preds['detection_classes'][i][j].numpy())
                    obj.class_name = OD_CLASSES[obj.class_idx - 1]
                    obj.confidence = \
                        float(preds['detection_scores'][i][j].numpy())
                    y1, x1, y2, x2 = \
                        list(preds['detection_boxes'][i][j].numpy())
                    obj.x = int(x1 * DET_SIZE[0])
                    obj.y = int(y1 * DET_SIZE[1])
                    obj.width = int((x2 - x1) * DET_SIZE[0])
                    obj.height = int((y2 - y1) * DET_SIZE[1])
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

        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=tf_config)

        try:
            saved_model_loaded = tf.saved_model.load(
                self.model_path, tags=[tag_constants.SERVING])
            self.graph = saved_model_loaded.signatures[
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        except:
            frozen_graph = tf.compat.v1.GraphDef()
            with open(self.model_path, 'rb') as f:
                frozen_graph.ParseFromString(f.read())
            tf.import_graph_def(frozen_graph, name='')

            self.graph = tf.compat.v1.get_default_graph()

    def predict(self, input_data, batch_size=40):

        if len(input_data) == 0:
            return []
        elif isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array(
                [np.asarray(fn) / 255 for fn in input_data], dtype=np.float32)

        else:
            pass

        results = []
        assert len(input_data.shape) == 4  # (batch_size, h, w, channel)
        num_imgs, im_width, im_height, _ = input_data.shape

        if num_imgs < batch_size:
            batch_size = num_imgs

        for idx in range(batch_size, num_imgs + batch_size, batch_size):
            input_data = tf.convert_to_tensor(
                input_data[(idx - batch_size):idx])

            preds = (self.graph(input_data)['prediction']).numpy()

            for i in range(len(preds)):
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
