import os
import time
from typing import List

# Limit TF logs:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow.python.compiler.tensorrt
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework import convert_to_constants
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

        saved_model_loaded = tf.saved_model.load(
            self.model_path, tags=[tag_constants.SERVING]
        )
        self.saved_model = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]

    def predict(self, input_data, batch_size=32):

        if len(input_data) == 0:
            return []
        elif isinstance(input_data, list):
            # allow list of paths as input
            input_data = np.array([np.asarray(fn) for fn in input_data], dtype=np.uint8)

        num_imgs, _, _, _ = input_data.shape

        results: List[DetectedObject] = []

        for idx in range(0, num_imgs, batch_size):
            input_data_tensor = tf.convert_to_tensor(input_data[idx : idx + batch_size])

            preds = self.saved_model(input_data_tensor)
            for i in range(len(preds["detection_classes"].numpy())):
                image_detects = []
                for j in range(int(preds["num_detections"][i].numpy())):
                    obj = DetectedObject()
                    obj.class_idx = int(preds["detection_classes"][i][j].numpy())
                    obj.class_name = OD_CLASSES[obj.class_idx - 1]
                    obj.confidence = float(preds["detection_scores"][i][j].numpy())
                    y1, x1, y2, x2 = list(preds["detection_boxes"][i][j].numpy())
                    obj.x = int(x1 * DET_SIZE[0])
                    obj.y = int(y1 * DET_SIZE[1])
                    obj.width = int((x2 - x1) * DET_SIZE[0])
                    obj.height = int((y2 - y1) * DET_SIZE[1])
                    image_detects.append(obj)
                results.append(image_detects)

        return results


class ClfModel:
    def __init__(self, model_path=None) -> None:
        if model_path is None:
            self.model_path = CLF_MODEL_PATH
        else:
            self.model_path = model_path

    def load(self) -> None:

        saved_model_loaded = tf.saved_model.load(
            self.model_path, tags=[tag_constants.SERVING]
        )
        graph_func = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]
        self.saved_model = convert_to_constants.convert_variables_to_constants_v2(
            graph_func
        )

    def predict(self, input_data, batch_size=40):

        if len(input_data) == 0:
            return []

        results: List[DetectedObject] = []
        assert len(input_data.shape) == 4  # (batch_size, h, w, channel)
        num_imgs, _, _, _ = input_data.shape

        for idx in range(0, num_imgs, batch_size):
            data_batch = (
                # NOTE This must change with model
                tf.keras.applications.mobilenet_v2.preprocess_input(
                    tf.convert_to_tensor(
                        input_data[idx : idx + batch_size], dtype=tf.float32
                    )
                )
            )
            res = self.saved_model(data_batch)

            for i in range(res[0].shape[0]):
                obj = DetectedObject()
                obj.class_idx = 1 if res[0][i] > 0.9 else 0
                results.append(obj)

        return results


@dataclass
class DetectedObject:
    class_idx: int = 0
    class_name: str = "unk"
    confidence: float = 0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
