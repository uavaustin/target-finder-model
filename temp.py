import tensorflow as tf
import pathlib
import numpy as np
from PIL import Image
import os

x = Image.open('temp.jpg')
x = np.asarray(x)

model_dir = os.path.join('models', 'faster_rcnn_resnet50_coco_2018_01_28', 'saved_model')
graph = tf.Graph()
with graph.as_default():
  with tf.compat.v1.Session() as sess:
    model = tf.compat.v1.saved_model.load(
      sess,
      ['serve'],
      model_dir,
    )
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')

    out = sess.run([num_detections, detection_scores, detection_classes], feed_dict={
      'image_tensor:0': np.array([x])
    })
    print(out)