# Changelog

## v1.0.0 (2020-01-27)

### Features 

- Switch model framework to TensorFlow to allow for model optimizations with TensorRT. 
- Using Faster-RCNN with Retinanet50 backbone and Inception-v3 pre-classification network. 

### Breaking Changes

- This model is _*not*_ compatible with `v0.3.0` since the deep learning framework has changed.

## v0.3.0-dev.1 (2019-04-21)

### Features 

- Introduce Darknet based deep learning models.
- Using combination of object detection and classification. 
- Improvements to color classification. 

### Breaking Changes

- This model is _*not*_ compatible with `v0.2.0` since these models were built with Darknet.

## v0.2.0 (2018-10-14)

### Features

- Model built with new shape generation system.
- Shape generation is deterministic.
- Improved alphanumeric fonts, placement, and sizing.

### Breaking Changes

- This model is _*not*_ compatible with `v0.1.0` since this model was built
  with a newer version of the `retrain.py` script for the Inception V3 model.

### Chores

- Fetch assets hosted on Bintray.
- Parallelize the shape generation in threads.
- Build on the model on CircleCI to speed up build times.

## v0.1.0 (2018-08-02)

Initial release.
