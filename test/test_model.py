"""Testing that the files can be accessed and are non-empty."""

import os

import target_finder_model


WEIGHTS_PATH = os.path.join('model', 'weights')
YOLO_WEIGHTS = os.path.join(WEIGHTS_PATH, 'yolo3detector-train_final.weights')
CLF_WEIGHTS = os.path.join(WEIGHTS_PATH, 'preclf-train_final.weights')


def test_model():
    assert os.path.exists(YOLO_WEIGHTS)
    assert os.path.exists(CLF_WEIGHTS)
