"""Testing that the files can be accessed and are non-empty."""

import os

import target_finder_model as tfm


WEIGHTS_PATH = os.path.join('model', 'weights')
YOLO_WEIGHTS = os.path.join(WEIGHTS_PATH, 'yolo3detector-train_final.weights')
CLF_WEIGHTS = os.path.join(WEIGHTS_PATH, 'preclf-train_final.weights')


def test_constants():
    """Test constants packaged with tfm"""
    assert tfm.CROP_SIZE[0] == tfm.CROP_SIZE[1]
    assert tfm.CROP_OVERLAP < tfm.CROP_SIZE[0]

    assert tfm.DETECTOR_SIZE[0] == tfm.DETECTOR_SIZE[1]
    assert tfm.PRECLF_SIZE[0] == tfm.PRECLF_SIZE[1]


def test_weights_exist():
    """Ensure weights were produced"""
    assert os.path.exists(YOLO_WEIGHTS)
    assert os.path.exists(CLF_WEIGHTS)
