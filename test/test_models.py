"""Testing that the files can be accessed and are non-empty."""

import os
import glob
from PIL import Image
import target_finder_model as tfm


def test_constants():
    """Test constants packaged with tfm"""

    OVERLAP = tfm.CONFIG['inputs']['cropping']['overlap']

    CROP_WIDTH = tfm.CONFIG['inputs']['cropping']['width']
    CROP_HEIGHT = tfm.CONFIG['inputs']['cropping']['height']

    CLF_WITH = tfm.CONFIG['inputs']['preclf']['width']
    CLF_HEIGHT = tfm.CONFIG['inputs']['preclf']['height']

    DET_WIDTH = tfm.CONFIG['inputs']['detector']['width']
    DET_HEIGHT = tfm.CONFIG['inputs']['detector']['height']

    assert CROP_WIDTH == CROP_HEIGHT
    assert OVERLAP < CROP_WIDTH

    assert DET_WIDTH == DET_HEIGHT
    assert CLF_WITH == CLF_HEIGHT

    assert len(tfm.OD_CLASSES) > 0
    assert len(tfm.CLF_CLASSES) > 0


def test_models():
    """Test loading the models"""

    models = {
        'frcnn': tfm.inference.DetectionModel(),
        'clf': tfm.inference.ClfModel()
    }

    models['frcnn'].load()
    models['clf'].load()

    """Test model inference"""

    detector_model = models['frcnn']
    clf_model = models['clf']

    clf_imgs = glob.glob('test/clf/*.png')
    od_imgs = glob.glob('test/od/*.png')

    regions = clf_model.predict([Image.open(img) for img in clf_imgs])
    out = detector_model.predict([Image.open(img) for img in od_imgs])
