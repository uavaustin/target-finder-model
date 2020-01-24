"""Testing that the files can be accessed and are non-empty."""

import target_finder_model as tfm


def test_constants():
    """Test constants packaged with tfm"""
    assert tfm.CROP_SIZE[0] == tfm.CROP_SIZE[1]
    assert tfm.CROP_OVERLAP < tfm.CROP_SIZE[0]

    assert tfm.DET_SIZE[0] == tfm.DET_SIZE[1]
    assert tfm.CLF_SIZE[0] == tfm.CLF_SIZE[1]

    assert len(tfm.OD_CLASSES) == 37
    assert len(tfm.CLF_CLASSES) == 2
