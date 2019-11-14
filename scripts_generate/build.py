#!/usr/bin/env python3

import multiprocessing
import os.path
import sys

from pull_assets import pull_all
from create_detection_data import generate_all_images as create_det_data
from create_clf_data import create_clf_images as create_clf_data
from create_orientation_data import generate_all_images as create_ori_data


if __name__ == '__main__':

    pull_all()

    create_det_data('detector_train', 5)
    create_det_data('detector_val', 5)

    create_clf_data('detector_train', 5)
    create_clf_data('detector_val', 5)

    create_orientation_data('orientation_data', 5)
    create_orientation_data('orientation_data', 5)