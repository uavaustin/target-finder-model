#!/usr/bin/env python3

import multiprocessing
import os.path
import sys

from pull_assets import pull_all
from create_full_images import generate_all_shapes
from create_detection_data import convert_data as det_convert_data
from create_clf_data import convert_data as clf_convert_data


if __name__ == '__main__':
    pull_all()
    generate_all_shapes('train', 5)
    generate_all_shapes('val', 5)
    det_convert_data('train', 5)
    clf_convert_data('val', 5)
