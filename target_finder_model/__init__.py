"""Entrypoint for the target_finder_model library.

This module contains the filenames used for target-finder so they can
be encapsulated in a single python library that can be fetched.
"""

from pkg_resources import resource_filename
import os

from .version import __version__

# Config
CONFIG_FN = resource_filename(__name__, os.path.join('data', 'config.yaml'))
with open(CONFIG_FN, 'r') as stream:
    import yaml
    CONFIG = yaml.safe_load(stream)

CLF_MODEL_PATH = resource_filename(
    __name__, os.path.join('data', 'optimized-clf'))
OD_MODEL_PATH = resource_filename(
    __name__, os.path.join('data', 'optimized-det'))

# Model Classes
OD_CLASSES = CONFIG['classes']['shapes'] + CONFIG['classes']['alphas']

CLF_CLASSES = CONFIG['classes']['types']

DET_SIZE = (CONFIG['inputs']['detector']['width'],
            CONFIG['inputs']['detector']['height'])

# Submodules
from . import inference
