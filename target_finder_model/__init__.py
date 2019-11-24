"""Entrypoint for the target_finder_model library.

This module contains the filenames used for target-finder so they can
be encapsulated in a single python library that can be fetched.
"""

from pkg_resources import resource_filename
import yaml
import os

from .version import __version__

# Config
CONFIG_FN = resource_filename(__name__, os.path.join('data', 'config.yaml'))
with open(CONFIG_FN, 'r') as stream:
    CONFIG = yaml.safe_load(stream)

# Builtin Saved Model
CLF_MODEL_PATH = resource_filename(__name__, os.path.join('data', 'clf.pb'))
OD_MODEL_PATH = resource_filename(__name__, os.path.join('data', 'det.pb'))

# Model Classes
CLASSES = []
for shape in CONFIG['classes']['shapes']:
    for alpha in CONFIG['classes']['alphas']:
        CLASSES.append('-'.join([shape, alpha]))

# Submodules
from . import inference