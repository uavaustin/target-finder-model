"""Entrypoint for the target_finder_model library.

This module contains the filenames used for target-finder so they can
be encapsulated in a single python library that can be fetched.
"""

from pkg_resources import resource_filename
import yaml
import os

from .version import __version__

# Config
config_fn = resource_filename(__name__, os.path.join('data', 'config.yaml'))
with open(config_fn, 'r') as stream:
    config = yaml.safe_load(stream)

# Model Classes
CLASSES = config['classes']['shapes'] + config['classes']['alphas']