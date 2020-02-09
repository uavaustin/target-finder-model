#!/usr/bin/env python

import os
from setuptools import setup, find_packages


def join(*paths):
    """Join and normalize several paths.
    Args:
        *paths (List[str]): The paths to join and normalize.
    Returns:
        str: The normalized path.
    """
    return os.path.normpath(os.path.join(*paths))


VERSION_PATH = join(__file__, '..', 'target_finder_model', 'version.py')


def get_version():
    """Get the version number without running version.py.
    Returns:
        str: The current uavaustin-target-finder version.
    """
    with open(VERSION_PATH, 'r') as version:
        out = {}
        exec(version.read(), out)
        return out['__version__']


setup(
    name='target-finder-model',
    version=get_version(),
    author='UAV Austin',
    url='https://github.com/uavaustin/target-finder-model',
    packages=find_packages(),
    install_requires=[
        "numpy==1.18.1",
        "pyyaml==5.3",
        "dataclasses==0.7",
        "Pillow==7.0.0",
    ],
    package_data={
        'target_finder_model': [
            'data/config.yaml',
            'data/det/saved_model.pb',
            'data/clf/saved_model.pb',
            'data/clf/variables/variables.data-00000-of-00001',
            'data/clf/variables/variables.index'
        ]
    },
    license='MIT'
)
