#!/bin/bash

pycodestyle --exclude=vendor target_finder_model generate scripts_tf test

python scripts_generate/build.py

python test/test_models.py

pytest --cov=target_finder_model test

cp .coverage* /host