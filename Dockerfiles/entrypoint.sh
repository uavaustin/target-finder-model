#!/bin/bash

pycodestyle --exclude=vendor --ignore=E402 target_finder_model generate scripts_tf test

python scripts_generate/build.py

python scripts_tf/create_tf_records.py
mv model_data/records /host/model_data

python test/test_models.py

pytest --cov=target_finder_model test
coveralls

ls -a
mv .coverage* /host