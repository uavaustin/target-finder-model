#!/bin/bash

echo "Running pycodestyle."
pycodestyle --exclude=vendor --ignore=E402 target_finder_model generate scripts_tf test

echo "Building data."
python scripts_generate/build.py

echo "Creating TF-Records."
python scripts_tf/create_tf_records.py
mv model_data/records /host/model_data

echo "Testing model API."
pytest --cov=target_finder_model test

if [ $1 == "coveralls" ]; then
    echo "Submitting coveralls."
    coveralls
fi