#!/bin/bash

echo "Running pycodestyle."
pycodestyle --exclude=vendor --ignore=E402 target_finder_model generate scripts_tf test

echo "Building data."
python scripts_generate/build.py

echo "Creating TF-Records."
python scripts_tf/create_tf_records.py

cp -r model_data/records /host/model_data

echo "Testing model API."
pytest --cov=target_finder_model test

if [ $1 == "coveralls" ]; then
    echo "Submitting coveralls."
    coveralls
fi

echo "Creating label map."
python scripts_tf/create_label_map.py

echo "Training classification model."
python scripts_tf/train_clf.py
# This command will fail on CI builds without gpu ()
if [ $? -eq 0 ]; then
    echo OK
else
    echo "Clf training script exit. Likely due to no gpu accessible."
fi

echo "Training detection model."
python /sources/models/research/object_detection/model_main.py \
    --pipeline_config_path=models/faster_rcnn_resnet50_coco_2018_01_28/pipeline.config \
    --model_dir=models/faster_rcnn_resnet50_coco_2018_01_28/checkpoints/ \
    --num_train_steps=5 \
    --sample_1_of_n_eval_examples=1 
# This command will fail on CI builds without gpu ()
if [ $? -eq 0 ]; then
    echo OK
else
    echo "Clf training script exit. Likely due to no gpu accessible."
fi