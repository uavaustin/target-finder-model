# Target Finder Model

> Related scripts and models for [target-finder](https://github.com/uavaustin/target-finder).

## Developer Instructions

### Install
1. Download `git clone https://github.com/uavaustin/target-finder-model && cd target-finder-model`
2. Install `pip install -r requirements-dev.txt`

### Generate Training Data
* `python scripts_generate/pull_assets.py` Download base shapes and background images
* `python scripts_generate/create_full_images.py` Create full-sized artificial images
* `python scripts_generate/create_detection_data.py` Convert full-sized images to training data for detection model

### Training
1. In a seperate folder `git clone https://github.com/tensorflow/models.git`
2. Run with `MODEL_NAME` set to one of the models in `models/`
```
python path/to/models/research/object_detection/model_main.py \
    --pipeline_config_path=models/MODEL_NAME/pipeline.config \
    --model_dir=models/MODEL_NAME \
    --num_train_steps=5000 \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr
```

## Testing

`TODO: Write tests`