# Target Finder Model

> Related scripts and models for [target-finder](https://github.com/uavaustin/target-finder).

## Developer Instructions

### Install
1. Download `git clone https://github.com/uavaustin/target-finder-model && cd target-finder-model`
2. Install `pip install -r requirements-dev.txt` (requires CUDA installation)

### Training

#### Generate Data
* `python scripts_generate/pull_assets.py` Download base shapes and background images
* `python scripts_generate/create_full_images.py` Create full-sized artificial images
* `python scripts_generate/create_detection_data.py` Convert full-sized images to training data for detection model
* `python scripts_tf/create_tf_records.py --image_dir ./scripts_generate/data --output_dir ./model_data` Reformat training files

#### Train Model
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
#### Freeze Model 
This create a `frozen_inference_graph.pb` that will be optimized with tensorrt.
```
python path/to/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=models/MODEL_NAME/pipeline.config \
    --trained_checkpoint_prefix=models/MODEL_NAME/model.ckpt \
    --output_directory=models/MODEL_NAME/frozen
```
## Repository Contents

#### Model Files
* `model_data/` Object detection training data, filled w/generate scripts
* `model/` Model configs, TensorFlow pipelines, training checkpoints

#### Scripts
* `scripts_generate/` Scripts for generating artifical training data
* `scripts_tf/` Helpful TensorFlow utilities

#### Misc
* `.circleci/` CI Config for automated builds and testing
* `Dockerfiles/` Docker resources for creating a prebuilt ML environment
* `target_finder_model/` The package that will be exported to [target-finder](https://github.com/uavaustin/target-finder) when a release is created.


## Testing

`TODO: Write tests`
